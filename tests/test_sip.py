import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mapprojax.projections import Tan
from mapprojax.sip import Sip

try:
    from astropy.wcs import WCS as AstroWCS
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

@pytest.mark.skipif(not HAS_ASTROPY, reason="Astropy not installed")
def test_sip_against_astropy():
    """
    Verify SIP implementation against astropy.wcs.
    """
    # 1. Create a Header with SIP coefficients
    # Using a simple example.
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 100
    header['NAXIS2'] = 100
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    header['CRPIX1'] = 50.0
    header['CRPIX2'] = 50.0
    header['CRVAL1'] = 180.0
    header['CRVAL2'] = 45.0
    header['CD1_1'] = -0.00028 # ~1 arcsec/pix
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = 0.00028
    
    # Forward distortion (Pix -> Foc)
    # u' = u + f(u,v), v' = v + g(u,v)
    # f(u,v) = A_2_0 * u^2 + A_0_2 * v^2
    header['A_ORDER'] = 2
    header['A_2_0'] = 1.5e-6
    header['A_0_2'] = 2.5e-6
    
    # g(u,v) = B_1_1 * u*v
    header['B_ORDER'] = 2
    header['B_1_1'] = 3.0e-6
    
    # Reverse distortion (Foc -> Pix)
    # u = U + F(U,V), v = V + G(U,V)
    # For simplicity, use arbitrary values, though in reality they should be inverses.
    # We test that our implementation matches Astropy's calculation for these coefficients.
    header['AP_ORDER'] = 2
    header['AP_2_0'] = -1.5e-6
    header['AP_0_2'] = -2.5e-6
    
    header['BP_ORDER'] = 2
    header['BP_1_1'] = -3.0e-6
    
    # 2. Create Astropy WCS
    awcs = AstroWCS(header)
    
    # 3. Create Mapprojax WCS
    # Extract SIP coeffs
    a = np.zeros((3, 3))
    a[2, 0] = header['A_2_0']
    a[0, 2] = header['A_0_2']
    
    b = np.zeros((3, 3))
    b[1, 1] = header['B_1_1']
    
    ap = np.zeros((3, 3))
    ap[2, 0] = header['AP_2_0']
    ap[0, 2] = header['AP_0_2']
    
    bp = np.zeros((3, 3))
    bp[1, 1] = header['BP_1_1']
    
    sip = Sip(a=a, b=b, ap=ap, bp=bp)
    
    cd = [[header['CD1_1'], header['CD1_2']], 
          [header['CD2_1'], header['CD2_2']]]
    crpix = [header['CRPIX1'] - 1.0, header['CRPIX2'] - 1.0]
    crval = [header['CRVAL1'], header['CRVAL2']]
    
    mwcs = Tan(crpix, cd, crval, sip=sip)
    
    # 4. Generate test points (pixels)
    # Avoid 0,0 to avoid singularities if any
    x = np.array([10.0, 50.0, 90.0, 50.0])
    y = np.array([10.0, 50.0, 90.0, 10.0])
    
    # 5. Test Pix -> Sky (Forward)
    # astropy: all_pix2world (1-based origin if using default, but we can pass 0)
    # Wait, astropy pix2world takes 0-based or 1-based depending on arg.
    # By default, wcs.all_pix2world(x, y, 0) uses 0-based.
    
    ra_astro, dec_astro = awcs.all_pix2world(x, y, 0)
    
    # mapprojax: proj expects radians input? No, proj expects RA/DEC in radians.
    # We want pix -> sky.
    # mapprojax.unproj(x, y) -> returns ra, dec in radians.
    
    ra_map_rad, dec_map_rad = mwcs.unproj(x, y)
    ra_map = np.degrees(ra_map_rad)
    dec_map = np.degrees(dec_map_rad)
    
    np.testing.assert_allclose(ra_map, ra_astro, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(dec_map, dec_astro, rtol=1e-10, atol=1e-8)
    
    # 6. Test Sky -> Pix (Inverse)
    # Using the sky coordinates we just got
    
    x_astro, y_astro = awcs.all_world2pix(ra_astro, dec_astro, 0)
    
    # mapprojax: proj(ra, dec) -> returns x, y
    x_map, y_map = mwcs.proj(np.radians(ra_astro), np.radians(dec_astro))
    
    np.testing.assert_allclose(x_map, x_astro, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_map, y_astro, rtol=1e-5, atol=1e-5)

def test_sip_manual_check():
    """
    Test SIP logic manually without Astropy to ensure basic correctness of polynomials.
    """
    # Simple case:
    # u' = u + 0.1 * u^2
    # v' = v
    a = np.zeros((3, 3))
    a[2, 0] = 0.1
    sip = Sip(a=a)
    
    # Pixel (10, 0) relative to CRPIX (0,0)
    # u = 10, v = 0
    # u' = 10 + 0.1 * 100 = 20
    # v' = 0
    
    res_u, res_v = sip.pix_to_foc(np.array([10.0]), np.array([0.0]))
    assert res_u[0] == pytest.approx(20.0)
    assert res_v[0] == pytest.approx(0.0)

def test_no_sip():
    """Test that sip=None behaves as standard WCS."""
    crpix = [50.0, 50.0]
    cd = [[-0.001, 0.0], [0.0, 0.001]] 
    crval = [180.0, 0.0]
    
    wcs = Tan(crpix, cd, crval, sip=None)
    
    # Just check it runs and produces expected linear result
    x = 60.0 # u = 10
    y = 50.0 # v = 0
    # u = 10 -> X = -0.01 deg
    
    ra_rad, dec_rad = wcs.unproj(x, y)
    ra = np.degrees(ra_rad)
    
    # At equator, RA should increase to left.
    # X = -0.01 deg -> RA = 180 - 0.01 = 179.99
    assert ra == pytest.approx(179.99)



# --- inverse: Newton's method against the forward polynomial ---------------

def _realistic_sip(seed=0):
    """Order-3 A/B plus approximate AP/BP, as in a real SIP header."""
    rng = np.random.default_rng(seed)
    a = np.zeros((4, 4))
    b = np.zeros((4, 4))
    a[2, 0], a[0, 2], a[1, 1], a[3, 0] = 1.5e-5, 2.5e-5, 1.0e-5, 2.0e-7
    b[0, 2], b[2, 0], b[1, 1] = -1.8e-5, 1.2e-5, 0.7e-5
    return Sip(a=a, b=b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_inverse_round_trip_through_forward(seed):
    """pix -> foc -> pix recovers the pixel for the synthetic case."""
    sip = _realistic_sip(seed)
    pix = np.array([0.0, 5.0, -20.0, 40.0, -45.0, 33.0])
    v = np.array([0.0, -3.0, 12.0, -30.0, 25.0, -41.0])
    U, V = sip.pix_to_foc(pix, v)
    x, y = sip.foc_to_pix(U, V)
    np.testing.assert_allclose(x, pix, rtol=0, atol=1e-9)
    np.testing.assert_allclose(y, v, rtol=0, atol=1e-9)


def test_ab_only_inverse_has_no_ap_bp_to_fall_back_on():
    """With A/B and no AP/BP the identity seed must still converge.

    Before the Newton solve this returned the input unchanged, a 1e-3 px
    error on a synthetic case and 1.6 px on a real one.
    """
    sip = _realistic_sip()
    pix = np.array([30.0, -25.0, 12.0])
    v = np.array([-18.0, 33.0, -7.0])
    U, V = sip.pix_to_foc(pix, v)
    x, y = sip.foc_to_pix(U, V)  # sip.ap is None
    np.testing.assert_allclose(x, pix, rtol=0, atol=1e-9)
    np.testing.assert_allclose(y, v, rtol=0, atol=1e-9)


def test_two_steps_already_reach_the_fp64_floor():
    """A solver that drops the Jacobian fails this; the plan's test.

    The field is sized so that convergence is comfortably inside the
    residual tolerance at step 2; the point is the quadratic rate, not the
    extreme corner of a synthetic order-3 field.
    """
    sip = _realistic_sip()
    pix = np.linspace(0.0, 50.0, 25)
    U, V = sip.pix_to_foc(pix, pix)
    two = Sip(a=sip.a, b=sip.b, ap=sip.ap, bp=sip.bp, inverse_steps=2)
    x, y = two.foc_to_pix(U, V)
    assert np.isfinite(x).all()
    assert np.abs(x - pix).max() < 1e-10
    assert np.abs(y - pix).max() < 1e-10


def test_inverse_residual_reports_convergence():
    sip = _realistic_sip()
    pix = np.array([25.0, -30.0, 8.0])
    v = np.array([-12.0, 27.0, -20.0])
    U, V = sip.pix_to_foc(pix, v)
    x, y = sip.foc_to_pix(U, V)
    r_u, r_v = sip.inverse_residual(x, y, U, V)
    assert np.abs(r_u).max() < 1e-12
    assert np.abs(r_v).max() < 1e-12


def test_non_convergent_points_return_nan():
    """A deliberately starved solver reports NaN instead of a wrong pixel."""
    sip = _realistic_sip()
    pix = np.array([60.0, -55.0])
    U, V = sip.pix_to_foc(pix, pix)
    starved = Sip(a=sip.a, b=sip.b, ap=sip.ap, bp=sip.bp, inverse_steps=0)
    x, y = starved.foc_to_pix(U, V)
    assert np.isnan(x).all() and np.isnan(y).all()


def test_inverse_steps_survive_serialization():
    sip = _realistic_sip()
    sip = Sip(a=sip.a, b=sip.b, ap=sip.ap, bp=sip.bp, inverse_steps=5)
    back = Sip.from_dict(sip.to_dict())
    assert back.inverse_steps == 5


def test_inverse_is_jittable_and_differentiable():
    sip = _realistic_sip()
    pix = np.array([20.0, -15.0, 5.0])
    U, V = sip.pix_to_foc(pix, pix)

    def f(a):
        return Sip(a=a, b=sip.b, ap=sip.ap, bp=sip.bp).foc_to_pix(U, V)[0].sum()

    jitted = jax.jit(lambda a: Sip(a=a, b=sip.b, ap=sip.ap, bp=sip.bp).foc_to_pix(U, V)[0].sum())
    assert np.allclose(float(jitted(jnp.asarray(sip.a))), float(f(jnp.asarray(sip.a))))
    grad = jax.grad(f)(jnp.asarray(sip.a))
    assert np.all(np.isfinite(np.asarray(grad)))


# --- the item 7 acceptance tests on the real SPHEREx L2 header --------------

def _l2():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import accuracy_reference as ar
    from mapprojax.wcs_utils import load_wcs_params_from_header

    header = ar.load_header("spherex_l2.hdr")
    crpix, cd, crval, sip = load_wcs_params_from_header(header)
    pix = ar.pixel_grid(header) - np.asarray(crpix)  # relative pixels, 33x33 + CRPIX
    return sip, pix


def _with(sip, **kw):
    fields = dict(a=sip.a, b=sip.b, ap=sip.ap, bp=sip.bp, inverse_steps=sip.inverse_steps)
    fields.update(kw)
    return Sip(**fields)


def test_l2_ap_bp_seeded_newton_reaches_the_floor_in_two_steps():
    """The plan's discriminating test: with AP/BP as the seed, two Newton
    steps reach < 1e-10 px over the full 2040x2040 detector. A fixed-point
    iteration without the Jacobian is ~1e-5 px here after two steps."""
    sip, pix = _l2()
    assert sip.ap is not None and sip.bp is not None
    U, V = sip.pix_to_foc(pix[:, 0], pix[:, 1])
    x, y = _with(sip, inverse_steps=2).foc_to_pix(U, V)
    err = np.hypot(x - pix[:, 0], y - pix[:, 1])
    assert np.isfinite(err).all()
    assert err.max() < 1e-10, err.max()


def test_l2_ab_only_inverse_converges_on_the_full_detector():
    """A/B with no AP/BP: identity seed, default steps, full detector (the
    regression case elsewhere is a 100 px field and converges early)."""
    sip, pix = _l2()
    ab = _with(sip, ap=None, bp=None)
    U, V = ab.pix_to_foc(pix[:, 0], pix[:, 1])
    x, y = ab.foc_to_pix(U, V)
    err = np.hypot(x - pix[:, 0], y - pix[:, 1])
    assert np.isfinite(err).all()
    assert err.max() < 1e-6, err.max()


def test_singular_jacobian_root_is_rejected():
    """u - u^2 = U folds at u = 1/2 (Jacobian 1 - 2u = 0). AP/BP seeds the
    solver straight onto that root; it solves the equation but has no
    inverse derivative, so it must come back NaN rather than du/dU = 6."""
    a = np.zeros((3, 3)); a[2, 0] = -1.0
    ap = np.zeros((3, 3)); ap[2, 0] = 4.0
    sip = Sip(a=a, b=np.zeros((3, 3)), ap=ap, bp=np.zeros((3, 3)))
    x, y = sip.foc_to_pix(np.array([0.25]), np.array([0.0]))
    assert np.isnan(x).all() and np.isnan(y).all()


def test_inverse_residual_follows_jax_inputs_under_jit():
    sip = _realistic_sip()
    f = jax.jit(lambda u: sip.inverse_residual(u, u, u, u))  # default xp=np
    r_u, r_v = f(jnp.asarray([3.0, -4.0]))
    assert np.all(np.isfinite(np.asarray(r_u)))


def test_non_default_steps_survive_pytree_and_cast():
    from mapprojax import TanJax
    from mapprojax.reproject import _cast_sip

    sip = _with(_realistic_sip(), inverse_steps=0)  # 0: every point unconverged
    leaves, treedef = jax.tree_util.tree_flatten(sip)
    assert jax.tree_util.tree_unflatten(treedef, leaves).inverse_steps == 0
    assert _cast_sip(sip, jnp.float32).inverse_steps == 0

    # Observable under jit: the starved solver must stay starved.
    U, V = sip.pix_to_foc(np.array([20.0]), np.array([-15.0]))
    x, _ = jax.jit(lambda s: s.foc_to_pix(jnp.asarray(U), jnp.asarray(V)))(sip)
    assert np.isnan(np.asarray(x)).all()

    wcs = TanJax(jnp.zeros(2), jnp.eye(2) * 1e-3, jnp.zeros(2), sip=sip)
    back = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(wcs)))
    assert back.sip.inverse_steps == 0


# --- inverse JVP/VJP: numerical gradient checks --------------------------------

@pytest.fixture
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _inverse_case(kind):
    """``ap_bp_seed``: A/B with a first-order AP/BP = -A/-B as the seed;
    ``identity_seed``: A/B only; ``zero``: all-zero A/B, identity seed."""
    if kind == "zero":
        return Sip(a=np.zeros((4, 4)), b=np.zeros((4, 4)))
    sip = _realistic_sip()
    if kind == "identity_seed":
        return sip
    return _with(sip, ap=-sip.a, bp=-sip.b)


CD0 = 1e-3      # deg/px
SIP_SCALE = 1e-5


@pytest.mark.parametrize("kind", ["ap_bp_seed", "identity_seed", "zero"])
def test_inverse_gradients_match_finite_differences(_x64, kind):
    """``check_grads`` (forward and reverse mode) on the SIP inverse inside
    ``native_to_pix``, with respect to CRPIX, CD and the A/B coefficients.

    The parameters are rescaled to O(1) so one finite-difference step size
    suits all of them (CD is ~1e-3, the SIP coefficients ~1e-5).
    """
    from jax.test_util import check_grads
    from mapprojax import TanJax
    from mapprojax.utils import radec_to_xyz

    sip = _inverse_case(kind)
    sky = jnp.radians(jnp.asarray([[0.01, -0.012, 0.004], [0.006, 0.009, -0.011]]))
    native = radec_to_xyz(sky[0], sky[1], xp=jnp)

    def pixels(crpix, cd_n, a_n, b_n):
        s = _with(sip, a=a_n * SIP_SCALE, b=b_n * SIP_SCALE)
        w = TanJax(crpix, cd_n * CD0, jnp.zeros(2), sip=s)
        x, y = w.native_to_pix(*native)
        return jnp.stack([x, y])

    args = (jnp.asarray([5.0, -3.0]), jnp.asarray([[-1.0, 0.01], [0.02, 1.0]]),
            jnp.asarray(sip.a, dtype=float) / SIP_SCALE, jnp.asarray(sip.b, dtype=float) / SIP_SCALE)
    assert np.all(np.isfinite(np.asarray(pixels(*args))))
    check_grads(pixels, args, order=1, modes=("fwd", "rev"), eps=1e-4, rtol=1e-5, atol=1e-5)
