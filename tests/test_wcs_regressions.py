"""Known-defect regressions tracked against Stage 2 of
``docs/plan/mapprojax improvement plan.md``.

Each test asserts the *desired* behaviour. Cases that fail today carry a
strict ``xfail`` naming their Stage 2 item, so the fix flips them to a pass
(and an unexpected pass fails the suite until the marker is removed).
Behaviour that is already correct is tested without a marker.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from mapprojax import (Reprojector, Sin, SinJax, SinJaxArray, SinJaxArrayMixed,
                        SinJaxMixed, Tan, TanJax, TanJaxArray, TanJaxArrayMixed,
                        TanJaxMixed)
from mapprojax.sip import Sip
from mapprojax.wcs_utils import load_wcs_params_from_header


@pytest.fixture(autouse=True)
def _x64_enabled():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


CD = [[-1e-3, 0.0], [0.0, 1e-3]]

SINGULAR = "Stage 2 singular or non-finite CD: reject concrete CD at construction"
PYTREE = "Stage 2 PyTree reconstruction: tree_unflatten must not call __init__"


def _xfail(reason, raises=AssertionError):
    return pytest.mark.xfail(strict=True, raises=raises, reason=reason)


# --- projection domains -----------------------------------------------------

@pytest.mark.parametrize("cls", [Tan, TanJax, Sin, SinJax], ids=lambda c: c.__name__)
@pytest.mark.parametrize("sep_deg", [135.0, 179.99])
def test_back_hemisphere_is_not_a_valid_pixel(cls, sep_deg):
    """Outside the domain the contract is NaN, not merely "not finite"."""
    wcs = cls([0.0, 0.0], CD, [0.0, 0.0])
    x, y = wcs.proj(np.radians(sep_deg), 0.0)
    assert np.isnan(x) and np.isnan(y), (x, y)


@pytest.mark.parametrize("cls", [Tan, TanJax, Sin, SinJax], ids=lambda c: c.__name__)
def test_mixed_valid_invalid_batch(cls):
    """Validity is per sample: masking one pixel must not spoil the others."""
    wcs = cls([0.0, 0.0], CD, [0.0, 0.0])
    sep = np.array([10.0, 135.0, 20.0, 179.99, 5.0])
    x, y = wcs.proj(np.radians(sep), np.zeros_like(sep))
    good = np.array([True, False, True, False, True])
    assert np.all(np.isfinite(x[good])), x
    assert np.all(np.isnan(x[~good])), x
    assert np.all(np.isnan(y[~good])), y


@pytest.mark.parametrize("cls", [Tan, TanJax, Sin, SinJax], ids=lambda c: c.__name__)
def test_front_hemisphere_is_finite(cls):
    wcs = cls([0.0, 0.0], CD, [0.0, 0.0])
    x, y = wcs.proj(np.radians(10.0), np.radians(-5.0))
    assert np.isfinite(x) and np.isfinite(y)


def test_tan_native_horizon_is_nan():
    # xn == 0 exactly: the tangent plane is at infinity.
    wcs = Tan([0.0, 0.0], CD, [0.0, 0.0])
    x, y = wcs.native_to_pix(np.array(0.0), np.array(1.0), np.array(0.0))
    assert np.isnan(x) and np.isnan(y)


def test_sin_plane_outside_unit_disc_is_nan_and_boundary_is_valid():
    wcs = Sin([0.0, 0.0], CD, [0.0, 0.0])
    r_px = np.degrees(1.0) / CD[1][1]  # plane radius 1 in pixels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        outside = wcs.unproj(np.array(r_px * 1.01), np.array(0.0))
    assert np.all(np.isnan(outside))
    on_edge = wcs.unproj(np.array(r_px * (1.0 - 1e-12)), np.array(0.0))
    assert np.all(np.isfinite(on_edge))


# --- singular / non-finite CD -----------------------------------------------

BAD_CD = {
    "singular": [[1e-3, 0.0], [1e-3, 0.0]],
    "nan": [[np.nan, 0.0], [0.0, 1e-3]],
}


@pytest.mark.parametrize("cls", [Tan, TanJax], ids=lambda c: c.__name__)
@pytest.mark.parametrize("kind", list(BAD_CD))
def test_concrete_bad_cd_is_rejected_at_construction(cls, kind):
    try:
        cls([0.0, 0.0], BAD_CD[kind], [0.0, 0.0])
    except ValueError:
        rejected = True
    else:
        rejected = False
    assert rejected, f"{cls.__name__} accepted a {kind} CD matrix"


@pytest.mark.parametrize("kind", list(BAD_CD))
def test_traced_bad_cd_yields_nan(kind):
    # Traced parameters cannot raise, so the pixel must not come back as a
    # valid finite pair. How far NaN spreads is backend-dependent (a NaN CD
    # poisons both coordinates on ROCm but only x on CPU), so only require
    # that the pair is invalid.
    f = jax.jit(lambda cd: TanJax(jnp.zeros(2), cd, jnp.zeros(2)).proj(1e-3, 1e-3))
    x, y = f(jnp.asarray(BAD_CD[kind]))
    assert not (np.isfinite(x) and np.isfinite(y))


# --- SIP inverse without AP/BP ----------------------------------------------

def _ab_only_header():
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 100, 100
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN-SIP", "DEC--TAN-SIP"
    h["CRPIX1"], h["CRPIX2"] = 50.0, 50.0
    h["CRVAL1"], h["CRVAL2"] = 180.0, 45.0
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -2.8e-4, 0.0, 0.0, 2.8e-4
    h["A_ORDER"], h["B_ORDER"] = 3, 3
    h["A_2_0"], h["A_0_2"], h["B_1_1"], h["A_3_0"] = 1.5e-5, 2.5e-5, 1.0e-5, 2.0e-7
    return h


PIX = np.array([[5.0, 5.0], [30.0, 20.0], [70.0, 80.0], [95.0, 60.0]])


def test_ab_only_forward_matches_astropy():
    h = _ab_only_header()
    crpix, cd, crval, sip = load_wcs_params_from_header(h)
    assert sip.ap is None and sip.bp is None
    ra, dec = WCS(h).all_pix2world(PIX, 0).T
    ra_m, dec_m = Tan(crpix, cd, crval, sip=sip).unproj(PIX[:, 0], PIX[:, 1])
    np.testing.assert_allclose(np.degrees(ra_m), ra, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.degrees(dec_m), dec, rtol=0, atol=1e-10)


def test_ab_only_inverse_matches_astropy():
    h = _ab_only_header()
    crpix, cd, crval, sip = load_wcs_params_from_header(h)
    ra, dec = WCS(h).all_pix2world(PIX, 0).T
    x, y = Tan(crpix, cd, crval, sip=sip).proj(np.radians(ra), np.radians(dec))
    np.testing.assert_allclose(np.column_stack([x, y]), PIX, rtol=0, atol=1e-6)


# --- PyTree reconstruction ---------------------------------------------------

@pytest.mark.parametrize("cls", [TanJax, SinJax], ids=lambda c: c.__name__)
def test_eval_shape_on_jax_wcs(cls):
    wcs = cls([0.0, 0.0], CD, [0.0, 0.0])
    out = jax.eval_shape(lambda w: w, wcs)
    assert out.crpix.shape == (2,)
    assert out.crval.shape == (2,)


def test_unflatten_preserves_leaf_dtypes():
    """No re-cast on reconstruction: leaves come back exactly as they went in.

    Construction itself promotes to f64 (``WCSBase.__init__`` uses
    ``dtype=float``), so the invariant is round-trip stability, not f32.
    """
    wcs = TanJax([0.0, 0.0], CD, [180.0, 0.0])
    leaves, treedef = jax.tree_util.tree_flatten(wcs)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    for name in ("crpix", "cd", "crval"):
        assert getattr(back, name).dtype == getattr(wcs, name).dtype, name


def test_unflatten_does_not_promote_f32_leaves():
    """A hand-built f32 object must survive unflatten without being promoted.

    ``tree_unflatten`` used to call ``__init__``, which applies
    ``dtype=float``; that silently rewrote f32 parameters to f64 on every
    ``jit`` boundary. Build with ``__new__`` to bypass construction and check
    the unflatten path in isolation.
    """
    f32 = lambda a: jnp.asarray(a, dtype=jnp.float32)
    wcs = TanJax.__new__(TanJax)
    wcs.crpix, wcs.cd, wcs.crval, wcs.sip = f32([0.0, 0.0]), f32(CD), f32([180.0, 0.0]), None
    leaves, treedef = jax.tree_util.tree_flatten(wcs)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    for name in ("crpix", "cd", "crval"):
        assert getattr(back, name).dtype == jnp.float32, name


def test_tree_map_cast_moves_every_leaf():
    """A pytree-level cast reaches crpix, cd and crval alike."""
    wcs = TanJax([0.0, 0.0], CD, [180.0, 0.0])
    cast = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=jnp.float32), wcs)
    for name in ("crpix", "cd", "crval"):
        assert getattr(cast, name).dtype == jnp.float32, name


def test_crval_gradient_survives_the_pytree():
    """r_matrix is derived, not a stored leaf, so crval stays differentiable."""
    def f(crval):
        wcs = TanJax([0.0, 0.0], CD, crval)
        ra, _ = wcs.unproj(jnp.array(10.0), jnp.array(10.0))
        return ra

    grad = jax.grad(f)(jnp.array([180.0, 0.0]))
    assert np.all(np.isfinite(grad))
    assert np.any(np.abs(grad) > 1e-9), f"crval gradient was zeroed: {grad}"


@pytest.mark.parametrize("cls", [TanJax, SinJax], ids=lambda c: c.__name__)
def test_jit_and_vmap_over_parameters(cls):
    def state(crpix):
        wcs = cls(crpix, CD, [180.0, 0.0])
        ra, dec = wcs.unproj(jnp.array(10.0), jnp.array(10.0))
        return ra, dec

    jitted = jax.jit(state)(jnp.array([0.0, 0.0]))
    vmapped = jax.vmap(state)(jnp.zeros((3, 2)))
    assert np.all(np.isfinite(np.asarray(jitted)))
    assert np.asarray(vmapped[0]).shape == (3,)


@pytest.mark.parametrize("cls", [TanJaxMixed, SinJaxMixed], ids=lambda c: c.__name__)
def test_mixed_unflatten_keeps_the_precision_split(cls):
    wcs = cls([0.0, 0.0], CD, [180.0, 0.0])
    leaves, treedef = jax.tree_util.tree_flatten(wcs)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.crpix.dtype == jnp.float32
    assert back.cd.dtype == jnp.float32
    assert back.crval.dtype == jnp.float64


def test_array_unflatten_preserves_dtypes():
    crvals = (jnp.asarray([180.0, 181.0]), jnp.asarray([0.0, 1.0]))
    wcs = TanJaxArray(jnp.asarray([0.0, 0.0]), jnp.asarray(CD), crvals)
    leaves, treedef = jax.tree_util.tree_flatten(wcs)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    for name in ("crpix", "cd"):
        assert getattr(back, name).dtype == getattr(wcs, name).dtype, name
    for got, want in zip(back.crvals, wcs.crvals):
        assert got.dtype == want.dtype


def test_sip_survives_unflatten():
    a = np.zeros((3, 3)); a[2, 0] = 1e-5
    sip = Sip(a=a)
    wcs = TanJax([0.0, 0.0], CD, [180.0, 0.0], sip=sip)
    leaves, treedef = jax.tree_util.tree_flatten(wcs)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(np.asarray(back.sip.a), a)
    assert back.sip is not None


# --- projection domains: gradients and the full pipeline ---------------------

MIXED_SEP = np.radians(np.array([10.0, 135.0, 20.0, 179.99, 5.0]))


@pytest.mark.parametrize("cls", [TanJax, SinJax], ids=lambda c: c.__name__)
def test_masked_loss_keeps_parameter_gradients_finite(cls):
    """A NaN reaching the CD multiply poisons its pullback (0 * NaN = NaN).

    Both a multiplicative mask and masking only in `_native_to_plane` fail
    this; the fix substitutes before the arithmetic and selects at the output.
    """
    cd = jnp.asarray(CD)

    def loss_cd(cd_):
        wcs = cls(jnp.zeros(2), cd_, jnp.zeros(2))
        x, y = wcs.proj(MIXED_SEP, jnp.zeros_like(MIXED_SEP))
        return jnp.nansum(x) + jnp.nansum(y)

    def loss_crval(crval_):
        wcs = cls(jnp.zeros(2), cd, crval_)
        x, _ = wcs.proj(MIXED_SEP, jnp.zeros_like(MIXED_SEP))
        return jnp.nansum(x)

    for name, grad in (("cd", jax.grad(loss_cd)(cd)),
                       ("crval", jax.grad(loss_crval)(jnp.zeros(2)))):
        assert np.all(np.isfinite(np.asarray(grad))), f"{cls.__name__} {name}: {grad}"


def test_masked_loss_keeps_sip_gradients_finite():
    a = jnp.zeros((3, 3)).at[2, 0].set(1e-5)

    def loss(a_):
        wcs = TanJax(jnp.zeros(2), jnp.asarray(CD), jnp.zeros(2), sip=Sip(a=a_))
        x, _ = wcs.proj(MIXED_SEP, jnp.zeros_like(MIXED_SEP))
        return jnp.nansum(x)

    assert np.all(np.isfinite(np.asarray(jax.grad(loss)(a))))


def test_gradients_are_finite_inside_the_domain():
    """Finite gradients are required inside the domain, not at SIN's limb."""
    sep = np.radians(np.array([1.0, 5.0, 10.0]))

    def loss(a_):
        wcs = TanJax(jnp.zeros(2), jnp.asarray(CD), jnp.zeros(2), sip=Sip(a=a_))
        x, y = wcs.proj(sep, np.zeros_like(sep))
        return jnp.sum(x ** 2 + y ** 2)

    jac = jax.jacrev(loss)(jnp.zeros((3, 3)))
    assert np.all(np.isfinite(np.asarray(jac)))


def test_sin_inverse_domain_is_nan_outside_the_unit_disc():
    """SIN's inverse is bounded (X^2 + Y^2 <= 1); TAN's is not."""
    wcs = SinJax(jnp.zeros(2), jnp.asarray(CD), jnp.zeros(2))
    r_px = np.degrees(2.0) / CD[1][1]  # 2 rad > 1 rad limb
    ra, dec = wcs.unproj(jnp.asarray([r_px, 0.0]), jnp.asarray([0.0, 0.0]))
    assert np.isnan(np.asarray(ra)[0]) and np.isnan(np.asarray(dec)[0])
    assert np.isfinite(np.asarray(ra)[1])


def test_tan_inverse_has_no_radius_limit():
    """TAN's inverse is finite for every finite pixel, so a "far" pixel is
    still valid there -- unlike SIN."""
    wcs = TanJax(jnp.zeros(2), jnp.asarray(CD), jnp.zeros(2))
    r_px = np.degrees(2.0) / CD[1][1]
    ra, dec = wcs.unproj(jnp.asarray(r_px), jnp.asarray(0.0))
    assert np.isfinite(np.asarray(ra)) and np.isfinite(np.asarray(dec))


def test_offset_apis_are_nan_outside_the_domain():
    wcs = Tan([0.0, 0.0], CD, [0.0, 0.0])
    ra_bad = np.radians(135.0)
    dalpha, ddelta = wcs.unproj_offset(np.array([np.degrees(1.0) / CD[1][1]]), np.array([0.0]))
    assert np.all(np.isfinite(dalpha))
    x, y = wcs.proj(ra_bad, 0.0)
    assert np.isnan(x) and np.isnan(y)


def test_reprojector_masks_invalid_source_pixels():
    """SIN's inverse is bounded, so a pixel past the limb is invalid there.

    TAN's inverse is unbounded (every finite pixel is on the forward
    hemisphere), so SIN is the projection that can express this in pixel
    space.
    """
    from mapprojax import Reprojector

    src = Sin(np.zeros(2), CD, [0.0, 0.0])
    tgt = Sin(np.zeros(2), CD, [0.0, 0.0])
    rp = Reprojector(src, tgt)
    r_px = np.degrees(1.0) / CD[1][1]  # the unit disc in pixels
    x, y = rp.pix_to_pix(np.array([0.0, r_px * 1.01, r_px * 0.5]), np.zeros(3))
    got = np.asarray(x)
    assert np.isnan(got[1]), got
    assert np.isfinite(got[0]) and np.isfinite(got[2]), got


# --- masked-loss gradients through every public method -----------------------
#
# Validity is carried as a flag to the public outputs; NaN is inserted only
# there. A NaN inserted mid-pipeline reaches a later multiply (CD inverse,
# SIP, rotation, offset trig, the second Reprojector leg) whose pullback gives
# 0 * NaN = NaN even under a loss that masks the invalid sample.

SIP_A = np.zeros((3, 3)); SIP_A[2, 0] = 2e-6; SIP_A[1, 1] = -1e-6
SIP_B = np.zeros((3, 3)); SIP_B[0, 2] = 2e-6; SIP_B[1, 1] = 1e-6
R_PX = np.degrees(1.0) / CD[1][1]  # SIN's unit disc in pixels

# pixel batch: valid, SIN-invalid (outside the disc), NaN input, valid
PIX_X = np.array([30.0, 1.5 * R_PX, np.nan, -40.0])
PIX_Y = np.array([-20.0, 0.0, 5.0, 25.0])
# sky batch: valid, back hemisphere (both projections), NaN input, valid
SKY_RA = np.radians(np.array([0.5, 179.0, 0.2, -0.8]))
SKY_DEC = np.radians(np.array([0.3, 0.0, np.nan, -0.4]))


def _masked(out):
    return sum(jnp.sum(jnp.where(jnp.isfinite(o), o, 0.0) ** 2) for o in out)


def _call(wcs, method):
    if method in ("unproj", "unproj_xyz", "unproj_offset", "pix_to_native"):
        return getattr(wcs, method)(jnp.asarray(PIX_X), jnp.asarray(PIX_Y))
    if method == "proj":
        return wcs.proj(jnp.asarray(SKY_RA), jnp.asarray(SKY_DEC))
    if method == "proj_offset":
        return wcs.proj_offset(jnp.asarray(SKY_RA), jnp.asarray(SKY_DEC))
    from mapprojax.utils import radec_to_xyz
    xyz = radec_to_xyz(jnp.asarray(SKY_RA), jnp.asarray(SKY_DEC), xp=jnp)
    if method == "proj_xyz":
        return wcs.proj_xyz(*xyz)
    return wcs.native_to_pix(*xyz)  # native_to_pix: the sky batch as native vectors


METHODS = ["proj", "proj_xyz", "proj_offset", "native_to_pix",
           "unproj", "unproj_xyz", "unproj_offset", "pix_to_native"]


@pytest.mark.parametrize("with_sip", [False, True], ids=["nosip", "sip"])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", [TanJax, SinJax], ids=lambda c: c.__name__)
def test_masked_loss_gradients_are_finite_everywhere(cls, method, with_sip):
    crval = jnp.asarray([10.0, 20.0])

    def build(params):
        sip = Sip(a=params["a"], b=params["b"]) if with_sip else None
        return cls(params["crpix"], params["cd"], params["crval"], sip=sip)

    params = {"crpix": jnp.asarray([3.0, -2.0]), "cd": jnp.asarray(CD), "crval": crval,
              "a": jnp.asarray(SIP_A), "b": jnp.asarray(SIP_B)}
    out = _call(build(params), method)
    valid = np.isfinite(np.asarray(out[0]))
    assert valid.any() and not valid.all(), (method, out)  # a genuinely mixed batch

    grads = jax.grad(lambda p: _masked(_call(build(p), method)))(params)
    for name, g in grads.items():
        assert np.all(np.isfinite(np.asarray(g))), f"{cls.__name__}.{method} d/d{name}: {g}"


@pytest.mark.parametrize("cls", [TanJax, SinJax], ids=lambda c: c.__name__)
def test_masked_loss_pixel_gradients_are_finite(cls):
    w = cls(jnp.zeros(2), jnp.asarray(CD), jnp.asarray([10.0, 20.0]),
            sip=Sip(a=jnp.asarray(SIP_A), b=jnp.asarray(SIP_B)))
    gx, gy = jax.grad(lambda x, y: _masked(w.unproj(x, y)), argnums=(0, 1))(
        jnp.asarray(PIX_X), jnp.asarray(PIX_Y))
    assert np.all(np.isfinite(np.asarray(gx))) and np.all(np.isfinite(np.asarray(gy)))


def test_masked_gradient_of_a_valid_sample_ignores_its_invalid_neighbours():
    """Masking must not just make gradients finite but leave the valid
    samples' contribution exactly as if the invalid ones were absent."""
    def loss(cd, xs, ys):
        w = SinJax(jnp.zeros(2), cd, jnp.asarray([10.0, 20.0]))
        return _masked(w.unproj(xs, ys))

    ok = np.isfinite(PIX_X) & (np.abs(PIX_X) < R_PX)
    full = jax.grad(loss)(jnp.asarray(CD), jnp.asarray(PIX_X), jnp.asarray(PIX_Y))
    alone = jax.grad(loss)(jnp.asarray(CD), jnp.asarray(PIX_X[ok]), jnp.asarray(PIX_Y[ok]))
    np.testing.assert_allclose(np.asarray(full), np.asarray(alone), rtol=1e-12, atol=0)


@pytest.mark.parametrize("method", ["unproj", "unproj_xyz", "unproj_offset"])
@pytest.mark.parametrize("crval", [(0.0, 0.0), (10.0, 20.0), (0.0, 90.0), (0.0, -90.0)],
                         ids=["origin", "ordinary", "north-pole", "south-pole"])
@pytest.mark.parametrize("bad_pixel", [(0.0, 1.5 * R_PX), (1.5 * R_PX, 0.0)],
                         ids=["vertical", "horizontal"])
def test_invalid_standin_never_hits_a_spherical_singularity(method, crval, bad_pixel):
    """An invalid SIN pixel on the vertical axis lands at native (0, 0, 2.09):
    finite, but with CRVAL (0, 0) it rotates onto the pole of the spherical
    conversion, whose derivative is infinite. The stand-in must be swapped in
    before atan2/hypot, for every reference point including the poles."""
    xs = jnp.asarray([10.0, bad_pixel[0]])
    ys = jnp.asarray([10.0, bad_pixel[1]])

    def loss(c, x, y):
        return _masked(getattr(SinJax(jnp.zeros(2), jnp.eye(2) * 1e-3, c), method)(x, y))

    c = jnp.asarray(crval)
    mixed = jax.grad(loss)(c, xs, ys)
    jitted = jax.jit(jax.grad(loss))(c, xs, ys)
    alone = jax.grad(loss)(c, xs[:1], ys[:1])
    assert np.all(np.isfinite(np.asarray(mixed))) and np.all(np.isfinite(np.asarray(jitted)))
    np.testing.assert_allclose(np.asarray(mixed), np.asarray(alone), rtol=1e-12, atol=0)


def _assert_mixed_equals_alone(loss, params, mixed_args, alone_args):
    """Eager and jitted masked-loss gradients on a mixed batch must be finite
    and equal to the valid sample's gradient computed on its own."""
    mixed = jax.grad(loss)(params, *mixed_args)
    jitted = jax.jit(jax.grad(loss))(params, *mixed_args)
    alone = jax.grad(loss)(params, *alone_args)
    for m, j, a in zip(jax.tree_util.tree_leaves(mixed), jax.tree_util.tree_leaves(jitted),
                       jax.tree_util.tree_leaves(alone)):
        assert np.all(np.isfinite(np.asarray(m))) and np.all(np.isfinite(np.asarray(j))), (m, j)
        np.testing.assert_allclose(np.asarray(m), np.asarray(a), rtol=1e-9, atol=1e-300)
        np.testing.assert_allclose(np.asarray(j), np.asarray(a), rtol=1e-9, atol=1e-300)


def test_nan_pixel_standin_cannot_land_on_the_sin_limb():
    """With CRPIX = (0, R) a stand-in pixel of (0, 0) sits exactly on SIN's
    limb (r = 1, sqrt(0)); the substitute must be a regular point instead."""
    R = np.degrees(1.0)
    xs, ys = jnp.asarray([10.0, np.nan]), jnp.asarray([R + 10.0, 0.0])

    def loss(p, x, y):
        return _masked(SinJax(p["crpix"], p["cd"], jnp.zeros(2)).unproj(x, y))

    params = {"crpix": jnp.asarray([0.0, R]), "cd": jnp.eye(2)}
    _assert_mixed_equals_alone(loss, params, (xs, ys), (xs[:1], ys[:1]))


@pytest.mark.parametrize("precision", ["f64", "f32"])
def test_invalid_source_sample_never_reaches_target_arithmetic(precision):
    """An invalid SIN source sample (outside the disc) is in the TAN target's
    domain but at a huge plane coordinate; SIP on it overflows. The source
    validity has to select a regular target stand-in before the target leg."""
    a = np.zeros((4, 4)); a[2, 0] = 1e-7; a[0, 3] = 1e-10
    b = np.zeros((4, 4)); b[0, 2] = 1e-7; b[3, 0] = 1e-10
    cd = np.eye(2) * 1e-3
    src = Sin(np.zeros(2), cd, [0.0, 0.0])
    tgt = Tan(np.zeros(2), cd, [0.0, 0.02], sip=Sip(a=a, b=b))
    rp = Reprojector(src, tgt, matrix_precision=precision)
    bad = 1.5 * np.degrees(1.0) / 1e-3
    xs, ys = jnp.asarray([10.0, 0.0]), jnp.asarray([10.0, bad])
    out = rp.pix_to_pix(xs, ys)
    assert np.isfinite(np.asarray(out[0]))[0] and np.isnan(np.asarray(out[0]))[1], out

    def loss(r, x, y):
        return _masked(r.pix_to_pix(x, y))

    # f32 gradients of the two batch shapes differ by rounding only.
    mixed = jax.grad(loss)(rp, xs, ys)
    jitted = jax.jit(jax.grad(loss))(rp, xs, ys)
    alone = jax.grad(loss)(rp, xs[:1], ys[:1])
    tol = 1e-9 if precision == "f64" else 1e-5
    for m, j, al in zip(jax.tree_util.tree_leaves(mixed), jax.tree_util.tree_leaves(jitted),
                        jax.tree_util.tree_leaves(alone)):
        for got in (m, j):
            assert np.all(np.isfinite(np.asarray(got))), got
            np.testing.assert_allclose(np.asarray(got), np.asarray(al), rtol=tol, atol=0)
    assert np.any(np.asarray(alone.tgt_leg.sip.a) != 0)


@pytest.mark.parametrize("proj_cls", [Sin, Tan], ids=lambda c: c.__name__)
@pytest.mark.parametrize("precision", ["f64", "f32"])
def test_reprojector_masked_gradients_for_both_legs(proj_cls, precision):
    """The source leg's invalid output is the target leg's input; it must
    reach the target leg as a flag, not as NaN."""
    sip = Sip(a=SIP_A, b=SIP_B)
    src = proj_cls([3.0, -2.0], CD, [10.0, 20.0], sip=sip)
    tgt = proj_cls([-4.0, 6.0], CD, [10.05, 20.02], sip=sip)
    rp = Reprojector(src, tgt, matrix_precision=precision)
    xs, ys = jnp.asarray(PIX_X), jnp.asarray(PIX_Y)

    out = rp.pix_to_pix(xs, ys)
    valid = np.isfinite(np.asarray(out[0]))
    assert valid.any() and not valid.all(), out

    g = jax.grad(lambda r: _masked(r.pix_to_pix(xs, ys)))(rp)
    for leg in ("src_leg", "tgt_leg"):
        leg_g = getattr(g, leg)
        for name, val in (("crpix", leg_g.crpix), ("cd", leg_g.cd),
                          ("sip.a", leg_g.sip.a), ("sip.b", leg_g.sip.b)):
            arr = np.asarray(val)
            assert np.all(np.isfinite(arr)), f"{leg}.{name}: {arr}"
            assert np.any(arr != 0), f"{leg}.{name} is identically zero"


# --- singular / non-finite CD: the scaled-determinant rule -----------------

@pytest.mark.parametrize(
    "cd, expect",
    [
        pytest.param([[1e-25, 0.0], [0.0, 1e-25]], True, id="tiny-f32-underflow"),
        pytest.param([[1e25, 0.0], [0.0, 1e25]], True, id="huge-f32-overflow"),
        pytest.param([[1e-3, 0.0], [1e-3, 0.0]], False, id="rank-deficient"),
        pytest.param([[0.0, 0.0], [0.0, 0.0]], False, id="zero"),
    ],
)
def test_scaled_determinant_accepts_representable_extremes(cd, expect):
    """A raw det == 0 test is wrong both ways: diag(1e-25) underflows an f32
    determinant to 0, and diag(1e25) overflows it to inf, though both are
    invertible. Only a genuinely rank-deficient matrix is invalid."""
    from mapprojax.base import _cd_inverse

    f32 = np.asarray(cd, dtype=np.float32)
    _, valid = _cd_inverse(f32, xp=np)
    assert bool(valid) is expect
    if expect:
        assert np.abs(np.asarray(f32, float) @ _cd_inverse(f32, xp=np)[0]
                      - np.eye(2)).max() < 1e-6


def test_singular_cd_is_rejected_at_construction():
    # Exactly singular in fp64 already: the ordinary rejection.
    cd = [[1e-3, 0.0], [1e-3 + 1e-12, 0.0]]
    with pytest.raises(ValueError):
        Tan(np.zeros(2), cd, np.zeros(2))


# Invertible in fp64 (det ~ 1e-14), singular once cast to f32 (1 + 1e-8 == 1).
CAST_LOSS_CD = [[1e-3, 1e-3], [1e-3, 1e-3 * (1 + 1e-8)]]


def test_matrix_that_loses_rank_when_cast_to_f32_is_rejected():
    """The check runs on the stored (cast) matrix, wherever the cast happens."""
    from mapprojax import Reprojector

    fp64 = Tan(np.zeros(2), CAST_LOSS_CD, np.zeros(2))  # valid in fp64
    Reprojector(fp64, fp64, matrix_precision="f64")      # and in an f64 leg
    with pytest.raises(ValueError):
        Reprojector(fp64, fp64, matrix_precision="f32")
    with pytest.raises(ValueError):
        TanJaxMixed(np.zeros(2), CAST_LOSS_CD, np.zeros(2))


@pytest.mark.parametrize("cls", [TanJaxMixed, SinJaxMixed, TanJaxArrayMixed, SinJaxArrayMixed],
                         ids=lambda c: c.__name__)
@pytest.mark.parametrize("kind", ["singular", "nan"])
def test_mixed_constructors_reject_bad_cd(cls, kind):
    cd = BAD_CD[kind]
    crval = (np.zeros(1), np.zeros(1)) if "Array" in cls.__name__ else np.zeros(2)
    with pytest.raises(ValueError):
        cls(np.zeros(2), cd, crval)


def test_cd_inverse_is_finite_on_the_invalid_branch():
    from mapprojax.base import _cd_inverse

    inv, valid = _cd_inverse(np.diag([np.inf, 1e-3]), xp=np)
    assert not bool(valid)
    assert np.all(np.isfinite(inv))


@pytest.mark.parametrize("impl", ["proj", "unproj"])
@pytest.mark.parametrize("cd", [[[np.inf, 0.0], [0.0, 1e-3]], BAD_CD["nan"]], ids=["inf", "nan"])
def test_masked_gradient_wrt_non_finite_cd_is_finite(impl, cd):
    def loss(cd):
        w = TanJax(jnp.zeros(2), cd, jnp.zeros(2))
        out = (w.proj(jnp.array([1e-3]), jnp.array([1e-3])) if impl == "proj"
               else w.unproj(jnp.array([10.0]), jnp.array([20.0])))
        return sum(jnp.sum(jnp.where(jnp.isfinite(o), o, 0.0)) for o in out)

    g = jax.jit(jax.grad(loss))(jnp.asarray(cd))
    assert np.all(np.isfinite(np.asarray(g))), g


@pytest.mark.parametrize("impl", ["proj", "unproj"])
@pytest.mark.parametrize("kind", list(BAD_CD))
def test_traced_bad_cd_is_nan_in_both_directions(impl, kind):
    """A traced CD cannot raise. pix_to_native multiplies rather than
    inverts, so it needs its own validity test -- otherwise a singular CD
    yields a finite, meaningless native vector."""
    if impl == "proj":
        f = jax.jit(lambda cd: TanJax(jnp.zeros(2), cd, jnp.zeros(2)).proj(1e-3, 1e-3))
    else:
        f = jax.jit(lambda cd: TanJax(jnp.zeros(2), cd, jnp.zeros(2)).unproj(10.0, 20.0))
    out = f(jnp.asarray(BAD_CD[kind]))
    assert all(np.isnan(np.asarray(v)) for v in out), out
