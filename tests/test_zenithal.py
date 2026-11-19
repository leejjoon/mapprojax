"""ZEA, STG and ARC: formulas, domains and finite gradients.

The plan's Milestone v0.5.0 numerical contract:

* each projection agrees with astropy's ``all_pix2world`` and has the same
  validity mask, away from a few-ulp band around each limb;
* the limbs are strict analytic bounds (ZEA ``R <= 2``, ARC ``R <= pi``,
  STG unbounded), equality included;
* the exact antipode, native ``(-1, 0, 0)``, is invalid for all three by
  design (it has no native longitude; WCSLIB returns an arbitrary limb point
  for ZEA and ARC), while points near it stay valid and accurate in float32;
* values and gradients are finite at the exact reference pixel -- where every
  invalid sample's stand-in also lands -- in every precision mode;
* ARC's series and exact branches are each within 2 ulp (float64) or 3 ulp
  (float32; see ``MAX_ULP``) of an extended-precision reference on both sides
  of the switch-over.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from mapprojax import Arc, Reprojector, Stg, Zea, from_astropy
from mapprojax.projections import _ARC_SERIES_LIMIT
from mapprojax.wcs_utils import load_wcs_params_from_header

CLASSES = {"ZEA": Zea, "STG": Stg, "ARC": Arc}
#: Plane-radius limb in radians; None means every finite plane point is valid.
LIMB = {"ZEA": 2.0, "STG": None, "ARC": np.pi}
DEG_PER_RAD = 180.0 / np.pi


@pytest.fixture(autouse=True)
def _x64_enabled():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _astropy_wcs(code, crval2, cd=((-0.15, 0.03), (0.06, 0.15)), crpix=513.0):
    w = WCS(naxis=2)
    w.wcs.ctype = [f"RA---{code}", f"DEC--{code}"]
    w.wcs.crpix = [crpix, crpix]
    w.wcs.crval = [150.0, crval2]
    w.wcs.cd = [list(row) for row in cd]
    if crval2 == 90.0:
        w.wcs.lonpole = 180.0
    w.wcs.set()
    return w


def _unit(lon, lat):
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def _sep_rad(lon1, lat1, lon2, lat2):
    d = np.linalg.norm(_unit(lon1, lat1) - _unit(lon2, lat2), axis=-1)
    return 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))


def _plane_radius(w, pix):
    """Plane radius in radians of 0-based pixels (no SIP)."""
    rel = pix - (np.asarray(w.wcs.crpix) - 1.0)
    return np.radians(np.linalg.norm(rel @ np.asarray(w.wcs.cd).T, axis=-1))


# --- agreement with astropy ------------------------------------------------


@pytest.mark.parametrize("crval2", [30.0, -90.0, 90.0])
@pytest.mark.parametrize("code", sorted(CLASSES))
def test_unproj_and_mask_match_astropy(code, crval2):
    w = _astropy_wcs(code, crval2)
    kernel = from_astropy(w, precision="f64")
    assert type(kernel) is CLASSES[code]
    # Pixels reaching ~310 degrees of plane radius: beyond both limbs.
    pix = np.random.default_rng(1).uniform(-1500.0, 2500.0, (20000, 2))
    limb = LIMB[code]
    if limb is not None:
        r = _plane_radius(w, pix)
        pix = pix[np.abs(r - limb) > 1e-9 * limb]  # skip the few-ulp band
    ra, dec = w.all_pix2world(pix[:, 0], pix[:, 1], 0)
    ok = np.isfinite(ra)
    lon, lat = (np.asarray(v) for v in kernel.unproj(jnp.asarray(pix[:, 0]), jnp.asarray(pix[:, 1])))
    np.testing.assert_array_equal(np.isfinite(lon), ok)
    if limb is not None:
        assert (~ok).any(), "the sample must reach beyond the limb"
    sep = _sep_rad(lon[ok], lat[ok], np.radians(ra[ok]), np.radians(dec[ok]))
    assert np.degrees(sep.max()) * 3.6e6 < 1e-4  # mas


@pytest.mark.parametrize("code", sorted(CLASSES))
def test_proj_round_trip_matches_astropy_world_to_pixel(code):
    w = _astropy_wcs(code, 30.0)
    kernel = from_astropy(w, precision="f64")
    pix = np.random.default_rng(2).uniform(-600.0, 1600.0, (5000, 2))
    ra, dec = w.all_pix2world(pix[:, 0], pix[:, 1], 0)
    ok = np.isfinite(ra)
    x, y = kernel.proj(jnp.radians(jnp.asarray(ra[ok])), jnp.radians(jnp.asarray(dec[ok])))
    err = np.hypot(np.asarray(x) - pix[ok, 0], np.asarray(y) - pix[ok, 1])
    assert err.max() < 1e-7


# --- limbs and the antipode --------------------------------------------------


def _limb(code, dtype):
    """The largest value of ``dtype`` that is <= the analytic limb: float32(pi)
    is *above* pi, so it is genuinely beyond ARC's limb."""
    limb = jnp.asarray(LIMB[code], dtype=dtype)
    if float(limb) > LIMB[code] or (code == "ARC" and dtype == jnp.float32):
        limb = jnp.nextafter(limb, jnp.asarray(0.0, dtype=dtype))
    return limb


def _kernel(cls, precision="f64"):
    """Plane radius in radians equals the pixel offset from (0, 0)."""
    cd = [[DEG_PER_RAD, 0.0], [0.0, DEG_PER_RAD]]
    return cls([0.0, 0.0], cd, [150.0, 30.0], precision=precision)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("code", ["ZEA", "ARC"])
def test_exact_limb_is_valid_and_beyond_is_not(code, dtype):
    k = _kernel(CLASSES[code])
    limb = _limb(code, dtype)
    beyond = jnp.nextafter(limb, jnp.asarray(np.inf, dtype=dtype))
    zero = jnp.zeros((), dtype=dtype)
    for X, Y in ((limb, zero), (zero, limb)):
        *vec, valid = k._plane_to_native(X, Y)
        assert bool(valid)
        assert all(np.isfinite(float(v)) for v in vec)
    *vec, valid = k._plane_to_native(beyond, zero)
    assert not bool(valid)
    assert all(np.isfinite(float(v)) for v in vec)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_arc_limb_maps_to_antipode_and_zea_limb_too(dtype):
    for code in ("ZEA", "ARC"):
        k = _kernel(CLASSES[code])
        x, y, z, valid = k._plane_to_native(_limb(code, dtype), jnp.zeros((), dtype=dtype))
        assert bool(valid)
        np.testing.assert_allclose([float(x), float(y), float(z)], [-1.0, 0.0, 0.0],
                                   atol=1e-6 if dtype == jnp.float32 else 1e-12)


def test_stg_every_finite_plane_point_is_valid_and_finite():
    k = _kernel(Stg)
    for dtype in (jnp.float32, jnp.float64):
        big = jnp.asarray(jnp.finfo(dtype).max / 4, dtype=dtype)
        for X in (jnp.asarray(1e20, dtype=dtype), big):
            x, y, z, valid = k._plane_to_native(X, jnp.zeros((), dtype=dtype))
            assert bool(valid)
            assert float(x) == -1.0 and float(z) == 0.0
            assert 0.0 <= float(y) <= 1e-10  # 4 / X, not exactly zero

            grad = jax.grad(lambda X: sum(k._plane_to_native(X, X * 0)[:3]))(X)
            assert np.isfinite(float(grad))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("code", sorted(CLASSES))
def test_exact_antipode_is_invalid_with_finite_stand_ins(code, dtype):
    k = _kernel(CLASSES[code])
    x, y, z = (jnp.asarray(v, dtype=dtype) for v in (-1.0, 0.0, 0.0))
    X, Y, valid = k._native_to_plane(x, y, z)
    assert not bool(valid)
    assert np.isfinite(float(X)) and np.isfinite(float(Y))
    px, py = k.native_to_pix(x, y, z)
    assert np.isnan(float(px)) and np.isnan(float(py))

    grad = jax.grad(lambda v: sum(k._native_to_plane(v, y, z)[:2]))(x)
    assert np.isfinite(float(grad))


def _radial(code, rho):
    return {"ZEA": 2 * np.sin(rho / 2), "STG": 2 * np.tan(rho / 2), "ARC": rho}[code]


@pytest.mark.parametrize("code", sorted(CLASSES))
def test_near_antipode_is_valid_and_accurate_in_f32(code):
    """In float32, x rounds to -1 within ~0.014 deg of the antipode; those
    points are still valid, and 1 + x must not cancel."""
    rho = np.radians([170.0, 178.0, 179.9, 179.99, 179.999])
    x64, y64 = np.cos(rho), np.sin(rho)
    k = _kernel(CLASSES[code], precision="f32")
    X, Y, valid = k._native_to_plane(jnp.asarray(x64, jnp.float32), jnp.asarray(y64, jnp.float32),
                                     jnp.zeros(rho.shape, jnp.float32))
    assert np.all(np.asarray(valid))
    assert float(jnp.asarray(x64[-1], jnp.float32)) == -1.0  # the case that matters
    # Reference: the same float32 inputs, evaluated in float64.
    x32 = np.asarray(jnp.asarray(x64, jnp.float32), np.float64)
    y32 = np.asarray(jnp.asarray(y64, jnp.float32), np.float64)
    rho32 = np.arctan2(y32, x32)
    np.testing.assert_allclose(np.asarray(X, np.float64), _radial(code, rho32), rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(Y), 0.0)


# --- finite gradients at the reference pixel and on masked samples --------


def _wcs(cls, precision, cd):
    crval = np.array([150.0, 30.0])
    return cls(jnp.array([512.0, 512.0]), cd, crval, precision=precision)


@pytest.mark.parametrize("precision", ["f64", "mixed", "f32"])
@pytest.mark.parametrize("code", sorted(CLASSES))
def test_gradients_are_finite_at_exact_crpix(code, precision):
    cls = CLASSES[code]
    cd0 = jnp.array([[-0.15, 0.03], [0.06, 0.15]])
    dtype = jnp.float32 if precision == "f32" else jnp.float64

    def forward(cd, crpix, px):
        k = cls(crpix, cd, np.array([150.0, 30.0]), precision=precision)
        px = jnp.asarray(px, dtype)
        if precision == "f32":
            a, b = k.unproj_offset(px, px)
        else:
            a, b = k.unproj(px, px)
        return (a + b).astype(jnp.float64)

    def inverse(cd, crpix):
        k = cls(crpix, cd, np.array([150.0, 30.0]), precision=precision)
        one = jnp.asarray(1.0, dtype)
        x, y = k.native_to_pix(one, one * 0, one * 0)  # the reference point
        return (x + y).astype(jnp.float64)

    crpix0 = jnp.array([512.0, 512.0])
    grads = jax.grad(forward, argnums=(0, 1, 2))(cd0, crpix0, 512.0)
    grads += jax.grad(inverse, argnums=(0, 1))(cd0, crpix0)
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g))), g


@pytest.mark.parametrize("precision", ["f64", "mixed"])
@pytest.mark.parametrize("code", ["ZEA", "ARC"])
def test_masked_samples_do_not_poison_parameter_gradients(code, precision):
    """Invalid pixels are swapped for the reference point -- ARC's singular
    point -- so a masked loss must still have a finite gradient."""
    cd0 = jnp.array([[DEG_PER_RAD * 1e-3, 0.0], [0.0, DEG_PER_RAD * 1e-3]])
    beyond = 1.5 * LIMB[code] / 1e-3
    px = jnp.array([512.0, 520.0, 512.0 + beyond])

    def loss(cd):
        k = _wcs(CLASSES[code], precision, cd)
        lon, lat = k.unproj(px, px * 0 + 512.0)
        ok = jnp.isfinite(lon)
        return jnp.sum(jnp.where(ok, lon + lat, 0.0))

    k = _wcs(CLASSES[code], precision, cd0)
    lon, _ = k.unproj(px, px * 0 + 512.0)
    assert np.isfinite(np.asarray(lon)).tolist() == [True, True, False]
    assert np.all(np.isfinite(np.asarray(jax.grad(loss)(cd0))))


@pytest.mark.parametrize("code", ["ZEA", "ARC"])
def test_gradient_just_inside_the_limb_is_finite(code):
    k = _kernel(CLASSES[code])
    X = jnp.asarray(LIMB[code] * (1 - 1e-6))
    grad = jax.grad(lambda X: sum(k._plane_to_native(X, X * 0)[:3]))(X)
    assert np.isfinite(float(grad))


def test_arc_gradient_at_tiny_radius_matches_finite_difference():
    k = _kernel(Arc)
    for s in (0.0, 1e-8, 1e-4, 0.0999, 0.1001, 0.9):
        def f(v):
            y = v
            x = jnp.sqrt(1 - y**2)
            return k._native_to_plane(x, y, y * 0)[0]

        g = float(jax.grad(f)(jnp.asarray(s)))
        assert np.isfinite(g)
        if s > 1e-6:
            h = 1e-7
            fd = (float(f(jnp.asarray(s + h))) - float(f(jnp.asarray(s - h)))) / (2 * h)
            assert g == pytest.approx(fd, rel=1e-6)


# --- ARC series versus exact branch --------------------------------------


def _ulp_error(actual, reference, dtype):
    actual = np.asarray(actual, dtype=np.longdouble)
    spacing = np.abs(np.spacing(np.asarray(reference, dtype=dtype))).astype(np.longdouble)
    return np.abs(actual - reference) / spacing


def _around_limit(dtype):
    limit = _ARC_SERIES_LIMIT[dtype == jnp.float32]
    s2 = np.geomspace(limit / 100, min(limit * 100, 0.9), 401)
    return s2


#: Max ulp error per branch. The plan asks for 2; float32 gets 3 because ROCm's
#: own float32 ``atan2`` is 2.04 ulp off on these inputs (CPU: 0.49), and the
#: exact branch rounds twice more. Measured 2026-11-20, forward float32:
#: series 1.51 (CPU and ROCm), exact 1.48 (CPU) / 2.48 (ROCm).
MAX_ULP = {jnp.float32: 3.0, jnp.float64: 2.0}


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_arc_forward_branches_within_ulp_bound(dtype):
    k = _kernel(Arc)
    s2 = _around_limit(dtype)
    y = np.sqrt(s2).astype(dtype)
    x = np.sqrt(1 - s2).astype(dtype)
    X, _, valid = k._native_to_plane(jnp.asarray(x), jnp.asarray(y), jnp.zeros_like(jnp.asarray(y)))
    assert np.all(np.asarray(valid))
    xl, yl = x.astype(np.longdouble), y.astype(np.longdouble)
    reference = yl * np.arctan2(yl, xl) / yl  # y * atan2(s, x) / s, with s = |y|
    assert _ulp_error(X, reference, dtype).max() <= MAX_ULP[dtype]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_arc_inverse_branches_within_ulp_bound(dtype):
    k = _kernel(Arc)
    r2 = _around_limit(dtype)
    X = np.sqrt(r2).astype(dtype)
    x, y, _, valid = k._plane_to_native(jnp.asarray(X), jnp.zeros_like(jnp.asarray(X)))
    assert np.all(np.asarray(valid))
    Xl = X.astype(np.longdouble)
    assert _ulp_error(x, np.cos(Xl), dtype).max() <= MAX_ULP[dtype]
    assert _ulp_error(y, np.sin(Xl), dtype).max() <= MAX_ULP[dtype]


def test_arc_forward_series_does_not_fire_near_the_antipode():
    """s**2 is small on both sides; only the reference side may use the
    series. The bug placed rho = pi - 5e-5 beside CRPIX."""
    k = _kernel(Arc)
    rho = np.pi - 5e-5
    X, _, valid = k._native_to_plane(jnp.asarray(np.cos(rho)), jnp.asarray(np.sin(rho)),
                                     jnp.asarray(0.0))
    assert bool(valid)
    assert float(X) == pytest.approx(rho, rel=1e-12)


# --- loaders and refusals ------------------------------------------------


def _fits_header(code, **cards):
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 100, 100
    h["CTYPE1"], h["CTYPE2"] = f"RA---{code}", f"DEC--{code}"
    h["CRPIX1"], h["CRPIX2"] = 50.5, 50.5
    h["CRVAL1"], h["CRVAL2"] = 150.0, 30.0
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -0.5, 0.05, 0.04, 0.5
    for key, value in cards.items():
        h[key] = value
    return h


@pytest.mark.parametrize("code", sorted(CLASSES))
def test_header_loader_accepts_zenithal_and_agrees_with_from_astropy(code):
    h = _fits_header(code)
    crpix, cd, crval, sip = load_wcs_params_from_header(h)
    loaded = CLASSES[code](crpix, cd, crval, sip=sip, precision="f64")
    converted = from_astropy(WCS(h), precision="f64")
    assert type(converted) is CLASSES[code]
    for a, b in ((loaded.crpix, converted.crpix), (loaded.cd, converted.cd),
                 (loaded.crval, converted.crval)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    pix = np.array([[0.0, 0.0], [99.0, 99.0], [49.5, 49.5], [10.0, 80.0]])
    ra, dec = WCS(h).all_pix2world(pix[:, 0], pix[:, 1], 0)
    lon, lat = loaded.unproj(jnp.asarray(pix[:, 0]), jnp.asarray(pix[:, 1]))
    sep = _sep_rad(np.asarray(lon), np.asarray(lat), np.radians(ra), np.radians(dec))
    assert np.degrees(sep.max()) * 3.6e6 < 1e-4


@pytest.mark.parametrize("code", sorted(CLASSES))
def test_zenithal_sip_is_accepted_and_applied(code):
    h = _fits_header(code)
    h["CTYPE1"], h["CTYPE2"] = f"RA---{code}-SIP", f"DEC--{code}-SIP"
    h["A_ORDER"], h["B_ORDER"] = 2, 2
    h["A_2_0"], h["B_0_2"], h["A_1_1"] = 2e-4, -1.5e-4, 1e-4
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    assert type(kernel) is CLASSES[code] and kernel.sip is not None
    pix = np.array([[0.0, 0.0], [99.0, 99.0], [10.0, 80.0]])
    ra, dec = w.all_pix2world(pix[:, 0], pix[:, 1], 0)
    lon, lat = kernel.unproj(jnp.asarray(pix[:, 0]), jnp.asarray(pix[:, 1]))
    sep = _sep_rad(np.asarray(lon), np.asarray(lat), np.radians(ra), np.radians(dec))
    assert np.degrees(sep.max()) * 3.6e6 < 1e-4


REFUSED = {
    "pv2_1": (dict(PV2_1=0.0), "PV"),
    "pv1_3": (dict(PV1_3=180.0), "PV"),
    "lonpole_0": (dict(LONPOLE=0.0), "LONPOLE"),
    "pole_without_lonpole": (dict(CRVAL2=90.0), "LONPOLE"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
@pytest.mark.parametrize("code", sorted(CLASSES))
def test_zenithal_refusals(code, case):
    cards, match = REFUSED[case]
    h = _fits_header(code, **cards)
    with pytest.raises(ValueError, match=match):
        load_wcs_params_from_header(h)
    with pytest.raises(ValueError, match=match):
        from_astropy(WCS(h), precision="f64")


@pytest.mark.parametrize("ctypes", [
    ("RA---ZEA", "DEC--TAN"), ("RA---AZP", "DEC--AZP"), ("RA---ZPN", "DEC--ZPN"),
    ("RA---AIR", "DEC--AIR"), ("RA---SZP", "DEC--SZP"),
])
def test_other_projection_codes_stay_refused(ctypes):
    h = _fits_header("ZEA", CTYPE1=ctypes[0], CTYPE2=ctypes[1])
    with pytest.raises(ValueError):
        load_wcs_params_from_header(h)


# --- the pipeline around the projection ----------------------------------


@pytest.mark.parametrize("code", sorted(CLASSES))
def test_reprojector_wide_field_round_trip(code):
    """A 100-degree-wide field through the f32 Reprojector onto itself."""
    cls = CLASSES[code]
    src = cls([511.5, 511.5], [[-0.1, 0.0], [0.0, 0.1]], np.array([30.0, 60.0]), precision="f32")
    xs = jnp.linspace(0.0, 1023.0, 33, dtype=jnp.float32)
    gx, gy = jnp.meshgrid(xs, xs)
    x, y, valid = Reprojector(src, src).pixel_map(gx.ravel(), gy.ravel())
    assert np.all(np.asarray(valid))
    err = np.hypot(np.asarray(x) - np.asarray(gx.ravel()), np.asarray(y) - np.asarray(gy.ravel()))
    assert err.max() < 2e-3
