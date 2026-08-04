"""Tests for unproj_offset/unproj_absolute/proj_offset/proj_absolute
(fo64_report_and_plan.md, Phase 2): dfreproject-style small-offset sky
coordinates that avoid ever forming an absolute (O(1)) RA/Dec value in the
bulk math, so they're accurate in float32 regardless of jax_enable_x64.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mapprojax.projections import Tan, Sin
from mapprojax.jax_projections import TanJax, SinJax
from mapprojax.sip import Sip

CRPIX = [512.0, 512.0]
CD = [[-0.001, 0.0], [0.0, 0.001]]


def _sample_pixels(n=2000, seed=0, lo=50.0, hi=974.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, n), rng.uniform(lo, hi, n)


@pytest.fixture(autouse=True)
def _x64_enabled():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


WCS_CLASSES = [Tan, Sin]


# ---------------------------------------------------------------------------
# unproj_absolute must reproduce unproj (both fp64, NumPy) to machine
# precision -- the offset formulas are algebraically exact (Delta_alpha) or
# accurate to a documented O(Delta_alpha^2) floor (Delta_delta), not
# approximations that would show up here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", WCS_CLASSES)
@pytest.mark.parametrize("crval", [[123.4, -37.7], [10.0, 60.0], [200.0, 0.0]])
def test_unproj_absolute_matches_unproj(cls, crval):
    wcs = cls(CRPIX, CD, crval)
    x, y = _sample_pixels()
    ra_ref, dec_ref = wcs.unproj(x, y)
    ra, dec = wcs.unproj_absolute(x, y)
    np.testing.assert_allclose(ra, ra_ref, atol=1e-10)
    np.testing.assert_allclose(dec, dec_ref, atol=1e-10)


@pytest.mark.parametrize("cls", WCS_CLASSES)
def test_proj_absolute_matches_proj(cls):
    wcs = cls(CRPIX, CD, [123.4, -37.7])
    x, y = _sample_pixels()
    ra, dec = wcs.unproj(x, y)
    x_ref, y_ref = wcs.proj(ra, dec)
    x_out, y_out = wcs.proj_absolute(ra, dec)
    np.testing.assert_allclose(x_out, x_ref, atol=1e-6)
    np.testing.assert_allclose(y_out, y_ref, atol=1e-6)


# ---------------------------------------------------------------------------
# Round trips, in pure NumPy fp64 (correctness of the algebra, independent
# of any fp32 accuracy question).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", WCS_CLASSES)
def test_offset_roundtrip(cls):
    wcs = cls(CRPIX, CD, [123.4, -37.7])
    x, y = _sample_pixels()
    dalpha_cosdec, ddelta = wcs.unproj_offset(x, y)
    x2, y2 = wcs.proj_offset(dalpha_cosdec, ddelta)
    np.testing.assert_allclose(x2, x, atol=1e-6)
    np.testing.assert_allclose(y2, y, atol=1e-6)


@pytest.mark.parametrize("cls", WCS_CLASSES)
def test_absolute_roundtrip(cls):
    wcs = cls(CRPIX, CD, [123.4, -37.7])
    x, y = _sample_pixels()
    ra, dec = wcs.unproj_absolute(x, y)
    x2, y2 = wcs.proj_absolute(ra, dec)
    np.testing.assert_allclose(x2, x, atol=1e-6)
    np.testing.assert_allclose(y2, y, atol=1e-6)


# ---------------------------------------------------------------------------
# RA wraparound: crval near the 0/360 boundary -- proj_absolute must wrap
# ra - ra0 correctly regardless of which side of the boundary ra lands on.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("crval_lon", [0.05, 359.95])
def test_ra_wraparound(crval_lon):
    wcs = Tan(CRPIX, CD, [crval_lon, 10.0])
    x = np.array([300.0, 500.0, 700.0])
    y = np.array([500.0, 500.0, 500.0])

    ra, dec = wcs.unproj_absolute(x, y)
    assert np.all(ra >= 0.0) and np.all(ra < 2 * np.pi)

    x2, y2 = wcs.proj_absolute(ra, dec)
    np.testing.assert_allclose(x2, x, atol=1e-6)
    np.testing.assert_allclose(y2, y, atol=1e-6)

    # Cross-check against the always-correct (fp64, no wrapping needed)
    # unproj/proj path.
    ra_ref, dec_ref = wcs.unproj(x, y)
    sep = np.abs(np.mod(ra - ra_ref + np.pi, 2 * np.pi) - np.pi)
    assert np.max(sep) < 1e-10


def test_ra_wraparound_input_outside_0_2pi():
    """proj_absolute must handle an input ra given as a small negative
    value or a value > 2*pi -- both real possibilities for a caller who
    computed ra elsewhere without normalizing it.
    """
    wcs = Tan(CRPIX, CD, [0.05, 10.0])
    x, y = np.array([500.0]), np.array([500.0])
    ra, dec = wcs.unproj_absolute(x, y)

    for shift in (0.0, -2 * np.pi, 2 * np.pi, 4 * np.pi):
        x_out, y_out = wcs.proj_absolute(ra + shift, dec)
        np.testing.assert_allclose(x_out, x, atol=1e-6)
        np.testing.assert_allclose(y_out, y, atol=1e-6)


# ---------------------------------------------------------------------------
# Near the celestial pole (crval declination close to +/-90).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dec0", [89.9, -89.9])
def test_near_pole(dec0):
    wcs = Tan(CRPIX, CD, [45.0, dec0])
    x, y = _sample_pixels()
    ra_ref, dec_ref = wcs.unproj(x, y)
    ra, dec = wcs.unproj_absolute(x, y)
    sep = np.abs(np.mod(ra - ra_ref + np.pi, 2 * np.pi) - np.pi)
    assert np.max(sep) < 1e-8
    np.testing.assert_allclose(dec, dec_ref, atol=1e-8)


# ---------------------------------------------------------------------------
# SIP compatibility: unproj_offset/proj_offset delegate to
# pix_to_native/native_to_pix, which already handle SIP -- confirm that
# isn't broken by this addition.
# ---------------------------------------------------------------------------


def test_with_sip():
    # ap/bp (approximate inverse) included, not just a/b (forward) --
    # native_to_pix's SIP correction is a no-op without them (Sip's own
    # foc_to_pix returns coeffs=None as zeros), which would make even the
    # plain unproj->proj round trip fail for reasons unrelated to this
    # test's actual target (confirmed against plain unproj/proj directly).
    a = np.zeros((3, 3)); a[2, 0] = 4e-6; a[1, 1] = -2e-6
    b = np.zeros((3, 3)); b[2, 0] = -3e-6; b[1, 1] = 2.5e-6
    ap = np.zeros((3, 3)); ap[2, 0] = -4e-6; ap[1, 1] = 2e-6
    bp = np.zeros((3, 3)); bp[2, 0] = 3e-6; bp[1, 1] = -2.5e-6
    sip = Sip(a=a, b=b, ap=ap, bp=bp)
    wcs = Tan(CRPIX, CD, [123.4, -37.7], sip=sip)
    x, y = _sample_pixels()

    ra_ref, dec_ref = wcs.unproj(x, y)
    ra, dec = wcs.unproj_absolute(x, y)
    np.testing.assert_allclose(ra, ra_ref, atol=1e-10)
    np.testing.assert_allclose(dec, dec_ref, atol=1e-10)

    # AP/BP here is only an approximate inverse of A/B (not independently
    # fitted, just negated -- a crude stand-in), so proj_absolute's round
    # trip through native_to_pix's *uncorrected* AP/BP evaluation carries
    # that same approximation error (~6.5e-3 px measured) -- the same
    # AP/BP-quality characteristic reprojax's _sip_refine.py exists to
    # correct on top of, not a defect in this module's offset formulas
    # (confirmed: unproj_absolute's own accuracy above, which doesn't
    # touch AP/BP at all, is unaffected).
    x2, y2 = wcs.proj_absolute(ra, dec)
    np.testing.assert_allclose(x2, x, atol=0.01)
    np.testing.assert_allclose(y2, y, atol=0.01)


# ---------------------------------------------------------------------------
# The actual point of Phase 2: correct in float32, without jax_enable_x64,
# and error stays near fp32's ~1e-7 rad floor rather than growing with
# field-center offset (unlike the naive "compute absolute RA/Dec in fp32"
# approach it replaces). Bounds have headroom over what was measured.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,jax_cls", [(Tan, TanJax), (Sin, SinJax)])
def test_unproj_offset_accurate_in_fp32_without_x64(cls, jax_cls):
    jax.config.update("jax_enable_x64", False)
    try:
        crval = [123.4, -37.7]
        wcs_ref = cls(CRPIX, CD, crval)  # fp64 ground truth
        wcs_32 = jax_cls(jnp.asarray(CRPIX), jnp.asarray(CD), jnp.asarray(crval))

        x, y = _sample_pixels(n=5000)
        ra_ref, dec_ref = wcs_ref.unproj(x, y)
        ra0, dec0 = np.radians(crval[0]), np.radians(crval[1])
        dalpha_ref = np.mod(ra_ref - ra0 + np.pi, 2 * np.pi) - np.pi
        ddelta_ref = dec_ref - dec0

        dalpha_cosdec, ddelta = wcs_32.unproj_offset(jnp.asarray(x), jnp.asarray(y))
        dalpha_cosdec, ddelta = np.asarray(dalpha_cosdec, dtype=np.float64), np.asarray(ddelta, dtype=np.float64)
        dalpha = dalpha_cosdec / np.cos(dec0)

        assert np.max(np.abs(dalpha - dalpha_ref)) < 1e-6
        assert np.max(np.abs(ddelta - ddelta_ref)) < 1e-6
    finally:
        jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("sep_deg", [0.5, 5.0, 20.0, 45.0])
def test_unproj_offset_error_does_not_grow_with_separation(sep_deg):
    """Contrast with the naive fp32 approach this replaces (see the
    session's derivation notebook): atan2(zn, xn) alone, or computing
    absolute RA/Dec in fp32 directly, both degrade well past this as
    separation grows; this formula stays near the fp32 rounding floor
    from sub-degree fields out to >45 deg.
    """
    jax.config.update("jax_enable_x64", False)
    try:
        crval = [37.0, 30.0]

        # A large-crpix WCS whose CD matrix is scaled so the full pixel
        # range spans `sep_deg` of true field of view.
        crpix_half = 4000.0
        cd_wide = [[-sep_deg / (2 * crpix_half), 0.0], [0.0, sep_deg / (2 * crpix_half)]]
        wcs_ref = Tan([crpix_half, crpix_half], cd_wide, crval)
        wcs_32 = TanJax(jnp.asarray([crpix_half, crpix_half]), jnp.asarray(cd_wide), jnp.asarray(crval))

        rng = np.random.default_rng(0)
        x = rng.uniform(0, 2 * crpix_half, 5000)
        y = rng.uniform(0, 2 * crpix_half, 5000)

        ra_ref, dec_ref = wcs_ref.unproj(x, y)
        ra0, dec0 = np.radians(crval[0]), np.radians(crval[1])
        dalpha_ref = np.mod(ra_ref - ra0 + np.pi, 2 * np.pi) - np.pi
        ddelta_ref = dec_ref - dec0

        dalpha_cosdec, ddelta = wcs_32.unproj_offset(jnp.asarray(x), jnp.asarray(y))
        dalpha_cosdec, ddelta = np.asarray(dalpha_cosdec, dtype=np.float64), np.asarray(ddelta, dtype=np.float64)
        dalpha = dalpha_cosdec / np.cos(dec0)

        assert np.max(np.abs(dalpha - dalpha_ref)) < 1e-6
        assert np.max(np.abs(ddelta - ddelta_ref)) < 1e-6
    finally:
        jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# unproj_absolute/proj_absolute must reject a JAX-backed WCS: its crval may
# already be truncated to float32 at construction time if jax_enable_x64
# was off then, and this method cannot recover precision already discarded.
# ---------------------------------------------------------------------------


def test_absolute_rejects_jax_backed_wcs():
    wcs_jax = TanJax(jnp.asarray(CRPIX), jnp.asarray(CD), jnp.asarray([123.4, -37.7]))
    x, y = _sample_pixels(n=10)
    with pytest.raises(TypeError):
        wcs_jax.unproj_absolute(x, y)
    with pytest.raises(TypeError):
        wcs_jax.proj_absolute(np.array([1.0]), np.array([0.5]))


def test_offset_works_on_jax_backed_wcs():
    """unproj_offset/proj_offset, unlike the absolute variants, have no
    such restriction -- that's the whole point.
    """
    wcs_jax = TanJax(jnp.asarray(CRPIX), jnp.asarray(CD), jnp.asarray([123.4, -37.7]))
    x, y = jnp.asarray([500.0, 600.0]), jnp.asarray([500.0, 400.0])
    dalpha_cosdec, ddelta = wcs_jax.unproj_offset(x, y)
    x2, y2 = wcs_jax.proj_offset(dalpha_cosdec, ddelta)
    np.testing.assert_allclose(np.asarray(x2), np.asarray(x), atol=1e-3)
    np.testing.assert_allclose(np.asarray(y2), np.asarray(y), atol=1e-3)
