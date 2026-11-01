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

from mapprojax import Sin, SinJax, Tan, TanJax
from mapprojax.wcs_utils import load_wcs_params_from_header


@pytest.fixture(autouse=True)
def _x64_enabled():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


CD = [[-1e-3, 0.0], [0.0, 1e-3]]

DOMAINS = "Stage 2 projection domains: back hemisphere must not map to a valid pixel"
SINGULAR = "Stage 2 singular or non-finite CD: reject concrete CD at construction"
SIP_INVERSE = "Stage 2 SIP inverse: A/B without AP/BP skips the inverse correction"
PYTREE = "Stage 2 PyTree reconstruction: tree_unflatten must not call __init__"


def _xfail(reason, raises=AssertionError):
    return pytest.mark.xfail(strict=True, raises=raises, reason=reason)


# --- projection domains -----------------------------------------------------

@pytest.mark.parametrize("cls", [Tan, TanJax, Sin, SinJax], ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    "sep_deg",
    [
        pytest.param(135.0, marks=_xfail(DOMAINS)),
        pytest.param(179.99, marks=_xfail(DOMAINS)),
    ],
)
def test_back_hemisphere_is_not_a_valid_pixel(cls, sep_deg):
    wcs = cls([0.0, 0.0], CD, [0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x, y = wcs.proj(np.radians(sep_deg), 0.0)
    assert not np.isfinite(x) and not np.isfinite(y)


@pytest.mark.parametrize("cls", [Tan, TanJax, Sin, SinJax], ids=lambda c: c.__name__)
def test_front_hemisphere_is_finite(cls):
    wcs = cls([0.0, 0.0], CD, [0.0, 0.0])
    x, y = wcs.proj(np.radians(10.0), np.radians(-5.0))
    assert np.isfinite(x) and np.isfinite(y)


def test_tan_native_horizon_is_not_finite():
    # xn == 0 exactly: the tangent plane is at infinity.
    wcs = Tan([0.0, 0.0], CD, [0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x, y = wcs.native_to_pix(np.array(0.0), np.array(1.0), np.array(0.0))
    assert not np.isfinite(x)


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
@pytest.mark.parametrize(
    "kind",
    [pytest.param(k, marks=_xfail(SINGULAR)) for k in BAD_CD],
)
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


@_xfail(SIP_INVERSE)
def test_ab_only_inverse_matches_astropy():
    h = _ab_only_header()
    crpix, cd, crval, sip = load_wcs_params_from_header(h)
    ra, dec = WCS(h).all_pix2world(PIX, 0).T
    x, y = Tan(crpix, cd, crval, sip=sip).proj(np.radians(ra), np.radians(dec))
    np.testing.assert_allclose(np.column_stack([x, y]), PIX, rtol=0, atol=1e-6)


# --- PyTree reconstruction ---------------------------------------------------

@_xfail(PYTREE, raises=TypeError)
def test_eval_shape_on_jax_wcs():
    wcs = TanJax([0.0, 0.0], CD, [0.0, 0.0])
    out = jax.eval_shape(lambda w: w, wcs)
    assert out.crpix.shape == (2,)
