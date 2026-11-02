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

from mapprojax import (Sin, SinJax, SinJaxArray, SinJaxMixed, Tan, TanJax,
                        TanJaxArray, TanJaxMixed)
from mapprojax.sip import Sip
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
