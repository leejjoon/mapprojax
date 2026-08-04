"""Tests for Reprojector / compose_relative_rotation (fo64_report_and_plan.md,
Phase 1): fused src-pixel -> tgt-pixel reprojection with no fp64 required on
the per-pixel path.

Accuracy is measured against the "reference" absolute-path pipeline
(``wcs_src.unproj`` -> ``wcs_tgt.proj``, plain NumPy, always float64), for
WCS pairs whose fields genuinely overlap or nearly so -- the regime the
design targets (see Reprojector's docstring: `M_rel` stays close to the
identity when fields overlap, so the bulk per-pixel math never differences
two O(1) quantities). A field separation much larger than the field of view
is a different, geometry-driven regime (a point far off one WCS's own
boresight projects to an enormous or undefined pixel position in a TAN/SIN
projection, regardless of floating point precision) and is deliberately not
exercised here as an accuracy case.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mapprojax.projections import Tan, Sin
from mapprojax.reproject import Reprojector, SUPPORTED_MATRIX_PRECISIONS
from mapprojax.sip import Sip
from mapprojax.utils import compose_relative_rotation
from mapprojax.jax_projections import TanJax


def make_wcs(cls, crval_deg, plate_scale_deg, crpix_half=400.0, sip=None):
    cd = [[-plate_scale_deg, 0.0], [0.0, plate_scale_deg]]
    return cls([crpix_half, crpix_half], cd, crval_deg, sip=sip)


def reference_pix_to_pix(wcs_src, wcs_tgt, x, y):
    ra, dec = wcs_src.unproj(x, y)
    return wcs_tgt.proj(ra, dec)


def sample_points(crpix_half, n=500, seed=0, margin=0.22):
    rng = np.random.default_rng(seed)
    lo, hi = crpix_half * margin, crpix_half * (2 - margin)
    return rng.uniform(lo, hi, size=n), rng.uniform(lo, hi, size=n)


def _sample_sip(crpix_half):
    # Scaled by 1/crpix_half**2 so the resulting distortion at the field
    # edge stays a fixed, realistic ~0.3-0.5 px regardless of image size --
    # a fixed absolute coefficient (fine for a 400 px half-width) would
    # blow up to thousands of pixels of "distortion" at crpix_half=40000,
    # testing an unrealistic scenario no real SIP file represents.
    scale = 0.4 / crpix_half**2
    a = np.zeros((3, 3)); a[2, 0] = scale; a[0, 2] = scale * 1.3
    b = np.zeros((3, 3)); b[1, 1] = scale * 2.0
    ap = np.zeros((3, 3)); ap[2, 0] = -scale; ap[0, 2] = -scale * 1.3
    bp = np.zeros((3, 3)); bp[1, 1] = -scale * 2.0
    return Sip(a=a, b=b, ap=ap, bp=bp)


# Separation, plate scale, and crpix_half chosen together so the two fields
# genuinely overlap (a real, if extreme, mosaicking scenario) -- not so the
# separation matches some fixed absolute number in isolation. Error bounds
# have headroom over what was actually measured (see PR/commit description);
# they pin the *order of magnitude*, not a brittle exact value.
OVERLAPPING_CASES = [
    # (sep_deg, plate_arcsec, crpix_half, max_f32_err_px)
    (0.3, 3.6, 400.0, 5e-4),
    (1.0, 3.6, 400.0, 2e-3),
    (5.0, 3.6, 4000.0, 5e-3),
    (20.0, 3.6, 4000.0, 2e-2),
    (1.0, 0.1, 40000.0, 5e-2),
    (5.0, 0.1, 40000.0, 1e-1),
]

WCS_PAIRS = [(Tan, Tan), (Tan, Sin), (Sin, Tan)]


@pytest.fixture(autouse=True)
def _x64_enabled():
    # Only the reference path and the "f64" matrix_precision mode need
    # this; Reprojector's default "f32" mode is designed to work
    # regardless (see test_works_without_jax_enable_x64), and each test
    # that cares about the disabled case flips it explicitly.
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("wcs_src_cls,wcs_tgt_cls", WCS_PAIRS)
@pytest.mark.parametrize("sep_deg,plate_arcsec,crpix_half,max_err", OVERLAPPING_CASES)
@pytest.mark.parametrize("use_sip", [False, True])
def test_f32_matches_reference_for_overlapping_fields(
    wcs_src_cls, wcs_tgt_cls, sep_deg, plate_arcsec, crpix_half, max_err, use_sip
):
    plate_deg = plate_arcsec / 3600.0
    sip = _sample_sip(crpix_half) if use_sip else None
    wcs_src = make_wcs(wcs_src_cls, [180.0, 30.0], plate_deg, crpix_half, sip=sip)
    wcs_tgt = make_wcs(wcs_tgt_cls, [180.0, 30.0 + sep_deg], plate_deg, crpix_half, sip=sip)

    x, y = sample_points(crpix_half)
    x_ref, y_ref = reference_pix_to_pix(wcs_src, wcs_tgt, x, y)

    rp = Reprojector(wcs_src, wcs_tgt, matrix_precision="f32")
    x_out, y_out = rp.pix_to_pix(x, y)

    err = np.sqrt((np.asarray(x_out) - x_ref) ** 2 + (np.asarray(y_out) - y_ref) ** 2)
    assert np.max(err) <= max_err


def test_f32_error_grows_with_separation():
    """Qualitative trend from the report, not a hardcoded number: a larger
    field-center separation should not be *more* accurate than a smaller
    one at the same plate scale/FOV.
    """
    plate_deg = 3.6 / 3600.0
    crpix_half = 4000.0
    errs = []
    for sep_deg in (1.0, 5.0, 20.0):
        wcs_src = make_wcs(Tan, [180.0, 30.0], plate_deg, crpix_half)
        wcs_tgt = make_wcs(Tan, [180.0, 30.0 + sep_deg], plate_deg, crpix_half)
        x, y = sample_points(crpix_half)
        x_ref, y_ref = reference_pix_to_pix(wcs_src, wcs_tgt, x, y)
        rp = Reprojector(wcs_src, wcs_tgt, matrix_precision="f32")
        x_out, y_out = rp.pix_to_pix(x, y)
        err = np.sqrt((np.asarray(x_out) - x_ref) ** 2 + (np.asarray(y_out) - y_ref) ** 2)
        errs.append(np.max(err))
    assert errs[0] <= errs[1] <= errs[2] * 2  # monotonic-ish, with slack for sampling noise


@pytest.mark.parametrize("wcs_src_cls,wcs_tgt_cls", WCS_PAIRS)
def test_f64_matches_reference_near_machine_precision(wcs_src_cls, wcs_tgt_cls):
    """matrix_precision="f64" should reproduce the current (pre-Phase-1)
    absolute-path behavior almost exactly -- this is the strongest sanity
    check that compose_relative_rotation/Reprojector's geometry is correct
    at all, independent of any fp32 accuracy question.
    """
    plate_deg = 3.6 / 3600.0
    wcs_src = make_wcs(wcs_src_cls, [180.0, 30.0], plate_deg, 400.0)
    wcs_tgt = make_wcs(wcs_tgt_cls, [180.0, 35.0], plate_deg, 400.0)

    x, y = sample_points(400.0)
    x_ref, y_ref = reference_pix_to_pix(wcs_src, wcs_tgt, x, y)

    rp = Reprojector(wcs_src, wcs_tgt, matrix_precision="f64")
    x_out, y_out = rp.pix_to_pix(x, y)

    np.testing.assert_allclose(np.asarray(x_out), x_ref, atol=1e-6)
    np.testing.assert_allclose(np.asarray(y_out), y_ref, atol=1e-6)


def test_works_without_jax_enable_x64():
    """Acceptance criterion from the plan: reprojection accuracy <= 1e-3 px
    vs the fp64 reference for overlapping fields, without jax_enable_x64.
    """
    jax.config.update("jax_enable_x64", False)
    try:
        plate_deg = 3.6 / 3600.0
        wcs_src = make_wcs(Tan, [180.0, 30.0], plate_deg, 400.0)
        wcs_tgt = make_wcs(Tan, [180.3, 30.0], plate_deg, 400.0)
        x, y = sample_points(400.0)
        x_ref, y_ref = reference_pix_to_pix(wcs_src, wcs_tgt, x, y)

        rp = Reprojector(wcs_src, wcs_tgt, matrix_precision="f32")
        x_out, y_out = rp.pix_to_pix(x, y)
        err = np.sqrt((np.asarray(x_out) - x_ref) ** 2 + (np.asarray(y_out) - y_ref) ** 2)
        assert np.max(err) <= 1e-3
    finally:
        jax.config.update("jax_enable_x64", True)


def test_no_float64_ops_in_jaxpr():
    """Acceptance criterion from the plan: no fp64 ops in the jaxpr of
    pix_to_pix, for matrix_precision="f32" -- checked with
    jax_enable_x64=True precisely because that's the harder case (with
    x64 off, JAX would truncate any accidental float64 request anyway).
    """
    plate_deg = 3.6 / 3600.0
    wcs_src = make_wcs(Tan, [180.0, 30.0], plate_deg, 400.0)
    wcs_tgt = make_wcs(Tan, [180.3, 30.0], plate_deg, 400.0)
    rp = Reprojector(wcs_src, wcs_tgt, matrix_precision="f32")

    jaxpr = jax.make_jaxpr(rp.pix_to_pix)(jnp.array([300.0, 500.0]), jnp.array([300.0, 500.0]))
    dtypes = {v.aval.dtype for eqn in jaxpr.jaxpr.eqns for v in eqn.outvars}
    assert jnp.float64 not in dtypes, f"unexpected float64 in jaxpr: {dtypes}"


def test_rejects_jax_backed_wcs():
    """compose_relative_rotation (and therefore Reprojector) must run on
    NumPy-backed WCS objects: a JAX one's r_matrix may already be
    truncated to float32 if x64 was never enabled, silently defeating the
    whole point of composing M_rel in true float64.
    """
    wcs_src_np = make_wcs(Tan, [180.0, 30.0], 3.6 / 3600.0, 400.0)
    wcs_tgt_jax = TanJax([400.0, 400.0], [[-1e-3, 0.0], [0.0, 1e-3]], [180.3, 30.0])

    with pytest.raises(TypeError):
        compose_relative_rotation(wcs_src_np, wcs_tgt_jax)
    with pytest.raises(TypeError):
        Reprojector(wcs_src_np, wcs_tgt_jax)


def test_invalid_matrix_precision_raises():
    wcs_src = make_wcs(Tan, [180.0, 30.0], 3.6 / 3600.0, 400.0)
    wcs_tgt = make_wcs(Tan, [180.3, 30.0], 3.6 / 3600.0, 400.0)
    with pytest.raises(ValueError):
        Reprojector(wcs_src, wcs_tgt, matrix_precision="f32x2")
    assert "f32x2" not in SUPPORTED_MATRIX_PRECISIONS  # removed -- see Reprojector's docstring Note


def test_frame_rotation_is_applied():
    """A nontrivial frame_rotation (e.g. standing in for an ICRS<->Galactic
    conversion) must actually change the result, and the change must match
    composing it directly into the reference rotation.
    """
    plate_deg = 3.6 / 3600.0
    wcs_src = make_wcs(Tan, [180.0, 30.0], plate_deg, 400.0)
    wcs_tgt = make_wcs(Tan, [180.3, 30.0], plate_deg, 400.0)
    x, y = sample_points(400.0)

    rp_identity = Reprojector(wcs_src, wcs_tgt, matrix_precision="f64")
    x_id, y_id = rp_identity.pix_to_pix(x, y)

    # A small rotation about the z-axis as a stand-in frame_rotation.
    theta = np.radians(2.0)
    frame_rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    rp_rot = Reprojector(wcs_src, wcs_tgt, frame_rotation=frame_rotation, matrix_precision="f64")
    x_rot, y_rot = rp_rot.pix_to_pix(x, y)

    assert np.max(np.abs(np.asarray(x_rot) - np.asarray(x_id))) > 1e-2

    # Cross-check against applying frame_rotation by hand to the reference.
    m_rel_by_hand = wcs_tgt.r_matrix @ frame_rotation @ wcs_src.r_matrix.T
    xn, yn, zn = wcs_src.pix_to_native(x, y)
    v = m_rel_by_hand @ np.stack([xn, yn, zn])
    x_expected, y_expected = wcs_tgt.native_to_pix(v[0], v[1], v[2])
    np.testing.assert_allclose(np.asarray(x_rot), x_expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(y_rot), y_expected, atol=1e-6)


def test_pytree_roundtrip_preserves_dtype():
    """Regression test for the bug this implementation hit and fixed:
    plain TanJax/SinJax's inherited WCSBase.__init__ hardcodes
    dtype=float, so naively reusing them for Reprojector's legs silently
    re-promoted crpix/cd/sip back to float64 (whenever jax_enable_x64 is
    on) on every pytree flatten/unflatten round-trip -- e.g. every time a
    Reprojector is passed through jax.jit. Reprojector's dedicated leg
    classes must not do this.
    """
    plate_deg = 3.6 / 3600.0
    wcs_src = make_wcs(Tan, [180.0, 30.0], plate_deg, 400.0, sip=_sample_sip(400.0))
    wcs_tgt = make_wcs(Tan, [180.3, 30.0], plate_deg, 400.0, sip=_sample_sip(400.0))
    rp = Reprojector(wcs_src, wcs_tgt, matrix_precision="f32")

    leaves, treedef = jax.tree_util.tree_flatten(rp)
    rp2 = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rp2._src_leg.crpix.dtype == jnp.float32
    assert rp2._src_leg.cd.dtype == jnp.float32
    assert rp2._src_leg.sip.a.dtype == jnp.float32
    assert rp2.m_rel.dtype == jnp.float32


def test_jit_over_reprojector_argument():
    """Reprojector as a traced jax.jit argument (not just closed over) --
    the scenario the pytree round-trip regression above guards against in
    practice.
    """
    plate_deg = 3.6 / 3600.0
    wcs_src = make_wcs(Tan, [180.0, 30.0], plate_deg, 400.0)
    wcs_tgt = make_wcs(Tan, [180.3, 30.0], plate_deg, 400.0)
    rp = Reprojector(wcs_src, wcs_tgt, matrix_precision="f32")

    @jax.jit
    def run(rp, x, y):
        return rp.pix_to_pix(x, y)

    x, y = sample_points(400.0, n=10)
    x_out, y_out = run(rp, jnp.asarray(x), jnp.asarray(y))
    x_direct, y_direct = rp.pix_to_pix(x, y)
    # Loose tolerance: jit vs eager can fuse/reassociate the underlying
    # float32 ops slightly differently (ULP-level XLA implementation
    # detail), not a correctness issue -- this test checks that jit-ing a
    # Reprojector argument works and gives essentially the same answer.
    np.testing.assert_allclose(np.asarray(x_out), np.asarray(x_direct), rtol=1e-5, atol=2e-4)
    np.testing.assert_allclose(np.asarray(y_out), np.asarray(y_direct), rtol=1e-5, atol=2e-4)


def test_vmap_over_pix_to_pix():
    plate_deg = 3.6 / 3600.0
    wcs_src = make_wcs(Tan, [180.0, 30.0], plate_deg, 400.0)
    wcs_tgt = make_wcs(Tan, [180.3, 30.0], plate_deg, 400.0)
    rp = Reprojector(wcs_src, wcs_tgt, matrix_precision="f32")

    x, y = sample_points(400.0, n=20)
    batched = jax.vmap(rp.pix_to_pix)(jnp.asarray(x), jnp.asarray(y))
    direct = rp.pix_to_pix(x, y)
    np.testing.assert_allclose(np.asarray(batched[0]), np.asarray(direct[0]))
    np.testing.assert_allclose(np.asarray(batched[1]), np.asarray(direct[1]))
