import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mapprojax.sip import Sip
from mapprojax.wcs import Sin, Tan


CD = np.array([[-1.0e-3, 0.0], [0.0, 1.0e-3]])


@pytest.fixture
def x64_enabled():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("cls", [Tan, Sin])
@pytest.mark.parametrize("precision", ["f32", "mixed", "f64"])
def test_construction_precision_and_reference_leaf(cls, precision, x64_enabled):
    crval = np.array([123.4, -37.7], dtype=np.float64)
    wcs = cls([3.0, 4.0], CD, crval, precision=precision)

    expected = jnp.float64 if precision == "f64" else jnp.float32
    assert wcs.precision == precision
    assert wcs.crpix.dtype == expected
    assert wcs.cd.dtype == expected
    assert wcs.crval is crval
    assert len(jax.tree_util.tree_leaves(wcs)) == 3


def test_traced_constructor_and_assignment_only_unflatten():
    abstract_wcs = jax.eval_shape(
        lambda cd, crval: Tan([0.0, 0.0], cd, crval, precision="f32"),
        jax.ShapeDtypeStruct((2, 2), jnp.float32),
        jax.ShapeDtypeStruct((2,), jnp.float32),
    )
    assert abstract_wcs.cd.shape == (2, 2)
    assert abstract_wcs.crval.shape == (2,)
    assert abstract_wcs.precision == "f32"

    wcs = Tan([0.0, 0.0], CD, [180.0, 30.0], precision="f32")
    rebuilt = jax.eval_shape(lambda value: value, wcs)
    assert rebuilt.crpix.shape == (2,)
    assert rebuilt.cd.dtype == jnp.float32
    assert rebuilt.crval.shape == (2,)
    assert rebuilt.precision == "f32"


def test_concrete_cd_is_checked_after_precision_cast():
    nearly_singular = [[1.0, 1.0], [1.0, 1.0 + 1.0e-8]]
    with pytest.raises(ValueError, match="CD matrix"):
        Tan([0.0, 0.0], nearly_singular, [0.0, 0.0], precision="f32")

    # The same value cannot be inspected during tracing; runtime pipeline
    # validity handles it without a host conversion of the tracer.
    abstract = jax.eval_shape(
        lambda cd: Tan([0.0, 0.0], cd, [0.0, 0.0], precision="f32"),
        jax.ShapeDtypeStruct((2, 2), jnp.float32),
    )
    assert abstract.cd.shape == (2, 2)


@pytest.mark.parametrize("precision", ["mixed", "f64"])
def test_double_precision_modes_require_x64(precision):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="requires float64"):
            Tan([0.0, 0.0], CD, [180.0, 30.0], precision=precision)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_f32_r_matrix_cannot_bypass_rotation_guard():
    wcs = Tan([0.0, 0.0], CD, [180.0, 30.0], precision="f32")
    with pytest.raises(TypeError, match="precision='f32'"):
        _ = wcs.r_matrix


def test_crval_sequences_normalize_once_and_reject_mixed_float_precision():
    wcs = Tan(
        [0.0, 0.0], CD,
        [np.float32(180.0), np.float32(30.0)],
        precision="f32",
    )
    assert isinstance(wcs.crval, np.ndarray)
    assert wcs.crval.shape == (2,)
    assert wcs.crval.dtype == np.float32

    with pytest.raises(TypeError, match="one floating precision"):
        Tan(
            [0.0, 0.0], CD,
            [np.float32(180.0), 30.0],
            precision="f32",
        )

    traced = jax.eval_shape(
        lambda value: Tan(
            [0.0, 0.0], CD, [value[0], value[1]], precision="f32"
        ),
        jax.ShapeDtypeStruct((2,), jnp.float32),
    )
    assert traced.crval.shape == (2,)
    assert traced.crval.dtype == jnp.float32


def test_with_precision_converts_numeric_leaves_and_preserves_crval_and_sip(x64_enabled):
    reference = np.array([180.0, 45.0], dtype=np.float64)
    sip = Sip(a=np.zeros((3, 3)), b=np.zeros((3, 3)), inverse_steps=7)
    source = Tan([1.0, 2.0], CD, reference, sip=sip, precision="f64")
    mixed = source.with_precision("mixed")
    single = mixed.with_precision("f32")

    assert mixed.precision == "mixed"
    assert mixed.crpix.dtype == mixed.cd.dtype == jnp.float32
    assert mixed.crval is reference
    assert mixed.sip.a.dtype == jnp.float32
    assert mixed.sip.inverse_steps == 7
    assert single.precision == "f32"
    assert single.crval is reference


@pytest.mark.parametrize("precision,expected", [
    ("f32", jnp.float32), ("mixed", jnp.float64), ("f64", jnp.float64),
])
def test_offset_scale_shares_reference_trig_dtype(precision, expected, x64_enabled):
    wcs = Tan([0.0, 0.0], CD, [123.4, -37.7], precision=precision)
    scale = wcs.offset_scale()
    expected_scale = jnp.cos(
        jnp.radians(jnp.asarray(wcs.crval[1], dtype=expected))
    )
    assert scale.dtype == expected
    np.testing.assert_array_equal(np.asarray(scale), np.asarray(expected_scale))

    offsets = wcs.unproj_offset(jnp.asarray(1.0), jnp.asarray(2.0))
    assert offsets[0].dtype == expected
    assert offsets[1].dtype == expected


@pytest.mark.parametrize("cls", [Tan, Sin])
@pytest.mark.parametrize("precision", ["f32", "mixed", "f64"])
def test_json_round_trip_preserves_contract_dtype_and_sip(cls, precision, x64_enabled):
    reference = np.array([180.0, 45.0], dtype=np.float64)
    sip = Sip(
        a=np.zeros((3, 3)), b=np.ones((3, 3)), ap=None, bp=np.zeros((3, 3)),
        inverse_steps=5,
    )
    wcs = cls([1.0, 2.0], CD, reference, sip=sip, precision=precision)

    encoded = json.dumps(wcs.to_dict())
    data = json.loads(encoded)
    restored = cls.from_dict(data)

    assert data["type"] == cls.__name__
    assert data["precision"] == precision
    assert data["crval_dtype"] == "float64"
    assert restored.crval.dtype == np.dtype("float64")
    assert restored.sip.ap is None
    assert restored.sip.bp is not None
    assert restored.sip.inverse_steps == 5
    np.testing.assert_array_equal(restored.sip.b, wcs.sip.b)


def test_json_contract_rejects_old_or_mismatched_type():
    wcs = Tan([0.0, 0.0], CD, [180.0, 30.0], precision="f32")
    data = wcs.to_dict()
    with pytest.raises(ValueError, match="precision-aware contract"):
        Tan.from_dict({key: value for key, value in data.items() if key != "crval_dtype"})
    data["type"] = "Sin"
    with pytest.raises(ValueError, match="expected WCS type"):
        Tan.from_dict(data)


def test_json_rejects_tree_cast_per_pixel_dtype_mismatch(x64_enabled):
    wcs = Tan([0.0, 0.0], CD, [180.0, 30.0], precision="f32")
    cast = jax.tree_util.tree_map(lambda value: jnp.asarray(value, dtype=jnp.float64), wcs)
    with pytest.raises(ValueError, match="inconsistent"):
        cast.to_dict()


def test_json_preserves_single_precision_reference_leaf():
    reference = np.array([180.0, 30.0], dtype=np.float32)
    wcs = Tan([0.0, 0.0], CD, reference, precision="f32")
    restored = Tan.from_dict(json.loads(json.dumps(wcs.to_dict())))
    assert restored.crval.dtype == np.dtype("float32")
    np.testing.assert_array_equal(restored.crval, reference)


def test_mixed_crval_gradient_survives_reconstruction(x64_enabled):
    def objective(crval):
        wcs = Tan([0.0, 0.0], CD, crval, precision="mixed")
        return wcs.unproj(jnp.asarray(10.0), jnp.asarray(20.0))[0]

    gradient = jax.grad(objective)(jnp.asarray([180.0, 30.0], dtype=jnp.float64))
    assert np.all(np.isfinite(gradient))
    assert np.any(np.abs(np.asarray(gradient)) > 1e-12)


def test_new_wcs_works_as_jit_argument_and_closed_over(x64_enabled):
    wcs = Tan([0.0, 0.0], CD, [180.0, 30.0], precision="mixed")
    x, y = jnp.asarray([1.0, 2.0]), jnp.asarray([3.0, 4.0])

    def evaluate(obj, px, py):
        return obj.unproj_offset(px, py)

    as_argument = jax.jit(evaluate)(wcs, x, y)
    closed_over = jax.jit(lambda px, py: evaluate(wcs, px, py))(x, y)
    for left, right in zip(as_argument, closed_over):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("cls", [Tan, Sin])
def test_vmap_pixel_grid_paired_samples_and_shared_grid_axes(cls, x64_enabled):
    wcses = [
        cls([0.0, 0.0], CD, centre, precision="f64")
        for centre in ([180.0, 30.0], [40.0, -20.0])
    ]
    batched_wcs = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves), *wcses
    )
    px = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    py = jnp.asarray([-2.0, 0.0, 4.0], dtype=jnp.float64)

    # One WCS over a pixel grid: normal broadcasting and explicit vmap agree.
    broadcast = wcses[0].unproj(px, py)
    one_wcs = jax.vmap(
        lambda x, y: wcses[0].unproj(x, y), in_axes=(0, 0), out_axes=0
    )(px, py)
    for expected, got in zip(broadcast, one_wcs):
        np.testing.assert_array_equal(expected, got)

    # Paired WCSes and samples: one output per input pair, shape (samples,).
    paired_x, paired_y = px[:2], py[:2]
    paired = jax.vmap(
        lambda w, x, y: w.unproj(x, y), in_axes=(0, 0, 0), out_axes=0
    )(batched_wcs, paired_x, paired_y)
    paired_expected = tuple(
        jnp.stack([
            wcs.unproj(x, y)[axis]
            for wcs, x, y in zip(wcses, paired_x, paired_y)
        ])
        for axis in range(2)
    )
    for expected, got in zip(paired_expected, paired):
        np.testing.assert_array_equal(expected, got)

    # Many WCSes over one shared grid: out_axes=1 yields (pixels, centres);
    # the default out_axes=0 yields (centres, pixels).
    shared = jax.vmap(
        lambda w: w.unproj(px, py), in_axes=0, out_axes=1
    )(batched_wcs)
    centres_first = jax.vmap(
        lambda w: w.unproj(px, py), in_axes=0, out_axes=0
    )(batched_wcs)
    assert shared[0].shape == shared[1].shape == (px.size, len(wcses))
    assert centres_first[0].shape == centres_first[1].shape == (len(wcses), px.size)
    for pixels_by_centre, centres_by_pixel in zip(shared, centres_first):
        np.testing.assert_array_equal(pixels_by_centre.T, centres_by_pixel)


def test_numeric_leaf_changes_do_not_change_jit_structure(x64_enabled):
    wcs = Tan([0.0, 0.0], CD, [180.0, 30.0], precision="mixed")
    other = Tan([1.0, 1.0], CD, [180.0, 30.0], precision="mixed")

    def evaluate(obj):
        return obj.unproj_offset(jnp.asarray([1.0]), jnp.asarray([2.0]))

    first = jax.make_jaxpr(evaluate)(wcs)
    second = jax.make_jaxpr(evaluate)(other)
    assert str(first) == str(second)


# --- C4 checkpoint repairs ------------------------------------------------------

import subprocess
import sys
import textwrap
from pathlib import Path

from mapprojax.utils import radec_to_xyz

REPO = Path(__file__).resolve().parents[1]
MODES = ["f32", "mixed", "f64"]


@pytest.mark.parametrize("components,expected", [
    ([np.float32(180.0), np.float32(30.0)], np.float32),
    ([180.0, 30.0], np.float64),
    ([np.float64(180.0), 30.0], np.float64),
    ([180, 30], np.float64),            # integer degrees are promoted, losslessly
    ([np.int32(180), 30], np.float64),
])
def test_crval_sequences_that_keep_every_component_are_accepted(components, expected):
    wcs = Tan([0.0, 0.0], CD, components, precision="f32")
    assert isinstance(wcs.crval, np.ndarray)
    assert wcs.crval.dtype == expected
    np.testing.assert_array_equal(wcs.crval, np.asarray(components, dtype=np.float64))


@pytest.mark.parametrize("components", [
    [np.float32(123.456789), 30],        # int beside f32 assembles to float64
    [np.float32(123.456789), 30.0],
    [np.float32(123.456789), np.float64(30.0)],
    [30.0, np.float32(123.456789)],
], ids=["f32-int", "f32-float", "f32-f64", "float-f32"])
def test_crval_sequence_that_would_promote_a_truncated_component_is_rejected(
        components, x64_enabled):
    # Accepting these would hand absolute rotation an apparently float64
    # reference whose first component is already truncated.
    with pytest.raises(TypeError, match="one floating precision"):
        Tan([0.0, 0.0], CD, components, precision="mixed")


def test_traced_crval_sequence_assembles_on_device_and_rejects_promotion():
    spec = jax.ShapeDtypeStruct((2,), jnp.float32)
    ok = jax.eval_shape(
        lambda v: Tan([0.0, 0.0], CD, [v[0], v[1]], precision="f32"), spec)
    assert ok.crval.dtype == jnp.float32
    with pytest.raises(TypeError, match="one floating precision"):
        jax.eval_shape(
            lambda v: Tan([0.0, 0.0], CD, [v[0], 30.0], precision="f32"), spec)


def test_header_list_and_array_crval_share_one_treedef(x64_enabled):
    from accuracy_reference import load_header
    from mapprojax.wcs_utils import load_wcs_params_from_header

    crpix, cd, crval, sip = load_wcs_params_from_header(load_header("spherex_l2.hdr"))
    assert isinstance(crval, list)
    from_header = Tan(crpix, cd, crval, sip=sip, precision="mixed")
    from_array = Tan(crpix, cd, np.asarray(crval, dtype=np.float64), sip=sip,
                     precision="mixed")
    assert (jax.tree_util.tree_structure(from_header)
            == jax.tree_util.tree_structure(from_array))
    # Same structure, so the two stack for vmap.
    stacked = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]),
                                     from_header, from_array)
    assert stacked.crval.shape == (2, 2)


def test_conversion_that_loses_cd_rank_is_rejected(x64_enabled):
    cast_loss_cd = [[1e-3, 1e-3], [1e-3, 1e-3 * (1 + 1e-8)]]
    wcs = Tan([0.0, 0.0], cast_loss_cd, [10.0, 20.0], precision="f64")
    for precision in ("f32", "mixed"):
        with pytest.raises(ValueError, match="CD matrix"):
            wcs.with_precision(precision)


def test_reconstruction_never_calls_the_constructor(monkeypatch, x64_enabled):
    sip = Sip(a=np.zeros((3, 3)), b=np.zeros((3, 3)), inverse_steps=4)
    wcs = Tan([1.0, 2.0], CD, [10.0, 20.0], sip=sip, precision="mixed")

    def refuse(*args, **kwargs):
        raise AssertionError("unflatten re-entered __init__")

    monkeypatch.setattr(Tan, "__init__", refuse)
    copied = jax.tree_util.tree_map(lambda leaf: leaf, wcs)
    abstract = jax.eval_shape(lambda w: w, wcs)
    out = jax.jit(lambda w: w.unproj_offset(jnp.asarray(1.0), jnp.asarray(2.0)))(wcs)
    assert copied.precision == abstract.precision == "mixed"
    assert copied.sip.inverse_steps == 4 and copied.crval is not None
    assert np.all(np.isfinite(np.asarray(out[0])))


def test_mixed_casts_at_both_rotation_boundaries(x64_enabled):
    # 0.05"/px: an f32 sky coordinate is ~1e-7 rad, i.e. ~0.4 px here, so
    # mixed must rotate f64 sky and cast to f32 only on the native side.
    scale = 0.05 / 3600.0
    cd = [[-scale, 0.0], [0.0, scale]]
    crval = np.array([123.456789, -37.7], dtype=np.float64)
    f64 = Tan([2048.0, 2048.0], cd, crval, precision="f64")
    mixed = Tan([2048.0, 2048.0], cd, crval, precision="mixed")

    x = jnp.asarray([10.0, 2048.0, 4000.0])
    y = jnp.asarray([20.0, 2048.0, 3900.0])
    ra, dec = f64.unproj(x, y)

    xm, ym = mixed.proj(ra, dec)
    assert xm.dtype == ym.dtype == jnp.float32              # native side f32
    assert float(jnp.max(jnp.abs(xm - x))) < 0.05
    assert float(jnp.max(jnp.abs(ym - y))) < 0.05

    native = mixed._celestial_to_native(*radec_to_xyz(ra, dec, xp=jnp))
    assert all(v.dtype == jnp.float32 for v in native)      # cast after rotation
    ra_m, dec_m = mixed.unproj(x, y)
    assert ra_m.dtype == dec_m.dtype == jnp.float64         # cast up before rotation
    np.testing.assert_allclose(np.asarray(ra_m), np.asarray(ra), rtol=0, atol=5e-9)


# Masked-loss gradients on the new classes. Batches mix valid samples with an
# out-of-domain sample (SIN: outside the unit disc; back hemisphere) and NaN.
SIP_A = np.zeros((3, 3)); SIP_A[2, 0] = 2e-6; SIP_A[1, 1] = -1e-6
SIP_B = np.zeros((3, 3)); SIP_B[0, 2] = 2e-6; SIP_B[1, 1] = 1e-6
R_PX = np.degrees(1.0) / CD[1][1]
PIX_X = np.array([30.0, 1.5 * R_PX, np.nan, -40.0])
PIX_Y = np.array([-20.0, 0.0, 5.0, 25.0])
SKY_RA = np.radians(np.array([0.5, 179.0, 0.2, -0.8]))
SKY_DEC = np.radians(np.array([0.3, 0.0, np.nan, -0.4]))

ABSOLUTE = ["proj", "proj_xyz", "unproj", "unproj_xyz"]
NATIVE = ["proj_offset", "unproj_offset", "native_to_pix", "pix_to_native"]


def _masked(out):
    # Not squared: f32 TAN pixels near the horizon would overflow a square.
    return sum(jnp.sum(jnp.where(jnp.isfinite(o), o, 0.0)) for o in out)


def _call(wcs, method, dtype):
    if method in ("unproj", "unproj_xyz", "unproj_offset", "pix_to_native"):
        return getattr(wcs, method)(jnp.asarray(PIX_X, dtype), jnp.asarray(PIX_Y, dtype))
    if method == "proj":
        return wcs.proj(jnp.asarray(SKY_RA), jnp.asarray(SKY_DEC))
    if method == "proj_offset":
        return wcs.proj_offset(jnp.asarray(SKY_RA, dtype), jnp.asarray(SKY_DEC, dtype))
    xyz = radec_to_xyz(jnp.asarray(SKY_RA), jnp.asarray(SKY_DEC), xp=jnp)
    if method == "proj_xyz":
        return wcs.proj_xyz(*xyz)
    return wcs.native_to_pix(*(v.astype(dtype) for v in xyz))


def _cases():
    for cls in (Tan, Sin):
        for precision in MODES:
            methods = NATIVE if precision == "f32" else NATIVE + ABSOLUTE
            for method in methods:
                yield pytest.param(cls, precision, method,
                                   id=f"{cls.__name__}-{precision}-{method}")


@pytest.mark.parametrize("with_sip", [False, True], ids=["nosip", "sip"])
@pytest.mark.parametrize("cls,precision,method", list(_cases()))
def test_new_class_masked_loss_gradients_are_finite(cls, precision, method,
                                                    with_sip, x64_enabled):
    dtype = jnp.float64 if precision == "f64" else jnp.float32
    ref_dtype = jnp.float32 if precision == "f32" else jnp.float64

    def build(p):
        sip = Sip(a=p["a"], b=p["b"]) if with_sip else None
        return cls(p["crpix"], p["cd"], p["crval"], sip=sip, precision=precision)

    params = {"crpix": jnp.asarray([3.0, -2.0], dtype), "cd": jnp.asarray(CD, dtype),
              "crval": jnp.asarray([10.0, 20.0], ref_dtype),
              "a": jnp.asarray(SIP_A, dtype), "b": jnp.asarray(SIP_B, dtype)}
    out = _call(build(params), method, dtype)
    valid = np.isfinite(np.asarray(out[0]))
    assert valid.any() and not valid.all(), (method, out)

    grads = jax.grad(lambda p: _masked(_call(build(p), method, dtype)))(params)
    for name, g in grads.items():
        assert np.all(np.isfinite(np.asarray(g))), f"{precision} {method} d/d{name}: {g}"


@pytest.mark.parametrize("precision", MODES)
def test_new_class_valid_sample_gradient_ignores_invalid_neighbours(precision, x64_enabled):
    dtype = jnp.float64 if precision == "f64" else jnp.float32
    ref_dtype = jnp.float32 if precision == "f32" else jnp.float64

    def loss(cd, xs, ys):
        w = Sin(jnp.zeros(2, dtype), cd, jnp.asarray([10.0, 20.0], ref_dtype),
                precision=precision)
        return _masked(w.unproj_offset(xs, ys))

    ok = np.isfinite(PIX_X) & (np.abs(PIX_X) < R_PX)
    cd = jnp.asarray(CD, dtype)
    full = jax.grad(loss)(cd, jnp.asarray(PIX_X, dtype), jnp.asarray(PIX_Y, dtype))
    alone = jax.grad(loss)(cd, jnp.asarray(PIX_X[ok], dtype), jnp.asarray(PIX_Y[ok], dtype))
    rtol = 1e-12 if precision == "f64" else 1e-6
    np.testing.assert_allclose(np.asarray(full), np.asarray(alone), rtol=rtol, atol=0)


def test_numeric_leaf_changes_reuse_the_compiled_function(x64_enabled):
    """Rule 4: changed numbers reuse the trace; changed static data retraces."""
    traces = []

    @jax.jit
    def evaluate(obj, px, py):
        traces.append(1)
        return obj.unproj_offset(px, py)

    x, y = jnp.asarray([1.0, 2.0]), jnp.asarray([3.0, 4.0])
    sip = Sip(a=SIP_A, b=SIP_B, inverse_steps=3)
    for cls in (Tan, Sin):
        for precision in MODES:
            traces.clear()
            ref = np.array([180.0, 30.0], dtype=np.float32 if precision == "f32"
                           else np.float64)
            first = cls([0.0, 0.0], CD, ref, sip=sip, precision=precision)
            moved = cls([1.0, -1.0], np.asarray(CD) * 1.01, ref + 1, sip=Sip(
                a=SIP_A * 2, b=SIP_B, inverse_steps=3), precision=precision)
            evaluate(first, x, y)
            evaluate(moved, x, y)
            assert len(traces) == 1, (cls.__name__, precision)
            # Positive control: inverse_steps is static aux data.
            other_steps = cls([0.0, 0.0], CD, ref, sip=Sip(
                a=SIP_A, b=SIP_B, inverse_steps=2), precision=precision)
            evaluate(other_steps, x, y)
            assert len(traces) == 2, (cls.__name__, precision)


SIP_CONFIGS = {
    "nosip": None,
    "ab_only": dict(a=SIP_A, b=SIP_B, inverse_steps=3),
    "ap_bp_seeded": dict(a=SIP_A, b=SIP_B, ap=-SIP_A, bp=-SIP_B, inverse_steps=5),
}


@pytest.mark.parametrize("sip_config", SIP_CONFIGS, ids=list(SIP_CONFIGS))
@pytest.mark.parametrize("precision", MODES)
@pytest.mark.parametrize("cls", [Tan, Sin])
def test_json_round_trip_preserves_outputs(cls, precision, sip_config, x64_enabled):
    config = SIP_CONFIGS[sip_config]
    sip = None if config is None else Sip(**config)
    wcs = cls([3.0, -2.0], CD, np.array([10.0, 20.0]), sip=sip, precision=precision)
    restored = cls.from_dict(json.loads(json.dumps(wcs.to_dict())))

    assert restored.precision == precision
    assert restored.crval.dtype == wcs.crval.dtype
    assert (restored.sip is None) == (sip is None)
    if sip is not None:
        assert restored.sip.inverse_steps == sip.inverse_steps
        assert (restored.sip.ap is None) == (sip.ap is None)
    x, y = jnp.asarray([5.0, -30.0]), jnp.asarray([7.0, 12.0])
    for left, right in zip(wcs.unproj_offset(x, y), restored.unproj_offset(x, y)):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    if precision != "f32":
        for left, right in zip(wcs.unproj(x, y), restored.unproj(x, y)):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def _run_fresh(code):
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=REPO,
                          capture_output=True, text=True, timeout=300,
                          env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_f32_class_with_x64_off_in_a_fresh_process():
    out = _run_fresh("""
        import json
        import jax, jax.numpy as jnp, numpy as np
        assert jax.config.jax_enable_x64 is False
        from mapprojax.sip import Sip
        from mapprojax.wcs import Tan
        CD = [[-1e-3, 0.0], [0.0, 1e-3]]
        a = np.zeros((3, 3)); a[2, 0] = 2e-6
        w = Tan([3.0, -2.0], CD, [10.0, 20.0], sip=Sip(a=a, b=a.T), precision="f32")
        assert w.crval.dtype == np.float64          # host reference kept as given
        d = w.unproj_offset(jnp.asarray([5.0]), jnp.asarray([7.0]))
        assert d[0].dtype == jnp.float32 and np.all(np.isfinite(np.asarray(d[0])))
        p = w.proj_offset(*d)
        assert np.allclose(np.asarray(p[0]), 5.0, atol=1e-3)
        for name, call in [("unproj", lambda: w.unproj(1.0, 2.0)),
                           ("proj", lambda: w.proj(0.1, 0.2)),
                           ("r_matrix", lambda: w.r_matrix)]:
            try:
                call()
            except TypeError:
                continue
            raise SystemExit(name + " did not raise")
        for mode in ("mixed", "f64"):
            try:
                Tan([0.0, 0.0], CD, [10.0, 20.0], precision=mode)
            except ValueError:
                continue
            raise SystemExit(mode + " constructed with x64 off")
        # JSON written with x64 off reloads the reference with NumPy, so the
        # float64 host reference survives; the per-pixel leaves stay f32.
        r = Tan.from_dict(json.loads(json.dumps(w.to_dict())))
        assert isinstance(r.crval, np.ndarray) and r.crval.dtype == np.float64
        assert r.cd.dtype == jnp.float32
        # A reference truncated by jit with x64 off is recorded as float32.
        t = jax.jit(lambda o: o)(w)
        assert t.to_dict()["crval_dtype"] == "float32"
        print("ok")
    """)
    assert "ok" in out


@pytest.mark.parametrize("precision", ["mixed", "f64"])
def test_guards_after_x64_is_switched_off(precision):
    out = _run_fresh(f"""
        import jax, jax.numpy as jnp, numpy as np
        jax.config.update("jax_enable_x64", True)
        from mapprojax.wcs import Tan
        w = Tan([0.0, 0.0], [[-1e-3, 0.0], [0.0, 1e-3]], [10.0, 20.0],
                precision="{precision}")
        jax.config.update("jax_enable_x64", False)
        x, y = jnp.asarray([1.0]), jnp.asarray([2.0])
        must_raise = {{
            "unproj": lambda: w.unproj(x, y),
            "proj": lambda: w.proj(x * 0.01, y * 0.01),
            "unproj_offset": lambda: w.unproj_offset(x, y),
            "proj_offset": lambda: w.proj_offset(x * 1e-4, y * 1e-4),
            "offset_scale": lambda: w.offset_scale(),
        }}
        if "{precision}" == "f64":
            must_raise["pix_to_native"] = lambda: w.pix_to_native(x, y)
        for name, call in must_raise.items():
            try:
                call()
            except (TypeError, ValueError):
                continue
            raise SystemExit(name + " did not raise")
        if "{precision}" == "mixed":
            n = w.pix_to_native(x, y)                 # f32 native side still works
            assert np.all(np.isfinite(np.asarray(n[0])))
        print("ok")
    """)
    assert "ok" in out
