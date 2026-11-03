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


def test_numeric_leaf_changes_do_not_change_jit_structure(x64_enabled):
    wcs = Tan([0.0, 0.0], CD, [180.0, 30.0], precision="mixed")
    other = Tan([1.0, 1.0], CD, [180.0, 30.0], precision="mixed")

    def evaluate(obj):
        return obj.unproj_offset(jnp.asarray([1.0]), jnp.asarray([2.0]))

    first = jax.make_jaxpr(evaluate)(wcs)
    second = jax.make_jaxpr(evaluate)(other)
    assert str(first) == str(second)
