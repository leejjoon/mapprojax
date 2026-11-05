"""Focused tests for Reprojector.pixel_map's finite coordinate contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mapprojax.reproject import Reprojector
from mapprojax.sip import Sip
from mapprojax.wcs import Sin, Tan


@pytest.fixture(autouse=True)
def _x64_enabled():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


CD = [[-1e-3, 0.0], [0.0, 1e-3]]


def _wcs(cls=Tan, crval=(12.0, 34.0), *, precision="f32", crpix=(0.0, 0.0),
         cd=CD, sip=None):
    return cls(crpix, cd, crval, sip=sip, precision=precision)


@pytest.mark.parametrize(
    "precision,dtype", [("f32", np.float32), ("f64", np.float64)]
)
def test_pixel_map_direction_dtype_broadcast_and_validity(precision, dtype):
    src = _wcs(crval=(12.0, 34.0), precision=precision)
    tgt = _wcs(crval=(12.02, 34.01), precision=precision)
    forward = Reprojector(src, tgt, matrix_precision=precision)
    reverse = Reprojector(tgt, src, matrix_precision=precision)

    x = jnp.asarray([[0.0], [2.0]], dtype=dtype)
    y = jnp.asarray([0.0, 1.0, 3.0], dtype=dtype)
    x_mapped, y_mapped, valid = forward.pixel_map(x, y)

    assert x_mapped.shape == y_mapped.shape == valid.shape == (2, 3)
    assert x_mapped.dtype == y_mapped.dtype == jnp.dtype(dtype)
    assert valid.dtype == jnp.bool_
    assert np.all(np.asarray(valid))
    assert not np.allclose(np.asarray(x_mapped), np.broadcast_to(x, (2, 3)))

    inverse = reverse.pixel_map(x_mapped[0, 0], y_mapped[0, 0])
    np.testing.assert_allclose(np.asarray(inverse[0]), 0.0, rtol=0, atol=2e-3)
    np.testing.assert_allclose(np.asarray(inverse[1]), 0.0, rtol=0, atol=2e-3)
    assert bool(inverse[2])


def test_invalid_projection_and_nonfinite_inputs_have_zero_standins():
    src = _wcs(crval=(0.0, 0.0))
    tgt = _wcs(crval=(80.0, 0.0))
    rp = Reprojector(src, tgt)
    x, y, valid = rp.pixel_map(jnp.asarray([0.0, 150000.0]),
                               jnp.asarray([0.0, 0.0]))
    assert np.asarray(valid)[0]
    assert not np.asarray(valid)[1]  # source point falls behind target TAN
    assert (float(x[1]), float(y[1])) == (0.0, 0.0)
    assert np.isfinite(np.asarray(x)).all() and np.isfinite(np.asarray(y)).all()

    nan_x, nan_y, nan_valid = rp.pixel_map(
        jnp.asarray([jnp.nan, jnp.inf]), jnp.asarray([0.0, 0.0])
    )
    assert not np.asarray(nan_valid).any()
    np.testing.assert_array_equal(np.asarray(nan_x), [0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(nan_y), [0.0, 0.0])

    # This API does not change the existing public NaN contract.
    old_x, old_y = rp.pix_to_pix(jnp.asarray([150000.0]), jnp.asarray([0.0]))
    assert np.isnan(np.asarray(old_x)[0]) and np.isnan(np.asarray(old_y)[0])


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_sin_exact_limb_is_valid_without_testing_its_derivative(precision):
    deg_per_radian = 180.0 / np.pi
    sin = _wcs(
        Sin, precision=precision, cd=[[deg_per_radian, 0.0], [0.0, deg_per_radian]]
    )
    rp = Reprojector.from_matrix(
        sin, sin, jnp.eye(3, dtype=jnp.float64 if precision == "f64" else jnp.float32),
        matrix_precision=precision,
    )
    x, y, valid = rp.pixel_map(jnp.asarray(1.0), jnp.asarray(0.0))
    assert bool(valid)
    assert np.isfinite(float(x)) and np.isfinite(float(y))


def test_sin_outside_disc_is_invalid():
    deg_per_radian = 180.0 / np.pi
    sin = _wcs(
        Sin, cd=[[deg_per_radian, 0.0], [0.0, deg_per_radian]]
    )
    x, y, valid = Reprojector(sin, sin).pixel_map(2.0, 0.0)
    assert not bool(valid)
    assert (float(x), float(y)) == (0.0, 0.0)


@pytest.mark.parametrize("projection", [Tan, Sin])
def test_nonfinite_target_crpix_is_zero_masked_and_pix_to_pix_unchanged(projection):
    src = _wcs()
    target = _wcs(projection, crpix=(np.inf, 0.0))
    rp = Reprojector(src, target)
    old_x, old_y = rp.pix_to_pix(0.0, 0.0)
    out_x, out_y, out_valid = rp.pixel_map(0.0, 0.0)
    assert np.isinf(float(old_x)) and np.isfinite(float(old_y))
    assert not bool(out_valid)
    assert (float(out_x), float(out_y)) == (0.0, 0.0)

    def masked_loss(points):
        x, y, valid = rp.pixel_map(points[:, 0], points[:, 1])
        return jnp.sum(jnp.where(valid, x + 0.3 * y, 0.0))

    grad = jax.grad(masked_loss)(jnp.asarray([[0.0, 0.0]], dtype=jnp.float32))
    assert np.isfinite(np.asarray(grad)).all()
    np.testing.assert_array_equal(np.asarray(grad), np.zeros((1, 2), dtype=np.float32))


def test_invalid_traced_cd_sip_and_nonfinite_final_coordinates():
    src = _wcs()

    @jax.jit
    def bad_cd_map(cd):
        target = Tan([0.0, 0.0], cd, [12.0, 34.0], precision="f32")
        rp = Reprojector.from_matrix(src, target, jnp.eye(3, dtype=jnp.float32))
        return rp.pixel_map(jnp.asarray(0.0), jnp.asarray(0.0))

    cd_x, cd_y, cd_valid = bad_cd_map(
        jnp.asarray([[1e-3, 0.0], [1e-3, 0.0]], dtype=jnp.float32)
    )
    assert not bool(cd_valid)
    assert (float(cd_x), float(cd_y)) == (0.0, 0.0)

    singular_a = np.zeros((2, 2))
    singular_a[1, 0] = -1.0
    bad_sip = _wcs(sip=Sip(a=singular_a, b=np.zeros((2, 2))))
    sip_x, sip_y, sip_valid = Reprojector(src, bad_sip).pixel_map(0.0, 0.0)
    assert not bool(sip_valid)
    assert (float(sip_x), float(sip_y)) == (0.0, 0.0)

    nonfinite_target = _wcs(crpix=(np.inf, 0.0))
    out_x, out_y, out_valid = Reprojector(src, nonfinite_target).pixel_map(0.0, 0.0)
    assert not bool(out_valid)
    assert (float(out_x), float(out_y)) == (0.0, 0.0)


class _IdentityLeg:
    """Exact coordinate-preserving leg for testing the half-open edge rule."""

    def with_precision(self, precision):
        self.dtype = jnp.float64 if precision == "f64" else jnp.float32
        return self

    def _pix_to_native_valid(self, x, y):
        x = jnp.asarray(x, dtype=self.dtype)
        y = jnp.asarray(y, dtype=self.dtype)
        valid = jnp.isfinite(x) & jnp.isfinite(y)
        return jnp.ones_like(x), jnp.where(valid, x, 0), jnp.where(valid, y, 0), valid

    def _native_to_pix_valid(self, xn, yn, zn, valid=None):
        finite = jnp.isfinite(xn) & jnp.isfinite(yn) & jnp.isfinite(zn)
        valid = finite if valid is None else jnp.asarray(valid, dtype=jnp.bool_) & finite
        return yn, zn, valid


def _identity_reprojector(precision):
    leg = _IdentityLeg()
    return Reprojector.from_matrix(
        leg, leg, jnp.eye(3, dtype=jnp.float64 if precision == "f64" else jnp.float32),
        matrix_precision=precision,
    )


@pytest.mark.parametrize("precision,dtype", [("f32", np.float32), ("f64", np.float64)])
def test_half_open_rectangular_and_one_by_one_bounds(precision, dtype):
    rp = _identity_reprojector(precision)

    def edge_values(lower, upper):
        return np.asarray([
            np.nextafter(dtype(lower), dtype(-np.inf)),
            dtype(lower),
            np.nextafter(dtype(lower), dtype(np.inf)),
            np.nextafter(dtype(upper), dtype(-np.inf)),
            dtype(upper),
            np.nextafter(dtype(upper), dtype(np.inf)),
        ], dtype=dtype)

    x = edge_values(-0.5, 4.5)
    y = np.zeros_like(x)
    got_x, got_y, valid = rp.pixel_map(x, y, bounds_shape=(3, 5))
    np.testing.assert_array_equal(np.asarray(valid), [False, True, True, True, False, False])
    np.testing.assert_array_equal(np.asarray(got_x)[~np.asarray(valid)], 0)
    np.testing.assert_array_equal(np.asarray(got_y)[~np.asarray(valid)], 0)
    np.testing.assert_array_equal(np.asarray(got_x)[np.asarray(valid)], x[np.asarray(valid)])

    y_edges = edge_values(-0.5, 2.5)
    _, _, valid_y = rp.pixel_map(np.zeros_like(y_edges), y_edges, bounds_shape=(3, 5))
    np.testing.assert_array_equal(
        np.asarray(valid_y), [False, True, True, True, False, False]
    )

    x1 = edge_values(-0.5, 0.5)
    y1 = edge_values(-0.5, 0.5)
    _, _, valid1 = rp.pixel_map(x1, y1, bounds_shape=(1, 1))
    np.testing.assert_array_equal(
        np.asarray(valid1), [False, True, True, True, False, False]
    )


@pytest.mark.parametrize(
    "bounds_shape,error", [
        ([2, 3], TypeError), ((2,), ValueError), ((2, 3, 4), ValueError),
        ((np.int64(2), 3), TypeError), ((True, 3), TypeError), ((2.0, 3), TypeError),
        ((0, 3), ValueError), ((2, -1), ValueError),
    ],
)
def test_bounds_shape_is_strictly_validated(bounds_shape, error):
    with pytest.raises(error, match="bounds_shape"):
        _identity_reprojector("f32").pixel_map(0.0, 0.0, bounds_shape=bounds_shape)


def test_bounds_shape_is_static_under_jit_and_stacked_vmap_preserves_leaves():
    rp = Reprojector(_wcs(), _wcs())
    with pytest.raises(TypeError, match="bounds_shape"):
        jax.jit(lambda shape: rp.pixel_map(0.0, 0.0, bounds_shape=shape))(
            (jnp.asarray(2), jnp.asarray(3))
        )

    stacked = jax.tree_util.tree_map(
        lambda value: jnp.stack([value, value]), rp
    )
    x = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    y = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    mapped = jax.vmap(
        lambda one: one.pixel_map(x, y, bounds_shape=(2, 2))
    )(stacked)
    assert mapped[0].shape == mapped[1].shape == mapped[2].shape == (2, 2)
    assert np.all(np.asarray(mapped[2]))


def test_pixel_map_supports_jit_vmap_and_f32_kernel_has_no_f64_arithmetic():
    rp = Reprojector(_wcs(), _wcs(crval=(12.01, 34.0)))
    x = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    y = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)

    empty = rp.pixel_map(jnp.empty((0,), dtype=jnp.float32),
                         jnp.empty((0,), dtype=jnp.float32))
    assert all(value.shape == (0,) for value in empty)
    assert empty[0].dtype == empty[1].dtype == jnp.float32
    assert empty[2].dtype == jnp.bool_

    compiled = jax.jit(lambda xx, yy: rp.pixel_map(xx, yy, bounds_shape=(4, 4)))
    jit_result = compiled(x, y)
    vmap_result = jax.vmap(
        lambda xx, yy: rp.pixel_map(xx, yy, bounds_shape=(4, 4))
    )(x, y)
    for jit_value, vmap_value in zip(jit_result, vmap_result):
        np.testing.assert_allclose(np.asarray(jit_value), np.asarray(vmap_value))

    closed = jax.make_jaxpr(
        lambda xx, yy: rp.pixel_map(xx, yy, bounds_shape=(4, 4))
    )(x, y)

    def float64_operations(jaxpr):
        found = []
        for equation in jaxpr.eqns:
            values = list(equation.invars) + list(equation.outvars)
            if any(getattr(getattr(value, "aval", None), "dtype", None) == jnp.float64
                   for value in values):
                found.append(equation.primitive.name)
            for parameter in equation.params.values():
                nested = parameter if isinstance(parameter, (tuple, list)) else (parameter,)
                for item in nested:
                    if hasattr(item, "eqns"):
                        found.extend(float64_operations(item))
                    elif hasattr(getattr(item, "jaxpr", None), "eqns"):
                        found.extend(float64_operations(item.jaxpr))
        return found

    assert not float64_operations(closed)


def test_pixel_map_numeric_leaves_reuse_trace_and_static_shape_retraces():
    traces = []

    @jax.jit
    def evaluate(obj, px, py):
        traces.append(1)
        return obj.pixel_map(px, py, bounds_shape=(4, 4))

    x = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    y = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    first = Reprojector(_wcs(), _wcs())
    moved = Reprojector(
        _wcs(crpix=(1.0, -1.0)), _wcs(crpix=(1.0, -1.0))
    )
    evaluate(first, x, y)
    evaluate(moved, x, y)
    assert len(traces) == 1

    static_precision = Reprojector(
        _wcs(precision="f64"), _wcs(precision="f64"), matrix_precision="f64"
    )
    evaluate(static_precision, x.astype(jnp.float64), y.astype(jnp.float64))
    assert len(traces) == 2
