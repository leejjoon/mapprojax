import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
from mapprojax import Tan, Sin


def test_tan_roundtrip():
    wcs = Tan([100.0, 100.0], [[-0.001, 0.0], [0.0, 0.001]], [45.0, 30.0], precision="f64")
    x, y = wcs.proj(np.radians(45.0), np.radians(30.0))
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(100.0)
    ra, dec = wcs.unproj(110.0, 100.0)
    x_res, y_res = wcs.proj(ra, dec)
    assert x_res == pytest.approx(110.0)
    assert y_res == pytest.approx(100.0)


def test_sin_roundtrip():
    wcs = Sin([50.0, 50.0], [[0.01, 0.0], [0.0, 0.01]], [0.0, 0.0], precision="f64")
    ra, dec = wcs.unproj(60.0, 40.0)
    x_res, y_res = wcs.proj(ra, dec)
    assert x_res == pytest.approx(60.0)
    assert y_res == pytest.approx(40.0)


def test_vmap_replaces_tan_array_broadcasting():
    centres = jnp.asarray([[45.0, 30.0], [50.0, 30.0], [55.0, 30.0]])
    pixels = jnp.asarray([[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]])

    def project(crval, pixel):
        wcs = Tan([100.0, 100.0], [[-0.001, 0.0], [0.0, 0.001]], crval, precision="f64")
        return wcs.unproj(pixel[0], pixel[1])

    ra, dec = jax.vmap(project)(centres, pixels)
    np.testing.assert_allclose(np.degrees(np.asarray(ra)), centres[:, 0], atol=1e-10)
    np.testing.assert_allclose(np.degrees(np.asarray(dec)), centres[:, 1], atol=1e-10)


def test_vmap_replaces_sin_array():
    centres = jnp.asarray([[0.0, 0.0], [10.0, 0.0]])

    def project(crval):
        wcs = Sin([50.0, 50.0], [[0.1, 0.0], [0.0, 0.1]], crval, precision="f64")
        return wcs.unproj(50.0, 50.0)

    ra, dec = jax.vmap(project)(centres)
    np.testing.assert_allclose(np.degrees(np.asarray(ra)), centres[:, 0], atol=1e-10)
    np.testing.assert_allclose(np.degrees(np.asarray(dec)), centres[:, 1], atol=1e-10)


def test_singular_cd_error():
    with pytest.raises(ValueError):
        Tan([0, 0], [[0, 0], [0, 0]], [0, 0], precision="f32")
