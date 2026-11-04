"""Offset-form sky coordinates and the host helpers that compose them.

``unproj_offset``/``proj_offset`` never form an absolute (O(1)) longitude or
latitude in the bulk math, so they stay accurate in float32 regardless of
``jax_enable_x64``. ``absolute_from_offset``/``offset_from_absolute`` add the
float64 reference on the host, using the producer's typed ``offset_scale()``.

These tests replace the retired ``unproj_absolute``/``proj_absolute`` methods
with helper compositions, keeping the original budgets.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mapprojax import absolute_from_offset, offset_from_absolute
from mapprojax.sip import Sip
from mapprojax.wcs import Tan, Sin

CRPIX = [512.0, 512.0]
CD = [[-0.001, 0.0], [0.0, 0.001]]
WCS_CLASSES = [Tan, Sin]
MAS_PER_RAD = np.degrees(1.0) * 3.6e6


def _sample_pixels(n=2000, seed=0, lo=50.0, hi=974.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, n), rng.uniform(lo, hi, n)


@pytest.fixture(autouse=True)
def _x64_enabled():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _to_absolute(wcs, x, y):
    return absolute_from_offset(
        wcs.unproj_offset(x, y), crval=wcs.crval, offset_scale=wcs.offset_scale()
    )


def _from_absolute(wcs, lon, lat):
    offset = offset_from_absolute(
        (lon, lat), crval=wcs.crval, offset_scale=wcs.offset_scale()
    )
    return wcs.proj_offset(*offset)


def _separation(lon1, lat1, lon2, lat2):
    """Great-circle separation in radians (haversine, float64)."""
    dlon = np.asarray(lon1, np.float64) - np.asarray(lon2, np.float64)
    dlat = np.asarray(lat1, np.float64) - np.asarray(lat2, np.float64)
    h = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return 2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# float64 helper compositions reproduce unproj/proj to machine precision.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", WCS_CLASSES)
@pytest.mark.parametrize("crval", [[123.4, -37.7], [10.0, 60.0], [200.0, 0.0]])
def test_absolute_from_offset_matches_unproj(cls, crval):
    wcs = cls(CRPIX, CD, crval, precision="f64")
    x, y = _sample_pixels()
    ra_ref, dec_ref = wcs.unproj(x, y)
    ra, dec = _to_absolute(wcs, x, y)
    assert ra.dtype == np.float64 and dec.dtype == np.float64
    np.testing.assert_allclose(ra, ra_ref, atol=1e-10)
    np.testing.assert_allclose(dec, dec_ref, atol=1e-10)


@pytest.mark.parametrize("cls", WCS_CLASSES)
def test_offset_from_absolute_matches_proj(cls):
    wcs = cls(CRPIX, CD, [123.4, -37.7], precision="f64")
    x, y = _sample_pixels()
    ra, dec = wcs.unproj(x, y)
    x_ref, y_ref = wcs.proj(ra, dec)
    x_out, y_out = _from_absolute(wcs, np.asarray(ra), np.asarray(dec))
    np.testing.assert_allclose(x_out, x_ref, atol=1e-6)
    np.testing.assert_allclose(y_out, y_ref, atol=1e-6)


@pytest.mark.parametrize("cls", WCS_CLASSES)
def test_offset_roundtrip(cls):
    wcs = cls(CRPIX, CD, [123.4, -37.7], precision="f64")
    x, y = _sample_pixels()
    x2, y2 = wcs.proj_offset(*wcs.unproj_offset(x, y))
    np.testing.assert_allclose(x2, x, atol=1e-6)
    np.testing.assert_allclose(y2, y, atol=1e-6)


@pytest.mark.parametrize("cls", WCS_CLASSES)
def test_absolute_roundtrip(cls):
    wcs = cls(CRPIX, CD, [123.4, -37.7], precision="f64")
    x, y = _sample_pixels()
    x2, y2 = _from_absolute(wcs, *_to_absolute(wcs, x, y))
    np.testing.assert_allclose(x2, x, atol=1e-6)
    np.testing.assert_allclose(y2, y, atol=1e-6)


# ---------------------------------------------------------------------------
# Longitude wrapping.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("crval_lon", [0.05, 359.95])
def test_ra_wraparound(crval_lon):
    wcs = Tan(CRPIX, CD, [crval_lon, 10.0], precision="f64")
    x = np.array([300.0, 500.0, 700.0])
    y = np.array([500.0, 500.0, 500.0])

    ra, dec = _to_absolute(wcs, x, y)
    assert np.all(ra >= 0.0) and np.all(ra < 2 * np.pi)

    x2, y2 = _from_absolute(wcs, ra, dec)
    np.testing.assert_allclose(x2, x, atol=1e-6)
    np.testing.assert_allclose(y2, y, atol=1e-6)

    ra_ref, dec_ref = wcs.unproj(x, y)
    sep = np.abs(np.mod(ra - np.asarray(ra_ref) + np.pi, 2 * np.pi) - np.pi)
    assert np.max(sep) < 1e-10


def test_ra_wraparound_input_outside_0_2pi():
    """Longitudes given unnormalized, negative or above 2*pi, still map back."""
    wcs = Tan(CRPIX, CD, [0.05, 10.0], precision="f64")
    x, y = np.array([500.0]), np.array([500.0])
    ra, dec = _to_absolute(wcs, x, y)
    for shift in (0.0, -2 * np.pi, 2 * np.pi, 4 * np.pi):
        x_out, y_out = _from_absolute(wcs, ra + shift, dec)
        np.testing.assert_allclose(x_out, x, atol=1e-6)
        np.testing.assert_allclose(y_out, y, atol=1e-6)


# ---------------------------------------------------------------------------
# Near the pole, in float64.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dec0", [89.9, -89.9])
def test_near_pole_f64(dec0):
    wcs = Tan(CRPIX, CD, [45.0, dec0], precision="f64")
    x, y = _sample_pixels()
    ra_ref, dec_ref = wcs.unproj(x, y)
    ra, dec = _to_absolute(wcs, x, y)
    sep = np.abs(np.mod(ra - np.asarray(ra_ref) + np.pi, 2 * np.pi) - np.pi)
    assert np.max(sep) < 1e-8
    np.testing.assert_allclose(dec, dec_ref, atol=1e-8)


# ---------------------------------------------------------------------------
# SIP: the offset methods delegate to pix_to_native/native_to_pix.
# ---------------------------------------------------------------------------


def test_with_sip():
    a = np.zeros((3, 3)); a[2, 0] = 4e-6; a[1, 1] = -2e-6
    b = np.zeros((3, 3)); b[2, 0] = -3e-6; b[1, 1] = 2.5e-6
    ap = np.zeros((3, 3)); ap[2, 0] = -4e-6; ap[1, 1] = 2e-6
    bp = np.zeros((3, 3)); bp[2, 0] = 3e-6; bp[1, 1] = -2.5e-6
    sip = Sip(a=a, b=b, ap=ap, bp=bp)
    wcs = Tan(CRPIX, CD, [123.4, -37.7], sip=sip, precision="f64")
    x, y = _sample_pixels()

    ra_ref, dec_ref = wcs.unproj(x, y)
    ra, dec = _to_absolute(wcs, x, y)
    np.testing.assert_allclose(ra, ra_ref, atol=1e-10)
    np.testing.assert_allclose(dec, dec_ref, atol=1e-10)

    # The AP/BP seed is a crude negation of A/B, so the round trip carries
    # its approximation error (~6.5e-3 px measured) unless the Newton steps
    # remove it; the bound is the original one.
    x2, y2 = _from_absolute(wcs, ra, dec)
    np.testing.assert_allclose(x2, x, atol=0.01)
    np.testing.assert_allclose(y2, y, atol=0.01)


# ---------------------------------------------------------------------------
# float32 without x64: error stays near the fp32 rounding floor, and does not
# grow with field-centre separation. Ground truth is the f64 class, built
# while x64 is on.
# ---------------------------------------------------------------------------


def _offset_truth(wcs_ref, crval, x, y):
    ra_ref, dec_ref = wcs_ref.unproj(x, y)
    ra0, dec0 = np.radians(crval[0]), np.radians(crval[1])
    dalpha_ref = np.mod(np.asarray(ra_ref) - ra0 + np.pi, 2 * np.pi) - np.pi
    return dalpha_ref, np.asarray(dec_ref) - dec0


@pytest.mark.parametrize("cls", WCS_CLASSES)
def test_unproj_offset_accurate_in_fp32_without_x64(cls):
    crval = [123.4, -37.7]
    x, y = _sample_pixels(n=5000)
    dalpha_ref, ddelta_ref = _offset_truth(
        cls(CRPIX, CD, crval, precision="f64"), crval, x, y
    )

    jax.config.update("jax_enable_x64", False)
    wcs_32 = cls(CRPIX, CD, crval, precision="f32")
    dalpha_cosdec, ddelta = wcs_32.unproj_offset(x, y)
    scale = wcs_32.offset_scale()
    assert dalpha_cosdec.dtype == jnp.float32 and scale.dtype == jnp.float32
    dalpha = np.asarray(dalpha_cosdec, np.float64) / np.float64(np.asarray(scale))

    assert np.max(np.abs(dalpha - dalpha_ref)) < 1e-6
    assert np.max(np.abs(np.asarray(ddelta, np.float64) - ddelta_ref)) < 1e-6

    # The same samples composed to absolute coordinates on the host.
    ra, dec = absolute_from_offset(
        (dalpha_cosdec, ddelta), crval=wcs_32.crval, offset_scale=scale
    )
    ra0, dec0 = np.radians(crval)
    assert np.max(np.abs(np.mod(ra - ra0 - dalpha_ref + np.pi, 2 * np.pi) - np.pi)) < 1e-6
    assert np.max(np.abs(dec - dec0 - ddelta_ref)) < 1e-6


@pytest.mark.parametrize("sep_deg", [0.5, 5.0, 20.0, 45.0])
def test_unproj_offset_error_does_not_grow_with_separation(sep_deg):
    """The naive fp32 forms (atan2(zn, xn), or absolute RA/Dec in fp32)
    degrade well past this as separation grows; this one stays near the fp32
    floor from sub-degree fields out to 45 degrees.
    """
    crval = [37.0, 30.0]
    crpix_half = 4000.0
    cd_wide = [[-sep_deg / (2 * crpix_half), 0.0], [0.0, sep_deg / (2 * crpix_half)]]
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 2 * crpix_half, 5000)
    y = rng.uniform(0, 2 * crpix_half, 5000)
    dalpha_ref, ddelta_ref = _offset_truth(
        Tan([crpix_half, crpix_half], cd_wide, crval, precision="f64"), crval, x, y
    )

    jax.config.update("jax_enable_x64", False)
    wcs_32 = Tan([crpix_half, crpix_half], cd_wide, crval, precision="f32")
    dalpha_cosdec, ddelta = wcs_32.unproj_offset(x, y)
    dalpha = (np.asarray(dalpha_cosdec, np.float64)
              / np.float64(np.asarray(wcs_32.offset_scale())))

    assert np.max(np.abs(dalpha - dalpha_ref)) < 1e-6
    assert np.max(np.abs(np.asarray(ddelta, np.float64) - ddelta_ref)) < 1e-6


def test_f32_offset_roundtrip_without_x64():
    jax.config.update("jax_enable_x64", False)
    wcs = Tan(CRPIX, CD, [123.4, -37.7], precision="f32")
    x, y = jnp.asarray([500.0, 600.0]), jnp.asarray([500.0, 400.0])
    x2, y2 = wcs.proj_offset(*wcs.unproj_offset(x, y))
    np.testing.assert_allclose(np.asarray(x2), np.asarray(x), atol=1e-3)
    np.testing.assert_allclose(np.asarray(y2), np.asarray(y), atol=1e-3)


# ---------------------------------------------------------------------------
# Poles with a float32 producer. The helpers must divide by the producer's
# own rounded cosine; a host-recomputed float64 cosine does not undo the
# scaling actually applied, which is catastrophic at the pole. Envelopes:
# 1024 px field at 3.6 arcsec/px, 2000 samples, against the f64 class.
# ---------------------------------------------------------------------------

# Measured on CPU (Nov 4 2026), Tan and Sin, both signs: max separation
# 0.82-0.87 mas at |dec0| = 89.9999 and 18.3-18.4 mas at |dec0| = 90; the
# host-cosine control gives 2.78e3 mas and 4.45e6-4.53e6 mas there.
POLE_ENVELOPE_MAS = {89.9999: 2.0, 90.0: 30.0}
HOST_COSINE_CONTROL_MIN_MAS = {89.9999: 1000.0, 90.0: 1.0e6}


@pytest.mark.parametrize("cls", WCS_CLASSES)
@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("abs_dec0", [89.9999, 90.0])
def test_f32_helpers_at_the_pole(cls, sign, abs_dec0):
    crval = np.asarray([45.0, sign * abs_dec0])
    x, y = _sample_pixels()
    ra_ref, dec_ref = cls(CRPIX, CD, crval, precision="f64").unproj(x, y)

    wcs_32 = cls(CRPIX, CD, crval, precision="f32")
    offset = wcs_32.unproj_offset(x, y)
    scale = wcs_32.offset_scale()
    ra, dec = absolute_from_offset(offset, crval=crval, offset_scale=scale)
    sep_mas = _separation(ra, dec, ra_ref, dec_ref) * MAS_PER_RAD
    assert np.all(np.isfinite(sep_mas))
    assert np.max(sep_mas) < POLE_ENVELOPE_MAS[abs_dec0]

    # Negative control: dividing by a host float64 cosine instead.
    host_cos = np.cos(np.radians(crval[1]))
    lon_bad = np.radians(crval[0]) + np.asarray(offset[0], np.float64) / host_cos
    lat_bad = np.radians(crval[1]) + np.asarray(offset[1], np.float64)
    bad_mas = _separation(lon_bad, lat_bad, ra_ref, dec_ref) * MAS_PER_RAD
    assert np.max(bad_mas) > HOST_COSINE_CONTROL_MIN_MAS[abs_dec0]


# ---------------------------------------------------------------------------
# Helper contract: dtypes, validity, zero scale, host-only, keyword-only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("precision,dtype", [("f32", jnp.float32), ("mixed", jnp.float64), ("f64", jnp.float64)])
@pytest.mark.parametrize("cls", WCS_CLASSES)
def test_helpers_preserve_producer_scale(cls, precision, dtype):
    wcs = cls(CRPIX, CD, np.asarray([359.95, 89.9999]), precision=precision)
    x = jnp.asarray([500.0, 512.0], dtype=dtype)
    y = jnp.asarray([501.0, 512.0], dtype=dtype)
    absolute = _to_absolute(wcs, x, y)
    assert all(value.dtype == np.float64 for value in absolute)
    assert np.all(np.isfinite(absolute[0])) and np.all(np.isfinite(absolute[1]))
    offset = offset_from_absolute(absolute, crval=wcs.crval, offset_scale=wcs.offset_scale())
    assert all(value.dtype == np.dtype(wcs.offset_scale().dtype) for value in offset)


def test_compiled_producer_scale_matches_eager():
    wcs = Tan(CRPIX, CD, [123.4, 89.9999], precision="f32")
    x, y = _sample_pixels(n=64)
    compiled = jax.jit(lambda w, x, y: (*w.unproj_offset(x, y), w.offset_scale()))
    dlon, dlat, scale = compiled(wcs, x, y)
    assert scale.dtype == jnp.float32
    assert np.asarray(scale) == np.asarray(wcs.offset_scale())
    lon, lat = absolute_from_offset((dlon, dlat), crval=wcs.crval, offset_scale=scale)
    lon_e, lat_e = _to_absolute(wcs, x, y)
    np.testing.assert_allclose(lon, lon_e, atol=1e-12)
    np.testing.assert_allclose(lat, lat_e, atol=1e-12)


def test_helpers_reject_mixed_offset_dtype():
    wcs = Tan(CRPIX, CD, [10.0, 30.0], precision="f32")
    with pytest.raises(TypeError):
        absolute_from_offset(
            (np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float32)),
            crval=wcs.crval, offset_scale=wcs.offset_scale(),
        )


def test_python_float_scale_means_float64():
    crval = np.asarray([10.0, 30.0])
    f32_offset = (np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    with pytest.raises(TypeError):
        absolute_from_offset(f32_offset, crval=crval, offset_scale=0.5)
    lon, lat = absolute_from_offset(
        (np.zeros(2), np.zeros(2)), crval=crval, offset_scale=0.5
    )
    np.testing.assert_allclose(lon, np.radians(10.0))


def test_inverse_accepts_float64_absolute_with_float32_scale():
    wcs = Tan(CRPIX, CD, [10.0, 30.0], precision="f32")
    absolute = (np.asarray([np.radians(10.01)]), np.asarray([np.radians(30.01)]))
    offset = offset_from_absolute(absolute, crval=wcs.crval, offset_scale=wcs.offset_scale())
    assert offset[0].dtype == np.float32 and offset[1].dtype == np.float32


def test_helpers_broadcast():
    crval = np.asarray([10.0, 30.0])
    lon, lat = absolute_from_offset(
        (np.zeros((3, 1)), np.zeros((1, 4))), crval=crval, offset_scale=0.5
    )
    assert lon.shape == lat.shape == (3, 4)


def test_helpers_preserve_nan_validity_and_wrap():
    crval = np.asarray([359.0, 30.0], dtype=np.float64)
    scale = np.asarray(1.0, dtype=np.float32)
    lon, lat = absolute_from_offset(
        (np.asarray([np.nan, 0.0, np.inf], dtype=np.float32),
         np.asarray([0.0, 0.0, 0.0], dtype=np.float32)),
        crval=crval, offset_scale=scale,
    )
    assert np.isnan(lon[0]) and np.isnan(lat[0])
    assert np.isnan(lon[2]) and np.isnan(lat[2])
    assert np.isfinite(lon[1]) and np.isfinite(lat[1])
    offset = offset_from_absolute(
        (np.asarray([2.0 * np.pi + 0.1, np.nan]), np.asarray([0.5, 0.5])),
        crval=crval, offset_scale=scale,
    )
    assert np.isfinite(offset[0][0]) and np.isfinite(offset[1][0])
    assert np.isnan(offset[0][1]) and np.isnan(offset[1][1])
    # offset_from_absolute wraps lon - lon0 into [-pi, pi).
    assert abs(offset[0][0]) < np.pi


def test_zero_scale_gives_nan_longitude_not_a_fabricated_value():
    crval = np.asarray([10.0, 90.0])
    lon, lat = absolute_from_offset(
        (np.asarray([1e-4, 0.0]), np.asarray([-1e-4, 0.0])),
        crval=crval, offset_scale=np.float64(0.0),
    )
    assert np.all(np.isnan(lon))
    np.testing.assert_allclose(lat, np.radians(90.0) + np.asarray([-1e-4, 0.0]))


def test_helpers_reject_tracers():
    crval = np.asarray([10.0, 30.0])

    @jax.jit
    def traced_absolute(dlon, dlat):
        return absolute_from_offset((dlon, dlat), crval=crval, offset_scale=0.5)

    with pytest.raises(TypeError, match="host-only"):
        traced_absolute(jnp.zeros(2), jnp.zeros(2))

    @jax.jit
    def traced_crval(c):
        return absolute_from_offset((np.zeros(2), np.zeros(2)), crval=c, offset_scale=0.5)

    with pytest.raises(TypeError, match="host-only"):
        traced_crval(jnp.asarray(crval))


def test_helpers_reject_bad_crval():
    offset = (np.zeros(2), np.zeros(2))
    with pytest.raises(TypeError):
        absolute_from_offset(offset, crval=np.asarray([10.0, 30.0], np.float32), offset_scale=0.5)
    with pytest.raises(ValueError):
        absolute_from_offset(offset, crval=np.asarray([10.0, 30.0, 0.0]), offset_scale=0.5)


def test_crval_and_scale_are_keyword_only():
    """A swapped positional (offset, crval) pair cannot be silently accepted."""
    crval = np.asarray([10.0, 30.0])
    offset = (np.float64(1e-4), np.float64(1e-4))
    with pytest.raises(TypeError):
        absolute_from_offset(crval, offset, 0.5)
    with pytest.raises(TypeError):
        offset_from_absolute(offset, crval, 0.5)


@pytest.mark.parametrize("crval", [
    [np.float32(10.0), 30],
    [np.float32(10.0), 30.0],
    (10.0, np.float32(30.0)),
])
def test_helpers_reject_crval_sequence_with_float32_component(crval):
    offset = (np.zeros(2), np.zeros(2))
    with pytest.raises(TypeError, match="float32 component"):
        absolute_from_offset(offset, crval=crval, offset_scale=0.5)
    with pytest.raises(TypeError, match="float32 component"):
        offset_from_absolute(offset, crval=crval, offset_scale=0.5)


def test_helpers_accept_python_float_crval_sequence():
    lon, lat = absolute_from_offset(
        (np.zeros(1), np.zeros(1)), crval=[10.0, 30.0], offset_scale=0.5
    )
    np.testing.assert_allclose(lon, np.radians(10.0))
