"""Dense-grid comparison of mapprojax against astropy.

Cases, sampling and metrics live in ``tests/accuracy_reference.py`` and are
shared with ``benchmarks/benchmark_accuracy.py``. The fp64 paths are held to a
1e-4 mas great-circle forward budget and a 1e-6 px inverse budget. The mixed
fp32 path is held to its per-case measured caps in
``tests/data/accuracy_baselines.json``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import accuracy_reference as ar
from mapprojax.wcs import Sin, Tan

FORWARD_MAS = ar.FORWARD_MAS
INVERSE_PX = ar.INVERSE_PX


@pytest.fixture(autouse=True)
def _x64_enabled():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


CASES = ar.cases()


def _params(predicate_marks, only=None):
    out = []
    for case in CASES:
        for impl in ar.implementations_for(case):
            if only is not None and not only(impl):
                continue
            marks = predicate_marks(case, impl)
            out.append(pytest.param(case, impl, id=f"{case.name}-{impl}", marks=marks))
    return out




@pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in CASES])
def test_oracle_converges(case):
    stats = ar.oracle_self_check(case)["roundtrip_px"]
    assert stats["n_nonfinite"] == 0
    assert stats["max"] < 1e-8


def test_astropy_overlap_mask_uses_half_open_exact_edges():
    # Literal expectations keep this Astropy-side oracle check independent of
    # Reprojector.pixel_map's candidate bounds predicate.
    x_lo = np.float64(-0.5)
    y_lo = np.float64(-0.5)
    x_hi = np.float64(3.5)
    y_hi = np.float64(2.5)
    points = np.array([
        [x_lo, y_lo],
        [np.nextafter(x_lo, -np.inf), 0.0],
        [np.nextafter(x_lo, np.inf), 0.0],
        [np.nextafter(x_hi, -np.inf), np.nextafter(y_hi, -np.inf)],
        [x_hi, 0.0],
        [np.nextafter(x_hi, np.inf), 0.0],
        [0.0, np.nextafter(y_lo, -np.inf)],
        [0.0, np.nextafter(y_lo, np.inf)],
        [0.0, np.nextafter(y_hi, -np.inf)],
        [0.0, y_hi],
        [0.0, np.nextafter(y_hi, np.inf)],
    ], dtype=np.float64)
    expected = np.array(
        [True, False, True, True, False, False, False, True, True, False, False]
    )
    np.testing.assert_array_equal(ar._overlap_mask(points, (3, 4)), expected)


@pytest.mark.parametrize("direction, expected_overlap", [("forward", 1009), ("reverse", 320)])
def test_pair_reference_overlap_rows_remain_stable(direction, expected_overlap):
    reference = ar.pair_reference(direction)
    assert reference["overlap"].shape == (1090,)
    assert int(reference["overlap"].sum()) == expected_overlap


@pytest.mark.parametrize("case, impl", _params(lambda c, i: ()))
def test_outputs_are_finite_on_valid_pixels(case, impl):
    result = ar.evaluate(case, impl)
    assert result["forward_mas"]["n_nonfinite"] == 0
    assert result["inverse_px"]["n_nonfinite"] == 0


@pytest.mark.parametrize("case, impl", _params(lambda c, i: (), only=lambda i: ar.budget(i, "forward") is not None))
def test_forward_matches_astropy(case, impl):
    stats = ar.evaluate(case, impl)["forward_mas"]
    budget = ar.budget(impl, "forward")
    assert stats["max"] < budget, f"{stats['max']:.3e} mas"


@pytest.mark.parametrize("case, impl", _params(lambda c, i: (), only=lambda i: ar.budget(i, "inverse") is not None))
def test_inverse_matches_astropy(case, impl):
    stats = ar.evaluate(case, impl)["inverse_px"]
    budget = ar.budget(impl, "inverse")
    assert stats["max"] < budget, f"{stats['max']:.3e} px"


# fp32 contract: no fp64 budget, so the fp32 path (``jax_mixed``) is held to
# its own measured floor from ``accuracy_baselines.json`` instead.
FP32 = [pytest.param(c, i, id=f"{c.name}-{i}")
        for c in CASES for i in ar.implementations_for(c) if ar.budget(i, "forward") is None]


@pytest.mark.parametrize("case, impl", FP32)
@pytest.mark.parametrize("metric", list(ar.METRIC_KEYS))
def test_fp32_within_measured_cap(case, impl, metric):
    key, unit = ar.METRIC_KEYS[metric]
    cap = ar.fp32_cap(case, impl, metric)
    stats = ar.evaluate(case, impl)[key]
    assert stats["max"] <= cap, f"{stats['max']:.3e} {unit} > cap {cap} {unit}"


# --- explicit sky -> pixel cases off the pixel grid --------------------------

def _ring(lon0_deg, lat0_deg, sep_deg, n=8):
    """n points at great-circle distance sep_deg around (lon0, lat0)."""
    pa = np.radians(np.arange(n) * 360.0 / n)
    d, lat0, lon0 = np.radians(sep_deg), np.radians(lat0_deg), np.radians(lon0_deg)
    lat = np.arcsin(np.sin(lat0) * np.cos(d) + np.cos(lat0) * np.sin(d) * np.cos(pa))
    lon = lon0 + np.arctan2(np.sin(pa) * np.sin(d) * np.cos(lat0),
                            np.cos(d) - np.sin(lat0) * np.sin(lat))
    return lon, lat


@pytest.mark.parametrize("proj", ["TAN", "SIN"])
@pytest.mark.parametrize("impl", ["numpy", "jax"])
@pytest.mark.parametrize("sep_deg", [30.0, 60.0, 85.0])
def test_wide_angle_front_hemisphere_matches_astropy(proj, impl, sep_deg):
    header = ar.synthetic_header(proj, (150.0, 30.0))
    lon, lat = _ring(150.0, 30.0, sep_deg)
    ref = np.column_stack(ar.astropy_wcs(header).wcs_world2pix(np.degrees(lon), np.degrees(lat), 0))
    x, y = ar.build(ar.Case("wide", header), impl).proj(lon, lat)
    got = np.column_stack([np.asarray(x), np.asarray(y)])
    # Pixel offsets reach ~3e5 at 85 deg for TAN, so compare relatively.
    rel = np.hypot(*(got - ref).T) / np.maximum(1.0, np.hypot(*ref.T))
    assert rel.max() < 1e-12


# --- Per-centre vmap oracle coverage ----------------------------------------

ARRAY_CENTRES = np.array([[150.0, 30.0], [359.99, 5.0], [45.0, 89.9], [200.0, -60.0]])


@pytest.mark.parametrize("cls, proj", [(Tan, "TAN"), (Sin, "SIN")])
def test_vmap_matches_independent_astropy_per_centre(cls, proj):
    crpix, cd, _, _ = ar._params(ar.synthetic_header(proj, (0.0, 0.0)))
    wcses = [
        cls(crpix, cd, centre, precision="f64") for centre in ARRAY_CENTRES
    ]
    batched_wcs = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves), *wcses
    )
    pix = ar.pixel_grid(ar.synthetic_header(proj, (0.0, 0.0)), 9)
    px, py = jnp.asarray(pix[:, 0]), jnp.asarray(pix[:, 1])

    # The stacked WCS axis is 0; out_axes=1 preserves the old (pixels, centres)
    # layout while each vmap body sees the same shared pixel grid.
    ra, dec = jax.vmap(
        lambda w: w.unproj(px, py), in_axes=0, out_axes=1
    )(batched_wcs)
    ra, dec = np.degrees(np.asarray(ra)), np.degrees(np.asarray(dec))
    references = np.stack([
        ar.oracle_world(ar.synthetic_header(proj, tuple(centre)), pix)
        for centre in ARRAY_CENTRES
    ])
    for k, ref in enumerate(references):
        sep = ar.separation_mas(ra[:, k], dec[:, k], ref[:, 0], ref[:, 1])
        assert sep.max() < FORWARD_MAS, f"centre {k}: {sep.max():.3e} mas"

    lon = jnp.radians(jnp.asarray(references[:, :, 0]))
    lat = jnp.radians(jnp.asarray(references[:, :, 1]))
    x, y = jax.vmap(
        lambda w, sky_lon, sky_lat: w.proj(sky_lon, sky_lat),
        in_axes=(0, 0, 0), out_axes=1,
    )(batched_wcs, lon, lat)
    got = np.stack([np.asarray(x).T, np.asarray(y).T], axis=-1)
    for k in range(len(ARRAY_CENTRES)):
        assert np.hypot(*(got[k] - pix).T).max() < INVERSE_PX, f"centre {k}"
