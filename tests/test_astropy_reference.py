"""Dense-grid comparison of mapprojax against astropy (fp64).

Cases, sampling and metrics live in ``tests/accuracy_reference.py`` and are
shared with ``benchmarks/benchmark_accuracy.py``. Budgets:

- forward (pixel -> sky): 1e-4 mas great-circle separation;
- inverse (sky -> pixel) without SIP: 1e-6 px;
Known Stage 2 defects (SIP inverse; near-pole forward through ``unproj``)
are strict ``xfail`` against those budgets, and each is also held to its
measured regression cap from ``tests/data/accuracy_baselines.json`` by a
separate ordinary test, so getting worse fails even while the target is
xfailed.
"""

import warnings

import jax
import numpy as np
import pytest

import mapprojax as mp

import accuracy_reference as ar

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


KNOWN = [
    pytest.param(case, impl, metric, id=f"{case.name}-{impl}-{metric}")
    for case in CASES
    for impl in ar.implementations_for(case)
    for metric in ar.METRIC_KEYS
    if ar.budget(impl, metric) is not None
    and ar.known_failure_cap_or_none(case, impl, metric) is not None
]


@pytest.mark.parametrize("case, impl, metric", KNOWN)
def test_known_failure_within_regression_cap(case, impl, metric):
    key, unit = ar.METRIC_KEYS[metric]
    cap = ar.known_failure_cap(case, impl, metric)
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


# --- Array variants ----------------------------------------------------------

ARRAY_CENTRES = np.array([[150.0, 30.0], [359.99, 5.0], [45.0, 89.9], [200.0, -60.0]])


@pytest.mark.parametrize(
    "cls_name, proj",
    [("TanArray", "TAN"), ("TanJaxArray", "TAN"), ("SinArray", "SIN"), ("SinJaxArray", "SIN")],
)
def test_array_variant_matches_independent_astropy_per_centre(cls_name, proj):
    crpix, cd, _, _ = ar._params(ar.synthetic_header(proj, (0.0, 0.0)))
    wcs = getattr(mp, cls_name)(crpix, cd, (ARRAY_CENTRES[:, 0], ARRAY_CENTRES[:, 1]))
    pix = ar.pixel_grid(ar.synthetic_header(proj, (0.0, 0.0)), 9)

    # Pixels as (P, 1) broadcast against the N centres -> (P, N).
    ra, dec = wcs.unproj(pix[:, 0:1], pix[:, 1:2])
    ra, dec = np.degrees(np.asarray(ra)), np.degrees(np.asarray(dec))
    for k, centre in enumerate(ARRAY_CENTRES):
        ref = ar.oracle_world(ar.synthetic_header(proj, tuple(centre)), pix)
        sep = ar.separation_mas(ra[:, k], dec[:, k], ref[:, 0], ref[:, 1])
        assert sep.max() < FORWARD_MAS, f"centre {k}: {sep.max():.3e} mas"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x, y = wcs.proj(np.radians(ref[:, 0:1]), np.radians(ref[:, 1:2]))
        got = np.column_stack([np.asarray(x)[:, k], np.asarray(y)[:, k]])
        assert np.hypot(*(got - pix).T).max() < INVERSE_PX, f"centre {k}"
