"""The real SPHEREx pair: ``Reprojector`` against astropy's full chain.

Directions follow ``Reprojector``'s API (first WCS supplies the pixels):

- forward: Galactic template pixel -> Galactic -> ICRS -> L2 exposure pixel,
  rotation Galactic -> ICRS. Ends in the L2 SIP inverse (a Newton solve
  against A/B, seeded by AP/BP), so it is exact in fp64 as well.
- reverse: L2 pixel -> ICRS -> Galactic -> template pixel, rotation
  ICRS -> Galactic. Uses only the SIP forward and a SIP-free TAN inverse, so
  it must be exact; this pins the rotation independently of SIP.

Overlap is decided from astropy's destination pixel, never mapprojax output.
"""

import jax
import numpy as np
import pytest

import accuracy_reference as ar

TARGET_PX = ar.TARGET_PX  # Stage 2 acceptance for the converged SIP inverse (fp64)


@pytest.fixture(autouse=True)
def _x64_enabled():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_rotation_matrix_is_proper_and_matches_astropy():
    m = ar.galactic_to_icrs_matrix()
    np.testing.assert_allclose(m @ m.T, np.eye(3), rtol=0, atol=1e-14)
    assert abs(np.linalg.det(m) - 1.0) < 1e-14

    rng = np.random.default_rng(1)
    glon, glat = rng.uniform(0, 360, 16), rng.uniform(-80, 80, 16)
    ra, dec = ar.galactic_to_icrs_deg(glon, glat)
    lo, la = np.radians(glon), np.radians(glat)
    v = m @ np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    got_ra, got_dec = np.degrees(np.arctan2(v[1], v[0])), np.degrees(np.arcsin(v[2]))
    assert ar.separation_mas(got_ra, got_dec, ra, dec).max() < 1e-4


def _ids(runs):
    return [f"{d}-{p}" for d, p in runs]


@pytest.mark.parametrize("direction, precision", ar.PAIR_RUNS, ids=_ids(ar.PAIR_RUNS))
def test_overlap_is_nonempty_and_finite(direction, precision):
    # Stats drop non-finite samples from max/rms, so count them explicitly.
    result = ar.evaluate_pair(precision, direction)
    assert result["n_overlap"] > 0
    assert result["pixel_error_px"]["n"] == result["n_overlap"]
    assert result["pixel_error_px"]["n_nonfinite"] == 0


@pytest.mark.parametrize("direction, precision", ar.PAIR_EXACT, ids=_ids(ar.PAIR_EXACT))
def test_sip_free_direction_is_exact(direction, precision):
    stats = ar.evaluate_pair(precision, direction)["pixel_error_px"]
    assert stats["max"] < TARGET_PX, f"{stats['max']:.3e} px"


CAPPED = [run for run in ar.PAIR_RUNS if run not in ar.PAIR_EXACT]


@pytest.mark.parametrize("direction, precision", CAPPED, ids=_ids(CAPPED))
def test_within_regression_cap(direction, precision):
    cap = ar.pair_cap(direction, precision)
    stats = ar.evaluate_pair(precision, direction)["pixel_error_px"]
    assert stats["max"] <= cap, f"{stats['max']:.3e} px > cap {cap} px"


def test_forward_f64_reaches_stage2_target():
    stats = ar.evaluate_pair("f64", "forward")["pixel_error_px"]
    assert stats["max"] < TARGET_PX, f"{stats['max']:.3e} px"


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_transposed_rotation_is_caught(direction):
    # Per sample: rejected (non-finite) or grossly wrong. Must not depend on
    # the back-hemisphere bug, which today returns finite mirrored pixels.
    c = ar.transposed_rotation_control(direction)
    assert c["n"] > 0 and c["detected"], c
