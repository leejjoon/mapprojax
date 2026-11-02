"""``load_wcs_params_from_header`` against astropy's interpretation.

Supported branches must reproduce astropy's pixel->sky mapping. Header keys
the loader does not understand (axis order, units, non-TAN/SIN projections,
LONPOLE, PV) must either be honoured or rejected with ``ValueError``; silent
misinterpretation is rejected, not guessed.
"""

from pathlib import Path
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from mapprojax import Sin, Tan
from mapprojax.wcs_utils import load_wcs_params_from_header

import accuracy_reference as ar

DATA_DIR = Path(__file__).resolve().parent / "data"

PIX = np.array([[0.0, 0.0], [10.0, 90.0], [49.5, 49.5], [99.0, 3.0], [70.0, 70.0]])


def _header(**cards):
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 100, 100
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRPIX1"], h["CRPIX2"] = 50.5, 50.5
    h["CRVAL1"], h["CRVAL2"] = 150.0, 30.0
    for k, v in cards.items():
        h[k] = v
    return h


def _unit(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def _max_sep_arcsec(lon1, lat1, lon2, lat2):
    d = np.linalg.norm(_unit(lon1, lat1) - _unit(lon2, lat2), axis=-1)
    return np.degrees(2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))).max() * 3600.0


def _assert_matches_astropy(h, pix=PIX, atol_arcsec=1e-6):
    crpix, cd, crval, sip = load_wcs_params_from_header(h)
    cls = Sin if "SIN" in h["CTYPE1"] else Tan
    ra, dec = cls(crpix, cd, crval, sip=sip).unproj(pix[:, 0], pix[:, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = WCS(h).all_pix2world(pix, 0)
    # astropy keeps the axis order of the header; compare as (lon, lat).
    lon_ax = 0 if h["CTYPE1"][:4] in ("RA--", "GLON", "ELON") else 1
    sep = _max_sep_arcsec(np.degrees(ra), np.degrees(dec), ref[:, lon_ax], ref[:, 1 - lon_ax])
    assert sep < atol_arcsec, f"max separation {sep:.3e} arcsec"


def _assert_supported_or_rejected(h):
    try:
        load_wcs_params_from_header(h)
    except ValueError:
        return
    _assert_matches_astropy(h)


# --- supported linear-transform branches ------------------------------------

LINEAR = {
    "cd_full": dict(CD1_1=-2.0e-3, CD1_2=3.0e-4, CD2_1=2.0e-4, CD2_2=2.1e-3),
    "cd_missing_offdiag": dict(CD1_1=-2.0e-3, CD2_2=2.1e-3),
    "pc_without_cdelt": dict(PC1_1=-2.0e-3, PC1_2=3.0e-4, PC2_1=2.0e-4, PC2_2=2.1e-3),
    "cdelt_default_pc": dict(CDELT1=-2.0e-3, CDELT2=2.1e-3),
    "cdelt_unequal_offdiag_pc": dict(
        CDELT1=-1.5e-3, CDELT2=2.5e-3, PC1_1=0.96, PC1_2=0.28, PC2_1=-0.28, PC2_2=0.96
    ),
}


@pytest.mark.parametrize("name", list(LINEAR))
def test_linear_branch_matches_astropy(name):
    _assert_matches_astropy(_header(**LINEAR[name]))


def test_cd_branch_parameters_are_verbatim():
    h = _header(**LINEAR["cd_full"])
    crpix, cd, crval, sip = load_wcs_params_from_header(h)
    assert crpix == [49.5, 49.5]
    assert crval == [150.0, 30.0]
    np.testing.assert_array_equal(cd, [[-2.0e-3, 3.0e-4], [2.0e-4, 2.1e-3]])
    assert sip is None


# --- keys the loader must honour or reject ----------------------------------

BASE = LINEAR["cdelt_default_pc"]


def test_galactic_axes_are_supported():
    _assert_matches_astropy(_header(CTYPE1="GLON-TAN", CTYPE2="GLAT-TAN", **BASE))


def test_explicit_default_lonpole_is_supported():
    _assert_matches_astropy(_header(LONPOLE=180.0, **BASE))


def test_latpole_does_not_affect_zenithal():
    # LATPOLE only disambiguates non-zenithal poles; it must be harmless here.
    _assert_matches_astropy(_header(LATPOLE=-30.0, **BASE))


def test_zero_pv_on_sin_is_supported():
    _assert_matches_astropy(
        _header(CTYPE1="RA---SIN", CTYPE2="DEC--SIN", PV2_1=0.0, PV2_2=0.0, **BASE)
    )


UNSUPPORTED = {
    "swapped_axes": dict(CTYPE1="DEC--TAN", CTYPE2="RA---TAN", CRVAL1=30.0, CRVAL2=150.0),
    "arcsec_units": dict(CUNIT1="arcsec", CUNIT2="arcsec", CDELT1=-7.2, CDELT2=7.56),
    "car_projection": dict(CTYPE1="RA---CAR", CTYPE2="DEC--CAR"),
    "nondefault_lonpole": dict(LONPOLE=170.0),
    "nonzero_sin_pv": dict(CTYPE1="RA---SIN", CTYPE2="DEC--SIN", PV2_1=0.1, PV2_2=-0.05),
}


@pytest.mark.parametrize("name", list(UNSUPPORTED))
def test_unsupported_key_is_honoured_or_rejected(name):
    cards = {**BASE, **UNSUPPORTED[name]}
    _assert_supported_or_rejected(_header(**cards))


# The two orientations that a 180-degree rotation reproduces, and the one that
# it does not unless the header says so explicitly (WCSLIB defaults to 0 at
# CRVAL2 = +90, and the error grows with field size: 141" on 101 px).


def test_south_pole_without_lonpole_is_supported():
    _assert_matches_astropy(_header(CRVAL2=-90.0, **BASE))


def test_north_pole_with_explicit_180_lonpole_is_supported():
    _assert_matches_astropy(_header(CRVAL2=90.0, LONPOLE=180.0, **BASE))


def test_north_pole_without_lonpole_is_rejected():
    with pytest.raises(ValueError, match="CRVAL2 = \\+90"):
        load_wcs_params_from_header(_header(CRVAL2=90.0, **BASE))


def test_crota2_without_cd_or_pc_is_rejected():
    spaced = {k: v for k, v in BASE.items() if not k.startswith("CDELT")}
    with pytest.raises(ValueError, match="CROTA2"):
        load_wcs_params_from_header(_header(CROTA2=30.0, **spaced))


def test_off_diagonal_only_cd_is_not_the_identity():
    crpix, cd, crval, sip = load_wcs_params_from_header(
        _header(CD1_2=3.0e-4, CD2_1=2.0e-4)
    )
    np.testing.assert_array_equal(cd, [[0.0, 3.0e-4], [2.0e-4, 0.0]])
    _assert_matches_astropy(_header(CD1_2=3.0e-4, CD2_1=2.0e-4))


def test_alternate_wcs_pv_cards_are_ignored():
    # The SPHEREx L2 header carries PV1_3W/PV2_3W for its WAVE-TAB axis;
    # they must not be read as primary celestial PV cards.
    h = ar.load_header("spherex_l2.hdr")
    assert "PV1_3W" in h
    load_wcs_params_from_header(h)


def test_nonstandard_ctype_shapes_are_rejected():
    for cards in (
        dict(CTYPE1="RA---TAN-SIPX", CTYPE2="DEC--TAN-SIP"),
        dict(CTYPE1="RA", CTYPE2="DEC"),
        dict(CTYPE1="RA---TAN", CTYPE2="GLAT-TAN"),
        dict(CTYPE1="RA---TAN", CTYPE2="DEC--SIN"),
    ):
        with pytest.raises(ValueError):
            load_wcs_params_from_header(_header(**{**BASE, **cards}))


# --- real headers -----------------------------------------------------------

def test_fixture_files_match_recorded_provenance():
    import hashlib
    import json

    provenance = json.loads((DATA_DIR / "provenance.json").read_text())
    assert provenance, "provenance.json is empty"
    for name, record in provenance.items():
        digest = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        assert digest == record["sha256"], f"{name} changed without updating provenance.json"


@pytest.mark.parametrize(
    "name, sep",
    [
        ("1904-66_TAN.hdr", ""),
        ("1904-66_SIN.hdr", ""),
        ("spherex_l2.hdr", "\n"),
        ("spherex_template.hdr", "\n"),
    ],
)
def test_fixture_header_loads_like_astropy(name, sep):
    # The recorded representation must be the one the file actually uses.
    assert ar.header_sep(name) == sep
    h = ar.load_header(name)
    n1, n2 = h["NAXIS1"], h["NAXIS2"]
    pix = np.array([[0.0, 0.0], [n1 - 1.0, 0.0], [0.0, n2 - 1.0], [n1 - 1.0, n2 - 1.0],
                    [h["CRPIX1"] - 1.0, h["CRPIX2"] - 1.0], [0.3 * n1, 0.7 * n2]])
    _assert_matches_astropy(h, pix)
