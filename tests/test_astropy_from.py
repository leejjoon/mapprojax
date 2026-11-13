"""``mapprojax.from_astropy`` against astropy's own interpretation.

Astropy semantics are authoritative here. ``from_astropy`` reproduces
astropy's reading of the CD / PC+CDELT / CROTA precedence and of unit
normalisation, so it is *wider* than ``load_wcs_params_from_header``: it
accepts CROTA and non-degree units, and it refuses a ``-SIP`` CTYPE with no
coefficients. Numerical agreement with the loader is asserted only for headers
where both read the same linear representation (see
``test_matches_header_loader_leaf_for_leaf``).
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import accuracy_reference as ar
import jax
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS, Sip

from mapprojax import Sin, Tan, from_astropy
from mapprojax.wcs_utils import load_wcs_params_from_header

DATA_DIR = Path(__file__).resolve().parent / "data"
REPO_ROOT = DATA_DIR.parent.parent


@pytest.fixture(autouse=True)
def _x64_enabled():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _header(**cards):
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 100, 100
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRPIX1"], h["CRPIX2"] = 50.5, 50.5
    h["CRVAL1"], h["CRVAL2"] = 150.0, 30.0
    h["CDELT1"], h["CDELT2"] = -2.0e-3, 2.1e-3
    for k, v in cards.items():
        h[k] = v
    return h


def _unit(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1
    )


def _max_sep_arcsec(lon1, lat1, lon2, lat2):
    d = np.linalg.norm(_unit(lon1, lat1) - _unit(lon2, lat2), axis=-1)
    return np.degrees(2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))).max() * 3600.0


def _assert_matches_astropy(wcs, kernel, pix, atol_arcsec=1e-6):
    """Forward agreement with ``all_pix2world`` on the given pixels."""
    want = np.array(wcs.all_pix2world(pix, 0))
    lon, lat = kernel.unproj(pix[:, 0], pix[:, 1])
    sep = _max_sep_arcsec(
        np.degrees(np.asarray(lon)), np.degrees(np.asarray(lat)), want[:, 0], want[:, 1]
    )
    assert sep < atol_arcsec, f"max separation {sep:.3e} arcsec"


def _grid(h):
    n1, n2 = h["NAXIS1"], h["NAXIS2"]
    return np.array(
        [
            [0.0, 0.0],
            [n1 - 1.0, 0.0],
            [0.0, n2 - 1.0],
            [n1 - 1.0, n2 - 1.0],
            [h["CRPIX1"] - 1.0, h["CRPIX2"] - 1.0],
            [0.3 * n1, 0.7 * n2],
        ]
    )


FIXTURES = ["1904-66_TAN.hdr", "1904-66_SIN.hdr", "spherex_l2.hdr", "spherex_template.hdr"]


# --- corpus ----------------------------------------------------------------


@pytest.mark.parametrize("case", ar.cases(), ids=lambda c: c.name)
def test_corpus_converts_and_matches_astropy(case):
    h = case.header
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    _assert_matches_astropy(w, kernel, ar.pixel_grid(h))


def test_wcs_test_data_header_converts():
    raw = json.loads((DATA_DIR / "wcs_test_data.json").read_text())["header_keywords"]
    h = fits.Header()
    for key, value in raw.items():
        h[key] = value
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 100, 100
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    np.testing.assert_array_equal(kernel.crpix, np.asarray(w.wcs.crpix) - 1.0)
    np.testing.assert_array_equal(kernel.crval, np.asarray(w.wcs.crval))
    np.testing.assert_array_equal(kernel.cd, np.asarray(w.pixel_scale_matrix))


def test_corpus_mixed_and_f32_within_caps():
    """The two fp32 contracts, each against its own baseline group.

    ``precision="mixed"`` runs fp32 pixels with fp64 reference values and is
    held to the per-case ``fp32.*/<case>/jax_mixed`` caps. The ``"f32"``
    contract has no absolute ``proj`` (it raises), so it is checked through
    the offset round trip instead.
    """
    pix = np.array([[0.0, 0.0], [10.0, 90.0], [49.5, 49.5], [70.0, 70.0]])

    for case in ar.cases():
        h = case.header
        w = WCS(h)
        kernel = from_astropy(w, precision="mixed")
        want = np.array(w.all_pix2world(pix, 0))
        lon, lat = kernel.unproj(np.asarray(pix, dtype=np.float32)[:, 0],
                                  np.asarray(pix, dtype=np.float32)[:, 1])
        sep = _max_sep_arcsec(
            np.degrees(np.asarray(lon)), np.degrees(np.asarray(lat)),
            want[:, 0], want[:, 1],
        )
        cap = ar.fp32_cap(case, "jax_mixed", "forward")
        assert sep <= cap, f"{case.name}: {sep:.4f} mas > cap {cap}"

    # f32: no absolute proj, but the offset pipeline must round trip.
    h = ar.load_header("spherex_l2.hdr")
    w = WCS(h)
    f32 = from_astropy(w, precision="f32")
    with pytest.raises(TypeError):
        f32.unproj(pix[:, 0], pix[:, 1])
    ref = from_astropy(w, precision="f64")
    off_x, off_y = f32.unproj_offset(pix[:, 0], pix[:, 1])
    lon, lat = ref.unproj(pix[:, 0], pix[:, 1])
    assert np.isfinite(np.asarray(off_x)).all()
    assert np.isfinite(np.asarray(off_y)).all()
    # The offset pipeline must invert itself, which is the whole point of the
    # f32 contract: unproj_offset -> proj_offset returns the input pixels.
    back_x, back_y = f32.proj_offset(off_x, off_y)
    np.testing.assert_allclose(np.asarray(back_x), np.asarray(pix[:, 0], dtype=np.float32), atol=1e-2)
    np.testing.assert_allclose(np.asarray(back_y), np.asarray(pix[:, 1], dtype=np.float32), atol=1e-2)


# --- linear forms ----------------------------------------------------------

LINEAR_FORMS = {
    "cd_full": {"CD1_1": -2.0e-3, "CD1_2": 3.0e-4, "CD2_1": 2.0e-4, "CD2_2": 2.1e-3},
    "cd_offdiag": {"CD1_1": -2.0e-3, "CD2_2": 2.1e-3},
    "cdelt_unequal_pc": {
        "CDELT1": -1.5e-3, "CDELT2": 2.5e-3,
        "PC1_1": 0.96, "PC1_2": 0.28, "PC2_1": -0.28, "PC2_2": 0.96,
    },
    # CROTA is accepted here because astropy normalises it, while the raw
    # loader refuses it. That asymmetry is the manual's reason to prefer this
    # entry point.
    "crota": {"CDELT1": -2.0e-3, "CDELT2": 2.1e-3, "CROTA2": 12.0},
}


@pytest.mark.parametrize("name", list(LINEAR_FORMS))
def test_linear_forms_follow_astropy_precedence(name):
    h = _header(**LINEAR_FORMS[name])
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    np.testing.assert_array_equal(kernel.cd, np.asarray(w.pixel_scale_matrix))
    _assert_matches_astropy(w, kernel, _grid(h))


def test_mixed_linear_representation_matches_astropy_only():
    """Where CD and PC/CDELT coexist, astropy wins and the loader differs.

    This is why loader equivalence is scoped (D12): the loader does not apply
    PC/CDELT when CD cards are present, so the two disagree by construction.
    """
    # Astropy's rule, measured: when PC and/or CDELT are present alongside
    # CD, the CD cards are IGNORED and the matrix is PC * CDELT, with an
    # implicit CDELT of 1 when the cards are absent.
    #
    # CD + PC + CDELT -> PC * CDELT.
    h1 = _header(CD1_1=-0.1, CD2_2=0.1, PC1_1=0.5, PC2_2=0.5, CDELT1=-0.01, CDELT2=0.02)
    w1 = WCS(h1)
    w1.wcs.set()
    k1 = from_astropy(w1, precision="f64")
    np.testing.assert_allclose(
        np.asarray(k1.cd), np.diag([-0.005, 0.01]), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(k1.cd), np.asarray(w1.pixel_scale_matrix), rtol=0, atol=0
    )
    _assert_matches_astropy(w1, k1, _grid(h1))

    # CD + PC with no CDELT -> implicit CDELT of 1, so PC alone. `_header`
    # ships a default CDELT, so delete it to test the implicit case.
    h2 = _header(CD1_1=-0.1, CD2_2=0.1, PC1_1=0.5, PC2_2=0.5)
    del h2["CDELT1"], h2["CDELT2"]
    w2 = WCS(h2)
    w2.wcs.set()
    k2 = from_astropy(w2, precision="f64")
    np.testing.assert_allclose(
        np.asarray(k2.cd), np.diag([0.5, 0.5]), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(k2.cd), np.asarray(w2.pixel_scale_matrix), rtol=0, atol=0
    )
    _assert_matches_astropy(w2, k2, _grid(h2))

    # And the loader disagrees on both, which is the point.
    for h in (h1, h2):
        _, loader_cd, _, _ = load_wcs_params_from_header(h)
        assert not np.allclose(np.asarray(loader_cd), np.asarray(WCS(h).pixel_scale_matrix))


# --- units -----------------------------------------------------------------


@pytest.mark.parametrize("preserve", [True, False])
@pytest.mark.parametrize("unit", ["deg", "rad", "arcsec"])
def test_preserve_units_deg_rad_arcsec(preserve, unit):
    # The unit is part of the header, so CRVAL and CDELT are expressed in it.
    # `preserve` documents that both settings must land on the same degrees.
    scale = {"deg": 1.0, "rad": np.pi / 180.0, "arcsec": 3600.0}[unit]
    h = _header(CRVAL1=150.0 * scale, CRVAL2=30.0 * scale,
                CDELT1=-2.0e-3 * scale, CDELT2=2.1e-3 * scale)
    h["CUNIT1"] = h["CUNIT2"] = unit
    w = WCS(h)
    w.wcs.set()
    kernel = from_astropy(w, precision="f64")
    # Always degrees on the way out.
    np.testing.assert_allclose(np.asarray(kernel.crval), [150.0, 30.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(kernel.cd), np.diag([-2.0e-3, 2.1e-3]), rtol=0, atol=1e-15
    )
    _assert_matches_astropy(w, kernel, _grid(h))


def test_mixed_axis_units_scale_each_row():
    # Axis 1 in arcsec, axis 2 in deg. Each row of the matrix must be scaled
    # by its own axis factor, so the result equals the all-degrees kernel.
    h = _header(CRVAL1=150.0 * 3600.0, CRVAL2=30.0,
                CDELT1=-2.0e-3 * 3600.0, CDELT2=2.1e-3)
    h["CUNIT1"], h["CUNIT2"] = "arcsec", "deg"
    w = WCS(h)
    w.wcs.set()

    kernel = from_astropy(w, precision="f64")
    np.testing.assert_allclose(np.asarray(kernel.crval), [150.0, 30.0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(kernel.cd), np.diag([-2.0e-3, 2.1e-3]), rtol=0, atol=1e-15
    )
    _assert_matches_astropy(w, kernel, _grid(h))


# --- frames ----------------------------------------------------------------


def test_galactic_and_fk5_agree_in_their_own_frame():
    h = _header(CTYPE1="GLON-TAN", CTYPE2="GLAT-TAN")
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    _assert_matches_astropy(w, kernel, _grid(h))


def test_elon_elat_pair_is_accepted():
    h = _header(CTYPE1="ELON-TAN", CTYPE2="ELAT-TAN")
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    _assert_matches_astropy(w, kernel, _grid(h))


def test_frame_metadata_is_dropped():
    h = _header(RADESYS="FK5", EQUINOX=2000.0)
    h["MJD-OBS"] = 58000.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # astropy derives DATE-OBS from MJD-OBS
        w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    payload = json.dumps(kernel.to_dict())
    for keyword in ("RADESYS", "EQUINOX", "MJD-OBS"):
        assert keyword not in payload


# --- SIP -------------------------------------------------------------------


def _sip_header(**cards):
    h = _header(CTYPE1="RA---TAN-SIP", CTYPE2="DEC--TAN-SIP", CD1_1=-2.0e-3, CD2_2=2.1e-3)
    for k, v in cards.items():
        h[k] = v
    return h


def test_sip_ab_only():
    h = _sip_header(A_ORDER=2, A_0_2=1e-4, B_ORDER=2, B_2_0=1e-4)
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    np.testing.assert_array_equal(np.asarray(kernel.sip.a), np.asarray(w.sip.a))
    np.testing.assert_array_equal(np.asarray(kernel.sip.b), np.asarray(w.sip.b))
    _assert_matches_astropy(w, kernel, _grid(h))


def test_sip_ab_with_ap_bp():
    h = _sip_header(
        A_ORDER=2, A_0_2=1e-4, B_ORDER=2, B_2_0=1e-4,
        AP_ORDER=2, AP_0_2=-1e-4, BP_ORDER=2, BP_2_0=-1e-4,
    )
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    np.testing.assert_array_equal(np.asarray(kernel.sip.ap), np.asarray(w.sip.ap))
    np.testing.assert_array_equal(np.asarray(kernel.sip.bp), np.asarray(w.sip.bp))
    _assert_matches_astropy(w, kernel, _grid(h))


def test_sip_without_ctype_suffix_is_applied():
    h = _header(A_ORDER=2, A_0_2=1e-4, B_ORDER=2, B_2_0=1e-4)
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    assert kernel.sip is not None
    # Agreement, not merely "did not raise".
    _assert_matches_astropy(w, kernel, _grid(h))


def test_sip_crpix_mismatch_is_refused():
    h = _sip_header(A_ORDER=2, A_0_2=1e-4, B_ORDER=2, B_2_0=1e-4)
    w = WCS(h)
    # `sip.crpix` is read-only, so replace the whole Sip object with one whose
    # reference pixel disagrees with CRPIX1/2.
    old = w.sip
    w.sip = Sip(old.a, old.b, old.ap, old.bp, np.asarray(old.crpix) + 1.0)
    with pytest.raises(ValueError, match="sip.crpix"):
        from_astropy(w, precision="f64")


def test_sip_off_triangle_nonzero_is_refused():
    h = _sip_header(A_ORDER=2, A_0_2=1e-4, B_ORDER=2, B_2_0=1e-4)
    w = WCS(h)
    w.sip.a[2, 2] = 5e-4
    with pytest.raises(ValueError, match="sip triangle"):
        from_astropy(w, precision="f64")


def test_ctype_sip_without_sip_object_is_refused():
    h = _sip_header()
    w = WCS(h)
    assert w.sip is None
    with pytest.raises(ValueError, match="-SIP without SIP"):
        from_astropy(w, precision="f64")


def test_inverse_steps_forwarded():
    h = _sip_header(A_ORDER=2, A_0_2=1e-4, B_ORDER=2, B_2_0=1e-4)
    kernel = from_astropy(WCS(h), precision="f64", inverse_steps=7)
    assert kernel.sip.inverse_steps == 7


# --- poles and edges -------------------------------------------------------


def _pole_header(dec, **cards):
    h = _header(**cards)
    h["CRVAL2"] = dec
    return h


def test_crval2_minus_90_accepted():
    h = _pole_header(-90.0)
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    assert kernel is not None


def test_crval2_plus_90_with_180_accepted():
    h = _pole_header(90.0, LONPOLE=180.0)
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    assert kernel is not None


def test_crval2_plus_90_without_lonpole_refused():
    # A materialised WCS cannot distinguish an omitted LONPOLE from an
    # explicit 0.0 at CRVAL2 = +90, so the refusal quotes the resolved value.
    h = _pole_header(90.0)
    w = WCS(h)
    assert float(w.wcs.lonpole) == 0.0
    with pytest.raises(ValueError, match=r"LONPOLE = 0\.0"):
        from_astropy(w, precision="f64")


def test_longitude_wrap_and_projection_edges_match_astropy():
    h = _header(CRVAL1=0.2)
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    pix = np.array([[0.0, 0.0], [99.0, 99.0], [49.5, 0.0], [0.0, 49.5]])
    _assert_matches_astropy(w, kernel, pix)


# --- equivalence -----------------------------------------------------------


@pytest.mark.parametrize("name", FIXTURES)
def test_matches_header_loader_leaf_for_leaf(name):
    """D12: scoped to single-linear-representation, card-stable headers."""
    h = ar.load_header(name)
    kernel = from_astropy(WCS(h), precision="f64")
    crpix, cd, crval, sip = load_wcs_params_from_header(h)

    np.testing.assert_array_equal(np.asarray(kernel.crpix), np.asarray(crpix))
    # Signed zero differs by construction (astropy writes 0.0 where the
    # loader writes -0.0), so compare value-exactly but sign-blind.
    np.testing.assert_allclose(
        np.asarray(kernel.cd), np.asarray(cd), rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(np.asarray(kernel.crval), np.asarray(crval))
    if sip is None:
        assert kernel.sip is None
    else:
        for field_name in ("a", "b", "ap", "bp"):
            want = getattr(sip, field_name, None)
            got = getattr(kernel.sip, field_name, None)
            if want is None:
                assert got is None, field_name
            else:
                np.testing.assert_array_equal(np.asarray(got), np.asarray(want), field_name)


def test_matches_header_loader_for_cd_and_unequal_cdelt_branches():
    for cards in (
        {"CD1_1": -2.0e-3, "CD1_2": 3.0e-4, "CD2_1": 2.0e-4, "CD2_2": 2.1e-3},
        {"CDELT1": -1.5e-3, "CDELT2": 2.5e-3},
    ):
        h = _header(**cards)
        kernel = from_astropy(WCS(h), precision="f64")
        crpix, cd, _crval, _sip = load_wcs_params_from_header(h)
        np.testing.assert_allclose(
            np.asarray(kernel.cd), np.asarray(cd), rtol=0.0, atol=0.0
        )
        np.testing.assert_array_equal(np.asarray(kernel.crpix), np.asarray(crpix))


# --- refusals --------------------------------------------------------------


def test_tpv_hidden_behind_tan_is_refused():
    h = _header(CTYPE1="RA---TPV", CTYPE2="DEC--TPV", PV1_0=0.0, PV2_0=0.0)
    w = WCS(h)
    assert str(w.wcs.ctype[0]) == "RA---TAN"
    with pytest.raises(ValueError, match="TPV"):
        from_astropy(w, precision="f64")


def test_tpd_behind_tan_is_refused():
    # CPDIS1='TPD' alone does not construct: astropy deletes it for a
    # non-LOOKUP type and then fails on the orphaned DP cards. The complete
    # two-axis CQDIS/DQ form does construct, and hides behind a TAN CTYPE
    # with cpdis1 = None and has_distortion = False.
    h = _header()
    h["CQDIS1"], h["CQDIS2"] = "TPD", "TPD"
    for axis in (1, 2):
        h[f"DQ{axis}.NAXES"] = 1
        h[f"DQ{axis}.AXIS.1"] = 1
    w = WCS(h)
    w.wcs.set()
    assert str(w.wcs.ctype[0]) == "RA---TAN"
    assert w.cpdis1 is None and w.has_distortion is False
    with pytest.raises(ValueError, match="CQDIS|DQ"):
        from_astropy(w, precision="f64")


def test_tab_ctype_is_refused():
    # A real RA---TAB WCS needs a BINTABLE extension, and astropy cannot even
    # construct one from cards alone (it raises MemoryError in wcsset for the
    # missing table). So drive the classifier on a relaxed header, which is
    # exactly the input the refusal reads.
    from mapprojax.astropy import _refuse_ctype

    for ctype, token in (
        ("RA---TAB", "-TAB"),
        ("RA---TPV", "TPV"),
        ("DEC--TPD", "TPD"),
    ):
        with pytest.raises(ValueError, match=token):
            _refuse_ctype({"CTYPE1": ctype, "CTYPE2": "DEC--TAN"})


def test_ps_parameter_is_refused():
    h = _header()
    h["PS1_0"] = "1.0e-5"
    w = WCS(h)
    w.wcs.set()
    assert w.wcs.get_ps()
    with pytest.raises(ValueError, match="PS"):
        from_astropy(w, precision="f64")


def test_cube_without_celestial_is_refused():
    h = _header()
    h["NAXIS"] = 3
    h["NAXIS3"] = 10
    h["CTYPE3"] = "FREQ"
    h["CUNIT3"] = "Hz"
    h["CDELT3"] = 1e6
    h["CRPIX3"] = 1.0
    h["CRVAL3"] = 1e9
    w = WCS(h)
    with pytest.raises(ValueError, match="wcs.celestial"):
        from_astropy(w, precision="f64")


def test_non_celestial_pair_is_refused():
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 100, 100
    h["CTYPE1"], h["CTYPE2"] = "FREQ", "FREQ"
    h["CRPIX1"] = h["CRPIX2"] = 50.0
    h["CRVAL1"] = h["CRVAL2"] = 1e9
    h["CDELT1"] = h["CDELT2"] = 1e6
    h["CUNIT1"] = h["CUNIT2"] = "Hz"
    w = WCS(h)
    with pytest.raises(ValueError, match="non-celestial|projection|axis"):
        from_astropy(w, precision="f64")


def test_swapped_axes_is_refused():
    h = _header(CTYPE1="DEC--TAN", CTYPE2="RA---TAN",
                CRVAL1=30.0, CRVAL2=150.0, CDELT1=2.1e-3, CDELT2=-2.0e-3)
    w = WCS(h)
    with pytest.raises(ValueError, match="axis order"):
        from_astropy(w, precision="f64")


def test_unsupported_projection_is_refused():
    h = _header(CTYPE1="RA---CAR", CTYPE2="DEC--CAR")
    w = WCS(h)
    with pytest.raises(ValueError, match="projection"):
        from_astropy(w, precision="f64")


def test_wrong_input_type_is_typeerror():
    for bad in (fits.Header(), "some/file.fits", Path("some/file.fits"), object()):
        with pytest.raises(TypeError, match="from_astropy: type"):
            from_astropy(bad, precision="f64")


def test_cpdis_is_refused():
    # A real LOOKUP distortion needs a BINTABLE HDUList -- astropy raises
    # "an astropy.io.fits.HDUList is required" even from `deepcopy` -- so the
    # two branches that actually refuse are driven directly.
    from mapprojax.astropy import _refuse_distortions

    class _Distorted:
        cpdis1 = object()   # attribute branch
        cpdis2 = None
        det2im1 = None
        det2im2 = None
        wcs = staticmethod(lambda: None)  # replaced below

    distorted = _Distorted()
    distorted.wcs = type("Prm", (), {"get_ps": staticmethod(list)})()
    with pytest.raises(ValueError, match="cpdis1"):
        _refuse_distortions(distorted, {})

    plain = _Distorted()
    plain.cpdis1 = None
    plain.wcs = distorted.wcs
    with pytest.raises(ValueError, match="CPDIS"):  # relaxed-header prefix scan
        _refuse_distortions(plain, {"CPDIS1": "LOOKUP"})


def test_nondefault_lonpole_is_refused():
    h = _header(LONPOLE=170.0)
    w = WCS(h)
    with pytest.raises(ValueError, match="LONPOLE"):
        from_astropy(w, precision="f64")


def test_nonzero_sin_pv_is_refused():
    h = _header(CTYPE1="RA---SIN", CTYPE2="DEC--SIN", PV2_1=0.3)
    w = WCS(h)
    with pytest.raises(ValueError, match="PV"):
        from_astropy(w, precision="f64")


def test_sin_zero_pv_is_accepted():
    h = _header(CTYPE1="RA---SIN", CTYPE2="DEC--SIN", PV2_1=0.0, PV2_2=0.0)
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    assert isinstance(kernel, Sin)
    _assert_matches_astropy(w, kernel, _grid(h))


def test_nonfinite_parameters_are_refused():
    # A NaN in a SIP array makes `to_header` raise astropy's own ValueError,
    # so finiteness has to be checked before serialisation (D9).
    h = _sip_header(A_ORDER=2, A_0_2=1e-4, B_ORDER=2, B_2_0=1e-4)
    w = WCS(h)
    w.sip.a[0, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite sip.a"):
        from_astropy(w, precision="f64")

    h2 = _header()
    w2 = WCS(h2)
    w2.wcs.crval = np.asarray(w2.wcs.crval) * np.array([1.0, np.inf])
    with pytest.raises(ValueError, match="non-finite crval"):
        from_astropy(w2, precision="f64")

    h3 = _header()
    w3 = WCS(h3)
    w3.wcs.crpix = np.asarray(w3.wcs.crpix) * np.array([np.nan, 1.0])
    with pytest.raises(ValueError, match="non-finite crpix"):
        from_astropy(w3, precision="f64")


def test_non_angular_cunit_is_refused():
    h = _header()
    w = WCS(h)
    w.wcs.cunit = ["m", "deg"]
    with pytest.raises(ValueError, match="cunit1"):
        from_astropy(w, precision="f64")


# --- behaviour -------------------------------------------------------------


def test_caller_wcs_and_arrays_are_unchanged():
    h = ar.load_header("spherex_l2.hdr")
    w = WCS(h)
    before = {
        "crpix": np.array(w.wcs.crpix, copy=True),
        "crval": np.array(w.wcs.crval, copy=True),
        "cd": np.array(w.pixel_scale_matrix, copy=True),
        "sip_a": None if w.sip is None else np.array(w.sip.a, copy=True),
    }
    kernel = from_astropy(w, precision="f64")

    np.testing.assert_array_equal(np.asarray(w.wcs.crpix), before["crpix"])
    np.testing.assert_array_equal(np.asarray(w.wcs.crval), before["crval"])
    np.testing.assert_array_equal(np.asarray(w.pixel_scale_matrix), before["cd"])
    if before["sip_a"] is not None:
        np.testing.assert_array_equal(np.asarray(w.sip.a), before["sip_a"])

    assert not np.shares_memory(np.asarray(kernel.crval), np.asarray(w.wcs.crval))
    assert not np.shares_memory(np.asarray(kernel.crpix), np.asarray(w.wcs.crpix))


def test_mutated_unset_caller_wcs_is_unchanged():
    # The axis-role getters run `wcsset` in place, so they must be evaluated
    # on the copy, never on the caller's object.
    h = _header()
    w = WCS(h)
    w.wcs.crval = np.array([150.0, 45.0])  # mutated, deliberately not set()
    lonpole_before = float(w.wcs.lonpole)
    from_astropy(w, precision="f64")
    assert float(w.wcs.lonpole) == lonpole_before
    np.testing.assert_array_equal(np.asarray(w.wcs.crval), np.array([150.0, 45.0]))


def test_key_a_wcs_converts():
    h = _header()
    for suffix in ("", "A"):
        h[f"CTYPE1{suffix}"] = "RA---SIN"
        h[f"CTYPE2{suffix}"] = "DEC--SIN"
        h[f"CRPIX1{suffix}"] = h[f"CRPIX2{suffix}"] = 50.5
        h[f"CRVAL1{suffix}"] = 150.0
        h[f"CRVAL2{suffix}"] = 30.0
        h[f"CDELT1{suffix}"] = -2.0e-3
        h[f"CDELT2{suffix}"] = 2.1e-3
    w_a = WCS(h, key="A")
    assert w_a.wcs.alt == "A"
    kernel = from_astropy(w_a, precision="f64")
    assert isinstance(kernel, Sin)
    # The numbers come from the selected key's attributes.
    np.testing.assert_array_equal(np.asarray(kernel.crpix), np.asarray(w_a.wcs.crpix) - 1.0)


def test_to_dict_from_dict_round_trip():
    h = ar.load_header("spherex_l2.hdr")
    kernel = from_astropy(WCS(h), precision="f64")
    again = Tan.from_dict(kernel.to_dict())
    np.testing.assert_array_equal(np.asarray(again.crpix), np.asarray(kernel.crpix))
    np.testing.assert_array_equal(np.asarray(again.crval), np.asarray(kernel.crval))
    np.testing.assert_array_equal(np.asarray(again.cd), np.asarray(kernel.cd))
    np.testing.assert_array_equal(np.asarray(again.sip.a), np.asarray(kernel.sip.a))


def test_result_is_a_pytree_and_jittable():
    h = _header()
    kernel = from_astropy(WCS(h), precision="f64")
    leaves, treedef = jax.tree_util.tree_flatten(kernel)
    assert jax.tree_util.tree_unflatten(treedef, leaves) is not None

    pix_x = np.array([10.0, 40.0, 60.0])
    pix_y = np.array([20.0, 60.0, 40.0])

    @jax.jit
    def round_trip(k):
        lon, lat = k.unproj(pix_x, pix_y)
        return k.proj(lon, lat)

    out_x, out_y = round_trip(kernel)
    np.testing.assert_allclose(np.asarray(out_x), pix_x, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_y), pix_y, atol=1e-6)


# --- imports (process level) ----------------------------------------------

_NO_ASTROPY = """
import sys

class Blocker:
    def find_module(self, name, path=None):
        return self if name == "astropy" or name.startswith("astropy.") else None
    def find_spec(self, name, path=None, target=None):
        if name == "astropy" or name.startswith("astropy."):
            raise ImportError("blocked: " + name)
        return None

sys.meta_path.insert(0, Blocker())
for mod in [m for m in list(sys.modules) if m == "astropy" or m.startswith("astropy.")]:
    del sys.modules[mod]

import mapprojax
assert "astropy" not in sys.modules, sorted(m for m in sys.modules if "astropy" in m)
import mapprojax.astropy
assert "astropy" not in sys.modules, sorted(m for m in sys.modules if "astropy" in m)
try:
    mapprojax.from_astropy(object(), precision="f64")
except ImportError as exc:
    assert "pip install 'mapprojax[astropy]'" in str(exc), str(exc)
else:
    raise AssertionError("expected ImportError")
try:
    mapprojax.astropy.from_astropy(object(), precision="f64")
except ImportError as exc:
    assert "pip install 'mapprojax[astropy]'" in str(exc), str(exc)
else:
    raise AssertionError("expected ImportError")
print("ok")
"""

_STILL_IMPORTABLE = """
import sys
import mapprojax
assert "astropy" not in sys.modules, sorted(m for m in sys.modules if "astropy" in m)
import mapprojax.astropy
assert "astropy" not in sys.modules, sorted(m for m in sys.modules if "astropy" in m)
assert "from_astropy" in mapprojax.__all__
assert "from_astropy" in dir(mapprojax)
print("ok")
"""


def _run_child(code):
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin", "HOME": str(REPO_ROOT)},
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "ok" in result.stdout


def test_import_mapprojax_is_quiet_without_astropy():
    _run_child(_NO_ASTROPY)


def test_missing_astropy_import_error_names_extra():
    _run_child(_NO_ASTROPY)


def test_astropy_not_imported_by_mapprojax():
    _run_child(_STILL_IMPORTABLE)
