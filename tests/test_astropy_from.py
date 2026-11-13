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


def _to_deg(wcs, values):
    """Convert ``all_pix2world`` output from the WCS's own unit to degrees.

    With ``preserve_units=True`` astropy returns world coordinates in the
    declared CUNIT (rad, arcsec, ...), while the kernel is always in degrees.
    """
    import astropy.units as u

    out = np.array(values, dtype=np.float64)
    for axis in (0, 1):
        unit = wcs.wcs.cunit[axis]
        text = str(unit).strip()
        if text and text != "deg":
            scale = float(u.Unit(text).to(u.deg))
            out[:, axis] = out[:, axis] * scale
    return out


def _assert_matches_astropy(wcs, kernel, pix, atol_arcsec=1e-6):
    """Forward agreement with ``all_pix2world`` on the given pixels."""
    want = _to_deg(wcs, wcs.all_pix2world(pix, 0))
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
    """The fp64 corpus contract: FORWARD_MAS forward, INVERSE_PX back.

    Both budgets come from the reference harness rather than local constants,
    so this cannot drift looser than the recorded accuracy claim.
    """
    h = case.header
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    pix = ar.pixel_grid(h)

    want = _to_deg(w, w.all_pix2world(pix, 0))
    lon, lat = kernel.unproj(pix[:, 0], pix[:, 1])
    fwd = float(ar.separation_mas(np.degrees(np.asarray(lon)),
                                   np.degrees(np.asarray(lat)),
                                   want[:, 0], want[:, 1]).max())
    assert fwd <= ar.FORWARD_MAS, f"{case.name}: forward {fwd:.3e} mas > {ar.FORWARD_MAS}"

    # Inverse against a converged all_world2pix, radial error in pixels.
    sky = np.array(w.all_pix2world(pix, 0), dtype=np.float64)
    back = np.array(w.all_world2pix(sky, 0, tolerance=ar.ORACLE_TOL,
                                     maxiter=ar.ORACLE_MAXITER, quiet=False))
    got_x, got_y = kernel.proj(np.radians(sky[:, 0]), np.radians(sky[:, 1]))
    radial = np.hypot(np.asarray(got_x) - back[:, 0], np.asarray(got_y) - back[:, 1])
    inv = float(radial.max())
    assert inv <= ar.INVERSE_PX, f"{case.name}: inverse {inv:.3e} px > {ar.INVERSE_PX}"


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
    for case in ar.cases():
        h = case.header
        w = WCS(h)
        kernel = from_astropy(w, precision="mixed")
        pix = ar.pixel_grid(h).astype(np.float32)
        want = np.array(w.all_pix2world(pix, 0))
        lon, lat = kernel.unproj(pix[:, 0], pix[:, 1])
        # The baselines' unit is mas, so compare in mas (the cap is ~14, not
        # 14 arcsec -- an arcsec comparison would be 1000x too loose).
        fwd = float(ar.separation_mas(np.degrees(np.asarray(lon)),
                                       np.degrees(np.asarray(lat)),
                                       want[:, 0], want[:, 1]).max())
        cap = ar.fp32_cap(case, "jax_mixed", "forward")
        assert fwd <= cap, f"{case.name}: {fwd:.4f} mas > cap {cap}"

        # Inverse direction, same contract: a converged oracle inverse and the
        # radial error, not the largest single component.
        sky = np.array(w.all_pix2world(pix.astype(np.float64), 0))
        want_back = np.array(w.all_world2pix(sky, 0, tolerance=ar.ORACLE_TOL,
                                              maxiter=ar.ORACLE_MAXITER, quiet=False))
        back_x, back_y = kernel.proj(np.radians(sky[:, 0]), np.radians(sky[:, 1]))
        inv = float(np.hypot(np.asarray(back_x) - want_back[:, 0],
                              np.asarray(back_y) - want_back[:, 1]).max())
        assert inv <= ar.fp32_cap(case, "jax_mixed", "inverse"), (
            f"{case.name}: inverse {inv} px > cap {ar.fp32_cap(case, 'jax_mixed', 'inverse')}"
        )

    # f32: no absolute proj, but the offset pipeline must round trip.
    h = ar.load_header("spherex_l2.hdr")
    w = WCS(h)
    f32 = from_astropy(w, precision="f32")
    with pytest.raises(TypeError):
        f32.unproj(pix[:, 0], pix[:, 1])
    pix = ar.pixel_grid(h)[:64].astype(np.float32)
    off_x, off_y = f32.unproj_offset(pix[:, 0], pix[:, 1])
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
    # `preserve_units=True` is what keeps cunit/CRVAL/CDELT in the declared
    # unit; the default normalises everything to degrees in __init__, so
    # without it `_deg_scale` would never see a non-degree axis.
    w = WCS(h, preserve_units=preserve)
    w.wcs.set()
    expected_unit = "deg" if (unit == "deg" or not preserve) else unit
    assert str(w.wcs.cunit[0]) == expected_unit, (str(w.wcs.cunit[0]), expected_unit)
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
    w = WCS(h, preserve_units=True)
    w.wcs.set()
    assert str(w.wcs.cunit[0]) == "arcsec" and str(w.wcs.cunit[1]) == "deg"

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
    # Accepting it is not enough: at a pole the sky position is degenerate, so
    # the pixels must be checked against astropy rather than just constructed.
    pix = np.array([[49.5, 49.5], [10.0, 20.0], [80.0, 70.0]])
    inside = np.isfinite(np.array(w.all_pix2world(pix, 0))).all(axis=1)
    assert inside.any()
    _assert_matches_astropy(w, kernel, pix[inside])


def test_crval2_plus_90_with_180_accepted():
    h = _pole_header(90.0, LONPOLE=180.0)
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    assert kernel is not None
    pix = np.array([[49.5, 49.5], [10.0, 20.0], [80.0, 70.0]])
    inside = np.isfinite(np.array(w.all_pix2world(pix, 0))).all(axis=1)
    assert inside.any()
    _assert_matches_astropy(w, kernel, pix[inside])


def test_crval2_plus_90_without_lonpole_refused():
    # A materialised WCS cannot distinguish an omitted LONPOLE from an
    # explicit 0.0 at CRVAL2 = +90, so the refusal quotes the resolved value.
    h = _pole_header(90.0)
    w = WCS(h)
    assert float(w.wcs.lonpole) == 0.0
    with pytest.raises(ValueError, match=r"LONPOLE = 0\.0"):
        from_astropy(w, precision="f64")


def test_longitude_wrap_matches_astropy():
    h = _header(CRVAL1=0.2)
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    pix = np.array([[0.0, 0.0], [99.0, 99.0], [49.5, 0.0], [0.0, 49.5]])
    _assert_matches_astropy(w, kernel, pix)


def test_sin_horizon_validity_mask_matches_astropy():
    """A SIN count past the unit disc must go NaN exactly where astropy does.

    A small ordinary field never reaches the horizon, so the old test proved
    nothing about the validity mask. This sweeps a coarse SIN grid across the
    disc edge and requires both implementations to agree sample for sample,
    then checks the values still inside the disc.
    """
    h = _header(
        CTYPE1="RA---SIN", CTYPE2="DEC--SIN",
        CRPIX1=50.5, CRPIX2=50.5, CRVAL1=150.0, CRVAL2=30.0,
        CDELT1=-0.5, CDELT2=0.5, NAXIS1=400,
    )
    w = WCS(h)
    kernel = from_astropy(w, precision="f64")
    x = np.linspace(49.5, 300.0, 60)
    y = np.full_like(x, 50.0)
    pix = np.column_stack([x, y])

    lon, lat = kernel.unproj(x, y)
    ours = ~np.isfinite(np.asarray(lon)) | ~np.isfinite(np.asarray(lat))
    theirs = ~np.isfinite(np.array(w.all_pix2world(pix, 0))).all(axis=1)

    assert ours.any(), "the sweep never left the disc, so it proves nothing"
    np.testing.assert_array_equal(ours, theirs, err_msg="SIN validity mask differs")

    # Inside the disc the values must still match astropy.
    _assert_matches_astropy(w, kernel, pix[~ours])


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


@pytest.mark.parametrize("name", FIXTURES)
def test_round_trip_fidelity_on_every_fixture(name):
    """to_dict/from_dict and jit must hold for every D12 fixture, not one."""
    h = ar.load_header(name)
    kernel = from_astropy(WCS(h), precision="f64")
    again = Tan.from_dict(kernel.to_dict()) if isinstance(kernel, Tan) else Sin.from_dict(
        kernel.to_dict()
    )
    for field in ("crpix", "crval", "cd"):
        np.testing.assert_array_equal(np.asarray(getattr(again, field)),
                                      np.asarray(getattr(kernel, field)))
    if kernel.sip is not None:
        for field in ("a", "b", "ap", "bp"):
            want = getattr(kernel.sip, field, None)
            got = getattr(again.sip, field, None)
            if want is None:
                assert got is None
            else:
                np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    pix_x = np.array([10.0, 40.0])
    pix_y = np.array([20.0, 60.0])

    @jax.jit
    def round_trip(k):
        lon, lat = k.unproj(pix_x, pix_y)
        return k.proj(lon, lat)

    out_x, out_y = round_trip(kernel)
    np.testing.assert_allclose(np.asarray(out_x), pix_x, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_y), pix_y, atol=1e-6)


def test_wcs_test_data_loader_equivalence():
    """The JSON fixture joins the D12 leaf-for-leaf set."""
    raw = json.loads((DATA_DIR / "wcs_test_data.json").read_text())["header_keywords"]
    h = fits.Header()
    for key, value in raw.items():
        h[key] = value
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 100, 100

    kernel = from_astropy(WCS(h), precision="f64")
    crpix, cd, crval, sip = load_wcs_params_from_header(h)
    np.testing.assert_array_equal(np.asarray(kernel.crpix), np.asarray(crpix))
    np.testing.assert_allclose(np.asarray(kernel.cd), np.asarray(cd), rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(kernel.crval), np.asarray(crval))
    if sip is not None:
        for field in ("a", "b", "ap", "bp"):
            want = getattr(sip, field, None)
            got = getattr(kernel.sip, field, None)
            if want is None:
                assert got is None
            else:
                np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


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


def _unset_wcs(crval2=30.0):
    """A WCS built by hand and never `set()`: `lonpole` is still NaN.

    A header-built WCS has already resolved LONPOLE, so it cannot detect the
    fault this test is named for.
    """
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crpix = [50.5, 50.5]
    w.wcs.crval = [150.0, crval2]
    w.wcs.cdelt = [-2.0e-3, 2.1e-3]
    return w


def test_mutated_unset_caller_wcs_is_unchanged():
    # The axis-role getters run `wcsset` in place, so they must be evaluated
    # on the copy, never on the caller's object.
    w = _unset_wcs()
    assert not np.isfinite(float(w.wcs.lonpole)), "fixture is already set()"
    from_astropy(w, precision="f64")
    assert not np.isfinite(float(w.wcs.lonpole)), "conversion set the caller"
    np.testing.assert_array_equal(np.asarray(w.wcs.crval), np.array([150.0, 30.0]))

    # The same must hold when the conversion is refused.
    refused = _unset_wcs(crval2=40.0)
    refused.wcs.cunit = ["m", "deg"]
    with pytest.raises(ValueError, match="cunit1"):
        from_astropy(refused, precision="f64")
    assert not np.isfinite(float(refused.wcs.lonpole)), "refusal set the caller"


def test_key_a_wcs_converts():
    h = _header()
    for suffix in ("", "A"):
        h[f"CTYPE1{suffix}"] = "RA---SIN"
        h[f"CTYPE2{suffix}"] = "DEC--SIN"
        h[f"CRPIX1{suffix}"] = h[f"CRPIX2{suffix}"] = 50.5
        h[f"CRVAL1{suffix}"] = 150.0
        h[f"CRVAL2{suffix}"] = 30.0
        # Distinct alternate values, so the test cannot pass by reading the
        # primary key twice.
        h[f"CDELT1{suffix}"] = -2.0e-3 if not suffix else -3.0e-3
        h[f"CDELT2{suffix}"] = 2.1e-3 if not suffix else 2.7e-3
    w_a = WCS(h, key="A")
    assert w_a.wcs.alt == "A"
    kernel = from_astropy(w_a, precision="f64")
    assert isinstance(kernel, Sin)
    # The numbers come from the selected key, so the alternate scale must win
    # over the primary one.
    np.testing.assert_array_equal(np.asarray(kernel.crpix), np.asarray(w_a.wcs.crpix) - 1.0)
    assert not np.allclose(np.asarray(kernel.cd), np.diag([-2.0e-3, 2.1e-3]))
    np.testing.assert_allclose(np.asarray(kernel.cd), np.diag([-3.0e-3, 2.7e-3]))


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
        env={
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "PATH": "/usr/bin:/bin",
            "HOME": str(REPO_ROOT),
            # Pin the backend: without it a venv that carries the ROCm plugin
            # but sees no device fails in the child with HIP_ERROR_NoDevice,
            # and a venv with a device would silently test the GPU instead.
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
        },
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "ok" in result.stdout


# `_NO_ASTROPY` covers the install hint; this shorter child isolates the
# quiet-import claim, under -W error so any warning is a failure.
_QUIET_IMPORT = """
import sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "astropy" or name.startswith("astropy."):
            raise ImportError("blocked: " + name)
        return None

sys.meta_path.insert(0, Blocker())

import warnings
warnings.simplefilter("error")
import mapprojax
import mapprojax.astropy
assert "astropy" not in sys.modules, sorted(m for m in sys.modules if "astropy" in m)
assert "from_astropy" in mapprojax.__all__
print("ok")
"""


def test_import_mapprojax_is_quiet_without_astropy():
    _run_child(_QUIET_IMPORT)


def test_missing_astropy_import_error_names_extra():
    _run_child(_NO_ASTROPY)


def test_astropy_not_imported_by_mapprojax():
    _run_child(_STILL_IMPORTABLE)


# --- additional plan-mandated coverage (Stage 2a gate) ---------------------


def test_sip_none_a_with_b_present_is_unreachable():
    """Q6: `a=None` with `b` present cannot reach `from_astropy` at all.

    `astropy.wcs.Sip` (and therefore `wcsset`) requires both A and B, so the
    assignment itself raises before any conversion can happen. The plan
    therefore pins the behaviour here instead of adding a refusal: the missing
    polynomial is simply not constructible on a real WCS.
    """
    h = _sip_header(A_ORDER=2, A_0_2=1e-4, B_ORDER=2, B_2_0=1e-4)
    w = WCS(h)
    old = w.sip
    with pytest.raises(Exception, match="Both A and B SIP transform must be defined"):
        w.sip = Sip(None, old.b, None, old.bp, np.asarray(old.crpix))

    # With both polynomials absent entirely, the kernel is a plain Tan and the
    # `-SIP` CTYPE suffix is what gets refused (the other half of the rule).
    w2 = WCS(_sip_header())
    assert w2.sip is None
    with pytest.raises(ValueError, match="-SIP without SIP"):
        from_astropy(w2, precision="f64")


def test_real_lookup_table_distortion_is_refused():
    """A genuine `DistortionLookupTable` through the public entry point.

    The private-helper tests use stand-ins because `_TAB`/`CPDIS` headers
    cannot be rebuilt from cards; this one proves the attribute path on a real
    astropy object.
    """
    from astropy.wcs import DistortionLookupTable

    w = WCS(_header())
    f32 = lambda value: np.array(value, dtype=np.float32)
    w.cpdis1 = DistortionLookupTable(
        f32(np.zeros((4, 4))), f32([0.0, 0.0]), f32([1.0, 1.0]), f32([1.0, 1.0])
    )
    with pytest.raises(ValueError, match=r"cpdis1 is unsupported"):
        from_astropy(w, precision="f64")


def test_det2im_is_refused():
    """The DET2IM attribute branch, which needs an HDUList to construct."""
    from mapprojax.astropy import _refuse_distortions

    class _Det2Im:
        cpdis1 = cpdis2 = det2im1 = det2im2 = None
        wcs = type("Prm", (), {"get_ps": staticmethod(list)})(),

    distorted = _Det2Im()
    distorted.det2im1 = object()
    with pytest.raises(ValueError, match="det2im1"):
        _refuse_distortions(distorted, {})


def test_nonfinite_cd_sentinel_is_caught():
    """A NaN in `wcs.cd` becomes the finite sentinel in `pixel_scale_matrix`.

    WCSLIB rewrites NaN to 9.87654321e+107 and `pixel_scale_matrix` keeps
    it, so `np.isfinite(pixel_scale_matrix)` passes. Only the raw pass over
    `wcs.wcs.cd` sees the NaN (D9).
    """
    h = _header()
    w = WCS(h)
    cd = np.asarray(w.wcs.cd) if w.wcs.has_cd() else None
    if cd is None:
        w.wcs.cd = np.array([[-2.0e-3, 0.0], [0.0, 2.1e-3]])
        w.wcs.set()
        cd = np.asarray(w.wcs.cd)
    cd = cd.copy()
    cd[1, 0] = np.nan
    w.wcs.cd = cd

    with pytest.raises(ValueError, match="non-finite cd"):
        from_astropy(w, precision="f64")

    # And prove the sentinel really does hide it after initialisation.
    import copy as _copy

    w2 = _copy.deepcopy(w)
    w2.wcs.set()
    assert bool(np.isfinite(np.asarray(w2.pixel_scale_matrix)).all())
    assert not bool(np.isfinite(np.asarray(w.wcs.cd)).all())


def test_nonfinite_sip_crpix_is_refused():
    h = _sip_header(A_ORDER=2, A_0_2=1e-4, B_ORDER=2, B_2_0=1e-4)
    w = WCS(h)
    old = w.sip
    bad = np.asarray(old.crpix, dtype=np.float64).copy()
    bad[0] = np.nan
    w.sip = Sip(old.a, old.b, old.ap, old.bp, bad)
    with pytest.raises(ValueError, match="non-finite sip.crpix"):
        from_astropy(w, precision="f64")


def test_second_axis_ctype_is_classified():
    """TPV/TPD/-TAB and `-SIP` can hide on CTYPE2 as well as CTYPE1."""
    from mapprojax.astropy import _refuse_ctype

    for ctype2, token in (
        ("DEC--TAB", "-TAB"),
        ("DEC--TPV", "TPV"),
    ):
        with pytest.raises(ValueError, match=token):
            _refuse_ctype({"CTYPE1": "RA---TAN", "CTYPE2": ctype2})


def test_f32_with_x64_off_emits_no_truncation_warnings():
    """f32 construction must be quiet with x64 disabled and warnings fatal."""
    child = """
import warnings
import numpy as np
warnings.simplefilter("error")
from astropy.io import fits
from astropy.wcs import WCS
from mapprojax import from_astropy

h = fits.Header()
h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 64, 64
h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
h["CRPIX1"], h["CRPIX2"] = 32.5, 32.5
h["CRVAL1"], h["CRVAL2"] = 150.0, 30.0
h["CDELT1"], h["CDELT2"] = -2.0e-3, 2.1e-3
kernel = from_astropy(WCS(h), precision="f32")
x, y = kernel.unproj_offset(np.array([10.0, 20.0], np.float32),
                             np.array([10.0, 20.0], np.float32))
assert np.isfinite(np.asarray(x)).all() and np.isfinite(np.asarray(y)).all()
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        check=False,
        text=True,
        cwd=REPO_ROOT,
        env={
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "PATH": "/usr/bin:/bin",
            "JAX_ENABLE_X64": "0",
            # Same reason as _run_child: the x64-off contract must be proven on
            # CPU, not on whatever device happens to be visible.
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
        },
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "ok" in result.stdout


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_f32_reprojector_pair_within_pair_caps(direction):
    """The f32 pair contract over kernels built by `from_astropy`.

    Drives the recorded two-field pair (SPHEREx template <-> SPHEREx L2) in
    both directions with `from_astropy` kernels, holding the overlap error to
    the recorded `pair.<direction>/f32` cap rather than reprojecting a field
    onto itself.

    Every expected overlap sample must be finite, matching the harness
    contract in tests/test_reproject_astropy.py: a NaN or inf inside the
    overlap is a defect, not a sample to skip with `nanmax`.
    """
    from mapprojax import Reprojector

    src_name, dst_name, _convert, transpose = ar.PAIR_DIRECTIONS[direction]
    w_src = from_astropy(WCS(ar.load_header(f"{src_name}.hdr")), precision="f32")
    w_dst = from_astropy(WCS(ar.load_header(f"{dst_name}.hdr")), precision="f32")
    m = ar.galactic_to_icrs_matrix()
    rotation = m if not transpose else m.T

    reproj = Reprojector(w_src, w_dst, frame_rotation=rotation, matrix_precision="f32")
    ref = ar.pair_reference(direction)
    src_pix = ref["src_pix"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x, y = reproj.pix_to_pix(src_pix[:, 0], src_pix[:, 1])
    got = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    inside = ref["overlap"]
    assert inside.sum() > 0, f"{direction}: no overlap samples at all"
    err = np.hypot(*(got - ref["dst_pix"]).T)[inside]
    assert np.isfinite(err).all(), (
        f"{direction}: {int((~np.isfinite(err)).sum())} non-finite of "
        f"{err.size} overlap samples"
    )
    cap = ar.pair_cap(direction, "f32")
    assert float(err.max()) <= cap, f"{direction}: {err.max():.3e} px > cap {cap} px"
