import json
from pathlib import Path
import numpy as np
import jax
# Ensure JAX 64-bit is enabled BEFORE importing mapprojax to avoid warning
jax.config.update("jax_enable_x64", True)
import pytest

import mapprojax
from mapprojax.wcs import Sin, Tan
from mapprojax.sip import Sip
from mapprojax.wcs_utils import load_wcs_params_from_header

DATA_DIR = Path(__file__).resolve().parent / 'data'


def test_wcs_consistency_with_json():
    json_path = DATA_DIR / 'wcs_test_data.json'
    assert json_path.exists(), f"required fixture missing: {json_path}"

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Use the utility to parse parameters from the header dictionary
    crpix, cd, crval, sip = load_wcs_params_from_header(data['header_keywords'])
    
    # The precision-explicit class is a JAX PyTree.
    wcs_f64 = Tan(crpix, cd, crval, sip=sip, precision="f64")
    
    # Prepare Data
    input_pixels = np.array(data['input_pixel_coordinates'])
    expected_world = np.array(data['output_world_coordinates']) # Degrees
    
    x_in = input_pixels[:, 0]
    y_in = input_pixels[:, 1]

    # --- Test SIP-Only Part (Internal Consistency & Focal Plane match) ---
    # We verify the internal consistency of the SIP implementation (Forward + Reverse).
    # We also compare against data['focal_plane_coordinates'] using the convention 
    # defined in the metadata: "Relative to FITS CRPIX (u = pix_0 - CRPIX_fits)".
    
    u_mapproj = x_in - crpix[0]
    v_mapproj = y_in - crpix[1]
    
    # 1. Forward SIP: (u, v) -> (U, V) [Internal Round-trip check]
    U_calc_m, V_calc_m = sip.pix_to_foc(u_mapproj, v_mapproj)
    u_back_m, v_back_m = sip.foc_to_pix(U_calc_m, V_calc_m)
    
    np.testing.assert_allclose(u_back_m, u_mapproj, rtol=0, atol=1e-6, err_msg="SIP Round Trip u mismatch")
    np.testing.assert_allclose(v_back_m, v_mapproj, rtol=0, atol=1e-6, err_msg="SIP Round Trip v mismatch")
    
    # 2. Match against JSON focal_plane_coordinates
    # Note: These values in the JSON are effectively (pix_0 - CRPIX_fits) + f(pix_0 - CRPIX_0),
    # which is sip.pix_to_foc(u_0base, v_0base) - 1.0. 
    # This mismatch is likely due to how the generation script (astropy) handled 
    # origins vs SIP polynomial relative coordinates.
    U_calc_f, V_calc_f = sip.pix_to_foc(u_mapproj, v_mapproj)
    expected_foc = np.array(data['focal_plane_coordinates'])
    
    np.testing.assert_allclose(U_calc_f - 1.0, expected_foc[:, 0], rtol=0, atol=1e-7, err_msg="SIP Focal Plane U mismatch")
    np.testing.assert_allclose(V_calc_f - 1.0, expected_foc[:, 1], rtol=0, atol=1e-7, err_msg="SIP Focal Plane V mismatch")

    # 3. Reverse from Focal Plane back to pixels.
    #
    # The saved focal coordinates are 1 px below mapprojax's 0-based focal
    # plane (U - expected == 1 for every sample), so the offset is added back
    # before the inversion. Comparing
    # foc_to_pix(expected_foc) against u - 1 mixes that conversion into a
    # nonlinear inverse and leaves a 0.011 px residue for order-3 SIP, which
    # the old 5e-2 tolerance hid.
    u_back_f, v_back_f = sip.foc_to_pix(expected_foc[:, 0] + 1.0, expected_foc[:, 1] + 1.0)
    np.testing.assert_allclose(u_back_f, u_mapproj, rtol=0, atol=1e-6, err_msg="SIP Reverse from Focal Plane mismatch")
    np.testing.assert_allclose(v_back_f, v_mapproj, rtol=0, atol=1e-6, err_msg="SIP Reverse from Focal Plane mismatch")
    
    print("SIP-only transformation (Focal Plane match & Round Trip) matched within tolerance.")

    # --- Test Forward Projection (Pixel -> World) ---
    
    ra_f64_rad, dec_f64_rad = wcs_f64.unproj(x_in, y_in)
    ra_f64, dec_f64 = np.degrees(ra_f64_rad), np.degrees(dec_f64_rad)
    
    # Compare with Expected
    # Using slightly loose tolerance as the JSON source might have subtle differences 
    # (e.g., astropy version differences, float precision in JSON)
    # But usually 1e-6 degrees is good enough for headers.
    
    np.testing.assert_allclose(ra_f64, expected_world[:, 0], rtol=0, atol=1e-6, err_msg="f64 RA mismatch")
    np.testing.assert_allclose(dec_f64, expected_world[:, 1], rtol=0, atol=1e-6, err_msg="f64 Dec mismatch")
    
    print("Forward Projection (Pixel -> World) matched expected values.")

    # --- Test Inverse Projection (World -> Pixel) ---
    
    ra_in_rad = np.radians(expected_world[:, 0])
    dec_in_rad = np.radians(expected_world[:, 1])
    
    x_out_f64, y_out_f64 = wcs_f64.proj(ra_in_rad, dec_in_rad)
    
    # Compare with Input Pixels
    # This validates the Round Trip + Inverse Logic
    # Keep the independent fixture's existing 1e-6 px inversion tolerance.
    np.testing.assert_allclose(x_out_f64, x_in, rtol=0, atol=1e-6, err_msg="f64 Inverse X mismatch")
    np.testing.assert_allclose(y_out_f64, y_in, rtol=0, atol=1e-6, err_msg="f64 Inverse Y mismatch")
    
    print("Inverse Projection (World -> Pixel) matched within 1e-6 px.")

if __name__ == "__main__":
    # Allow running directly
    try:
        test_wcs_consistency_with_json()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_root_exports_precision_wcs_without_legacy_aliases():
    assert mapprojax.Tan is Tan
    assert mapprojax.Sin is Sin
    for obsolete in (
        "TanArray", "SinArray", "TanJax", "SinJax", "TanJaxArray",
        "SinJaxArray", "TanJaxMixed", "SinJaxMixed", "TanJaxArrayMixed",
        "SinJaxArrayMixed", "WCSBase", "WCSArrayBase",
    ):
        assert not hasattr(mapprojax, obsolete), obsolete


@pytest.mark.parametrize("cls", [Tan, Sin], ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("precision", ["f32", "mixed", "f64"])
@pytest.mark.parametrize("with_sip", [False, True], ids=["no-sip", "ab-only-sip"])
def test_wcs_json_round_trip_preserves_precision_and_state(cls, precision, with_sip):
    sip = None
    if with_sip:
        a = np.zeros((3, 3)); a[2, 0] = 1e-5
        b = np.zeros((3, 3)); b[0, 2] = -2e-5
        sip = Sip(a=a, b=b, inverse_steps=4)

    wcs = cls([12.0, -7.0], [[-1e-3, 2e-6], [3e-6, 1e-3]],
              [120.0, 35.0], sip=sip, precision=precision)
    restored = cls.from_dict(json.loads(json.dumps(wcs.to_dict())))

    assert restored.precision == precision
    assert restored.crval_dtype == wcs.crval_dtype
    for name in ("crpix", "cd", "crval"):
        np.testing.assert_array_equal(np.asarray(getattr(restored, name)),
                                      np.asarray(getattr(wcs, name)))
    if with_sip:
        assert restored.sip.ap is None and restored.sip.bp is None
        assert restored.sip.inverse_steps == 4
        np.testing.assert_array_equal(np.asarray(restored.sip.a), np.asarray(wcs.sip.a))
        np.testing.assert_array_equal(np.asarray(restored.sip.b), np.asarray(wcs.sip.b))
    else:
        assert restored.sip is None

    x = np.asarray([10.0, 15.0])
    y = np.asarray([-2.0, 4.0])
    method = "unproj_offset" if precision == "f32" else "unproj"
    for actual, expected in zip(getattr(restored, method)(x, y),
                                getattr(wcs, method)(x, y)):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_wcs_json_rejects_mismatched_and_legacy_types():
    data = Tan([0.0, 0.0], [[-1e-3, 0.0], [0.0, 1e-3]], [0.0, 0.0],
               precision="f64").to_dict()
    data["type"] = "Sin"
    with pytest.raises(ValueError, match="expected WCS type"):
        Tan.from_dict(data)

    legacy = {key: data[key] for key in ("crpix", "cd", "crval", "sip")}
    with pytest.raises(ValueError, match="precision-aware contract"):
        Tan.from_dict(legacy)
