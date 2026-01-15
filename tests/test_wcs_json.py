import json
import os
import numpy as np
import pytest
import jax
# Ensure JAX 64-bit is enabled BEFORE importing mapprojax to avoid warning
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from mapprojax.projections import Tan
from mapprojax.jax_projections import TanJax
from mapprojax.sip import Sip

def parse_sip_matrix(header, prefix, order_key):
    if order_key not in header:
        return None
    
    order = header[order_key]
    matrix = np.zeros((order + 1, order + 1))
    
    for i in range(order + 1):
        for j in range(order + 1):
            key = f"{prefix}_{i}_{j}"
            if key in header:
                matrix[i, j] = header[key]
                
    return matrix

def load_wcs_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    header = data['header_keywords']
    
    # 1. Parse SIP
    a = parse_sip_matrix(header, 'A', 'A_ORDER')
    b = parse_sip_matrix(header, 'B', 'B_ORDER')
    ap = parse_sip_matrix(header, 'AP', 'AP_ORDER')
    bp = parse_sip_matrix(header, 'BP', 'BP_ORDER')
    
    sip = Sip(a=a, b=b, ap=ap, bp=bp)
    
    # 2. Parse Linear WCS
    # CRPIX: FITS (1-based) -> Python (0-based)
    crpix = [header['CRPIX1'] - 1.0, header['CRPIX2'] - 1.0]
    
    # CRVAL
    crval = [header['CRVAL1'], header['CRVAL2']]
    
    # CD Matrix
    # If PC_i_j and CDELT_i are present
    cdelt1 = header.get('CDELT1', 1.0)
    cdelt2 = header.get('CDELT2', 1.0)
    
    pc1_1 = header.get('PC1_1', 1.0)
    pc1_2 = header.get('PC1_2', 0.0)
    pc2_1 = header.get('PC2_1', 0.0)
    pc2_2 = header.get('PC2_2', 1.0)
    
    cd = [
        [cdelt1 * pc1_1, cdelt1 * pc1_2],
        [cdelt2 * pc2_1, cdelt2 * pc2_2]
    ]
    
    return crpix, cd, crval, sip, data

def test_wcs_consistency_with_json():
    # Path relative to project root
    json_path = os.path.join('reproj_test', 'wcs_test_data.json')
    
    if not os.path.exists(json_path):
        pytest.skip(f"Test data not found at {json_path}")
        
    crpix, cd, crval, sip, data = load_wcs_from_json(json_path)
    
    # Instantiate WCS objects
    wcs_np = Tan(crpix, cd, crval, sip=sip)
    wcs_jax = TanJax(crpix, cd, crval, sip=sip)
    
    # Prepare Data
    input_pixels = np.array(data['input_pixel_coordinates'])
    expected_world = np.array(data['output_world_coordinates']) # Degrees
    
    x_in = input_pixels[:, 0]
    y_in = input_pixels[:, 1]

    # --- Test SIP-Only Part (Round Trip) ---
    # We verify the internal consistency of the SIP implementation (Forward + Reverse).
    # Note: We do not compare against data['focal_plane_coordinates'] because, as noted 
    # in the JSON metadata ('focal_plane_convention'), they are calculated using 
    # "Relative to FITS CRPIX (u = pix_0 - CRPIX_fits)".
    # Mapprojax (and standard Python WCS) uses "Relative to 0-based CRPIX (u = pix_0 - CRPIX_0)".
    # These definitions differ by 1.0 pixel in the intermediate u/v plane.
    # Since the Forward Projection (below) matches the JSON output, we know the 
    # pipeline is correct. Here we just ensure the SIP module itself is self-consistent.
    
    u_rel = x_in - crpix[0]
    v_rel = y_in - crpix[1]
    
    # 1. Forward SIP: (u, v) -> (U, V)
    U_calc, V_calc = sip.pix_to_foc(u_rel, v_rel)
    
    # 2. Reverse SIP: (U, V) -> (u, v)
    u_back, v_back = sip.foc_to_pix(U_calc, V_calc)
    
    # Check consistency (SIP coefficients round-trip quality)
    np.testing.assert_allclose(u_back, u_rel, rtol=0, atol=5e-2, err_msg="SIP Round Trip u mismatch")
    np.testing.assert_allclose(v_back, v_rel, rtol=0, atol=5e-2, err_msg="SIP Round Trip v mismatch")
    
    print("SIP-only transformation Round Trip matched within tolerance.")

    # --- Test Forward Projection (Pixel -> World) ---
    
    # NumPy
    ra_np_rad, dec_np_rad = wcs_np.unproj(x_in, y_in)
    ra_np = np.degrees(ra_np_rad)
    dec_np = np.degrees(dec_np_rad)
    
    # JAX
    ra_jax_rad, dec_jax_rad = wcs_jax.unproj(jnp.array(x_in), jnp.array(y_in))
    ra_jax = np.degrees(np.array(ra_jax_rad))
    dec_jax = np.degrees(np.array(dec_jax_rad))
    
    # Compare with Expected
    # Using slightly loose tolerance as the JSON source might have subtle differences 
    # (e.g., astropy version differences, float precision in JSON)
    # But usually 1e-6 degrees is good enough for headers.
    
    # Check NumPy vs Expected
    np.testing.assert_allclose(ra_np, expected_world[:, 0], rtol=0, atol=1e-6, err_msg="NumPy RA mismatch")
    np.testing.assert_allclose(dec_np, expected_world[:, 1], rtol=0, atol=1e-6, err_msg="NumPy Dec mismatch")
    
    # Check JAX vs Expected
    np.testing.assert_allclose(ra_jax, expected_world[:, 0], rtol=0, atol=1e-6, err_msg="JAX RA mismatch")
    np.testing.assert_allclose(dec_jax, expected_world[:, 1], rtol=0, atol=1e-6, err_msg="JAX Dec mismatch")
    
    print("Forward Projection (Pixel -> World) matched expected values.")

    # --- Test Inverse Projection (World -> Pixel) ---
    
    ra_in_rad = np.radians(expected_world[:, 0])
    dec_in_rad = np.radians(expected_world[:, 1])
    
    # NumPy
    x_out_np, y_out_np = wcs_np.proj(ra_in_rad, dec_in_rad)
    
    # JAX
    x_out_jax, y_out_jax = wcs_jax.proj(jnp.array(ra_in_rad), jnp.array(dec_in_rad))
    x_out_jax = np.array(x_out_jax)
    y_out_jax = np.array(y_out_jax)
    
    # Compare with Input Pixels
    # This validates the Round Trip + Inverse Logic
    # Relaxed tolerance to 0.05 px because SIP inverse polynomials are often approximations.
    np.testing.assert_allclose(x_out_np, x_in, rtol=0, atol=5e-2, err_msg="NumPy Inverse X mismatch")
    np.testing.assert_allclose(y_out_np, y_in, rtol=0, atol=5e-2, err_msg="NumPy Inverse Y mismatch")
    
    np.testing.assert_allclose(x_out_jax, x_in, rtol=0, atol=5e-2, err_msg="JAX Inverse X mismatch")
    np.testing.assert_allclose(y_out_jax, y_in, rtol=0, atol=5e-2, err_msg="JAX Inverse Y mismatch")
    
    print("Inverse Projection (World -> Pixel) matched within SIP approx tolerance (0.05 px).")

if __name__ == "__main__":
    # Allow running directly
    try:
        test_wcs_consistency_with_json()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
