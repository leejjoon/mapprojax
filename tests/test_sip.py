import numpy as np
import pytest
from mapprojax.projections import Tan
from mapprojax.sip import Sip

try:
    from astropy.wcs import WCS as AstroWCS
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

@pytest.mark.skipif(not HAS_ASTROPY, reason="Astropy not installed")
def test_sip_against_astropy():
    """
    Verify SIP implementation against astropy.wcs.
    """
    # 1. Create a Header with SIP coefficients
    # Using a simple example.
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 100
    header['NAXIS2'] = 100
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    header['CRPIX1'] = 50.0
    header['CRPIX2'] = 50.0
    header['CRVAL1'] = 180.0
    header['CRVAL2'] = 45.0
    header['CD1_1'] = -0.00028 # ~1 arcsec/pix
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = 0.00028
    
    # Forward distortion (Pix -> Foc)
    # u' = u + f(u,v), v' = v + g(u,v)
    # f(u,v) = A_2_0 * u^2 + A_0_2 * v^2
    header['A_ORDER'] = 2
    header['A_2_0'] = 1.5e-6
    header['A_0_2'] = 2.5e-6
    
    # g(u,v) = B_1_1 * u*v
    header['B_ORDER'] = 2
    header['B_1_1'] = 3.0e-6
    
    # Reverse distortion (Foc -> Pix)
    # u = U + F(U,V), v = V + G(U,V)
    # For simplicity, use arbitrary values, though in reality they should be inverses.
    # We test that our implementation matches Astropy's calculation for these coefficients.
    header['AP_ORDER'] = 2
    header['AP_2_0'] = -1.5e-6
    header['AP_0_2'] = -2.5e-6
    
    header['BP_ORDER'] = 2
    header['BP_1_1'] = -3.0e-6
    
    # 2. Create Astropy WCS
    awcs = AstroWCS(header)
    
    # 3. Create Mapprojax WCS
    # Extract SIP coeffs
    a = np.zeros((3, 3))
    a[2, 0] = header['A_2_0']
    a[0, 2] = header['A_0_2']
    
    b = np.zeros((3, 3))
    b[1, 1] = header['B_1_1']
    
    ap = np.zeros((3, 3))
    ap[2, 0] = header['AP_2_0']
    ap[0, 2] = header['AP_0_2']
    
    bp = np.zeros((3, 3))
    bp[1, 1] = header['BP_1_1']
    
    sip = Sip(a=a, b=b, ap=ap, bp=bp)
    
    cd = [[header['CD1_1'], header['CD1_2']], 
          [header['CD2_1'], header['CD2_2']]]
    crpix = [header['CRPIX1'] - 1.0, header['CRPIX2'] - 1.0]
    crval = [header['CRVAL1'], header['CRVAL2']]
    
    mwcs = Tan(crpix, cd, crval, sip=sip)
    
    # 4. Generate test points (pixels)
    # Avoid 0,0 to avoid singularities if any
    x = np.array([10.0, 50.0, 90.0, 50.0])
    y = np.array([10.0, 50.0, 90.0, 10.0])
    
    # 5. Test Pix -> Sky (Forward)
    # astropy: all_pix2world (1-based origin if using default, but we can pass 0)
    # Wait, astropy pix2world takes 0-based or 1-based depending on arg.
    # By default, wcs.all_pix2world(x, y, 0) uses 0-based.
    
    ra_astro, dec_astro = awcs.all_pix2world(x, y, 0)
    
    # mapprojax: proj expects radians input? No, proj expects RA/DEC in radians.
    # We want pix -> sky.
    # mapprojax.unproj(x, y) -> returns ra, dec in radians.
    
    ra_map_rad, dec_map_rad = mwcs.unproj(x, y)
    ra_map = np.degrees(ra_map_rad)
    dec_map = np.degrees(dec_map_rad)
    
    np.testing.assert_allclose(ra_map, ra_astro, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(dec_map, dec_astro, rtol=1e-10, atol=1e-8)
    
    # 6. Test Sky -> Pix (Inverse)
    # Using the sky coordinates we just got
    
    x_astro, y_astro = awcs.all_world2pix(ra_astro, dec_astro, 0)
    
    # mapprojax: proj(ra, dec) -> returns x, y
    x_map, y_map = mwcs.proj(np.radians(ra_astro), np.radians(dec_astro))
    
    np.testing.assert_allclose(x_map, x_astro, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_map, y_astro, rtol=1e-5, atol=1e-5)

def test_sip_manual_check():
    """
    Test SIP logic manually without Astropy to ensure basic correctness of polynomials.
    """
    # Simple case:
    # u' = u + 0.1 * u^2
    # v' = v
    a = np.zeros((3, 3))
    a[2, 0] = 0.1
    sip = Sip(a=a)
    
    # Pixel (10, 0) relative to CRPIX (0,0)
    # u = 10, v = 0
    # u' = 10 + 0.1 * 100 = 20
    # v' = 0
    
    res_u, res_v = sip.pix_to_foc(np.array([10.0]), np.array([0.0]))
    assert res_u[0] == pytest.approx(20.0)
    assert res_v[0] == pytest.approx(0.0)

def test_no_sip():
    """Test that sip=None behaves as standard WCS."""
    crpix = [50.0, 50.0]
    cd = [[-0.001, 0.0], [0.0, 0.001]] 
    crval = [180.0, 0.0]
    
    wcs = Tan(crpix, cd, crval, sip=None)
    
    # Just check it runs and produces expected linear result
    x = 60.0 # u = 10
    y = 50.0 # v = 0
    # u = 10 -> X = -0.01 deg
    
    ra_rad, dec_rad = wcs.unproj(x, y)
    ra = np.degrees(ra_rad)
    
    # At equator, RA should increase to left.
    # X = -0.01 deg -> RA = 180 - 0.01 = 179.99
    assert ra == pytest.approx(179.99)

