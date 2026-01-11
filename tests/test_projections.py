import numpy as np
import pytest
from mapprojax import Tan, Sin, TanArray, SinArray

def test_tan_roundtrip():
    """Test basic TAN projection round-trip consistency."""
    crpix = [100.0, 100.0]
    # CD in radians/pixel (approx 0.057 deg/pixel if value is 0.001)
    cd = [[-0.001, 0.0], [0.0, 0.001]] 
    crval = [45.0, 30.0] # Degrees
    
    wcs = Tan(crpix, cd, crval)
    
    # Test Reference pixel
    x, y = wcs.proj(np.radians(45.0), np.radians(30.0))
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(100.0)
    
    # Test Inverse at reference
    ra, dec = wcs.unproj(100.0, 100.0)
    assert ra == pytest.approx(np.radians(45.0))
    assert dec == pytest.approx(np.radians(30.0))
    
    # Test off-center point
    # Move 10 pixels in X
    x_target = 110.0
    y_target = 100.0
    
    ra, dec = wcs.unproj(x_target, y_target)
    x_res, y_res = wcs.proj(ra, dec)
    
    assert x_res == pytest.approx(x_target)
    assert y_res == pytest.approx(y_target)

def test_sin_roundtrip():
    """Test basic SIN (Orthographic) projection round-trip."""
    crpix = [50.0, 50.0]
    cd = [[0.01, 0.0], [0.0, 0.01]]
    crval = [0.0, 0.0]
    
    wcs = Sin(crpix, cd, crval)
    
    x_target, y_target = 60.0, 40.0
    ra, dec = wcs.unproj(x_target, y_target)
    x_res, y_res = wcs.proj(ra, dec)
    
    assert x_res == pytest.approx(x_target)
    assert y_res == pytest.approx(y_target)
    
def test_tan_array_broadcasting():
    """Test TanArray broadcasting with multiple CRVALs."""
    crpix = [100.0, 100.0]
    cd = [[-0.001, 0.0], [0.0, 0.001]]
    
    # 3 WCS instances
    ra_centers = [45.0, 50.0, 55.0]
    dec_centers = [30.0, 30.0, 30.0]
    
    wcs_arr = TanArray(crpix, cd, (ra_centers, dec_centers))
    
    # 1. Project center points (should all be 100, 100)
    x, y = wcs_arr.proj(np.radians(ra_centers), np.radians(dec_centers))
    np.testing.assert_allclose(x, 100.0)
    np.testing.assert_allclose(y, 100.0)
    
    # 2. Broadcast single point (45, 30) against 3 WCSs
    # Point matches 1st WCS center
    x_bc, y_bc = wcs_arr.proj(np.radians(45.0), np.radians(30.0))
    assert x_bc.shape == (3,)
    assert x_bc[0] == pytest.approx(100.0)
    assert x_bc[1] != pytest.approx(100.0)
    
    # 3. Broadcast multiple points against multiple WCSs
    # Points: (N=2)
    pts_ra = np.array([45.0, 55.0]) 
    pts_dec = np.array([30.0, 30.0])
    
    # Standard broadcasting (3,) and (2,) -> (2, 3) or (3, 2)?
    # Actually numpy broadcasting rules: (3,) and (2,) are incompatible unless one is (3,1) or (1,2).
    # We need to reshape inputs if we want cartesian product.
    
    # Case A: pts shape (3,) matching WCS shape (3,)
    pts_ra_match = np.array([45.0, 50.0, 55.0])
    pts_dec_match = np.array([30.0, 30.0, 30.0])
    x_match, y_match = wcs_arr.proj(np.radians(pts_ra_match), np.radians(pts_dec_match))
    np.testing.assert_allclose(x_match, 100.0)
    
    # Case B: 2 points vs 3 WCSs -> explicit reshape
    # WCS is (3,). Points (2, 1). Result (2, 3).
    pts_ra_col = pts_ra.reshape(2, 1)
    pts_dec_col = pts_dec.reshape(2, 1)
    
    x_cart, y_cart = wcs_arr.proj(np.radians(pts_ra_col), np.radians(pts_dec_col))
    assert x_cart.shape == (2, 3)
    
    # (0, 0): Point 0 (45) vs WCS 0 (45) -> 100
    assert x_cart[0, 0] == pytest.approx(100.0)
    # (1, 2): Point 1 (55) vs WCS 2 (55) -> 100
    assert x_cart[1, 2] == pytest.approx(100.0)

def test_sin_array():
    """Test SinArray basic functionality."""
    crpix = [50.0, 50.0]
    cd = [[0.1, 0.0], [0.0, 0.1]]
    crvals = ([0.0, 10.0], [0.0, 0.0]) # 2 centers
    
    wcs_arr = SinArray(crpix, cd, crvals)
    
    # Proj center
    x, y = wcs_arr.proj(np.radians([0.0, 10.0]), np.radians([0.0, 0.0]))
    np.testing.assert_allclose(x, 50.0)
    np.testing.assert_allclose(y, 50.0)

def test_singular_cd_error():
    """Test that singular CD matrix raises ValueError."""
    wcs = Tan([0,0], [[0,0],[0,0]], [0,0])
    with pytest.raises(ValueError):
        wcs.proj(0.0, 0.0)
