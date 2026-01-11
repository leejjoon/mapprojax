import numpy as np
import pytest
from mapprojax.utils import radec_to_xyz, xyz_to_radec, rotation_matrix, apply_rotation, apply_rotation_transpose

def test_radec_to_xyz_basic():
    # (0, 0) -> (1, 0, 0)
    x, y, z = radec_to_xyz(0.0, 0.0)
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.0)

    # (90, 0) -> (0, 1, 0)
    x, y, z = radec_to_xyz(90.0, 0.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(1.0)
    assert z == pytest.approx(0.0)
    
    # (0, 90) -> (0, 0, 1)
    x, y, z = radec_to_xyz(0.0, 90.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(1.0)

def test_xyz_to_radec_basic():
    # (1, 0, 0) -> (0, 0)
    ra, dec = xyz_to_radec(1.0, 0.0, 0.0)
    assert ra == pytest.approx(0.0)
    assert dec == pytest.approx(0.0)

    # (0, 1, 0) -> (90, 0)
    ra, dec = xyz_to_radec(0.0, 1.0, 0.0)
    assert ra == pytest.approx(90.0)
    assert dec == pytest.approx(0.0)
    
    # (0, 0, 1) -> (0, 90) - RA is arbitrary at pole, but usually 0 or derived from atan2(0,0)=0
    ra, dec = xyz_to_radec(0.0, 0.0, 1.0)
    assert dec == pytest.approx(90.0)
    
def test_roundtrip_radec_xyz():
    ra_in = 123.456
    dec_in = -45.678
    x, y, z = radec_to_xyz(ra_in, dec_in)
    ra_out, dec_out = xyz_to_radec(x, y, z)
    
    assert ra_in == pytest.approx(ra_out)
    assert dec_in == pytest.approx(dec_out)

def test_rotation_matrix_identity():
    # If crval is (0, 0), the rotation matrix should align celestial (1,0,0) to native (1,0,0)
    # However, our definition says:
    # M rotates celestial vector INTO native frame.
    # Native center is always (1, 0, 0).
    # Celestial center is crval (lon, lat).
    # So M * v(crval) should be (1, 0, 0).
    
    lon, lat = 45.0, 30.0
    mat = rotation_matrix(lon, lat)
    
    # Vector at center
    xc, yc, zc = radec_to_xyz(lon, lat)
    
    # Rotate
    xn, yn, zn = apply_rotation(mat, xc, yc, zc)
    
    # Should be (1, 0, 0)
    assert xn == pytest.approx(1.0)
    assert yn == pytest.approx(0.0)
    assert zn == pytest.approx(0.0)

def test_apply_rotation_transpose():
    # Transpose should invert the rotation (since it's orthogonal)
    lon, lat = 45.0, 30.0
    mat = rotation_matrix(lon, lat)
    
    xn, yn, zn = 1.0, 0.0, 0.0 # Native center
    
    # Un-rotate
    xc, yc, zc = apply_rotation_transpose(mat, xn, yn, zn)
    
    # Should be celestial center
    xc_exp, yc_exp, zc_exp = radec_to_xyz(lon, lat)
    
    assert xc == pytest.approx(xc_exp)
    assert yc == pytest.approx(yc_exp)
    assert zc == pytest.approx(zc_exp)

def test_broadcasting():
    # Test array inputs
    ra = np.array([0.0, 90.0, 180.0])
    dec = np.array([0.0, 0.0, 0.0])
    
    x, y, z = radec_to_xyz(ra, dec)
    assert x.shape == (3,)
    assert x[0] == pytest.approx(1.0)
    assert x[1] == pytest.approx(0.0)
    assert x[2] == pytest.approx(-1.0)