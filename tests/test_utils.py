import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mapprojax.utils import radec_to_xyz, xyz_to_radec, rotation_matrix, apply_rotation, apply_rotation_transpose

class TestUtils(unittest.TestCase):
    def test_radec_to_xyz_basic(self):
        # (0, 0) -> (1, 0, 0)
        x, y, z = radec_to_xyz(0.0, 0.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, 0.0)

        # (90, 0) -> (0, 1, 0)
        x, y, z = radec_to_xyz(90.0, 0.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertAlmostEqual(z, 0.0)
        
        # (0, 90) -> (0, 0, 1)
        x, y, z = radec_to_xyz(0.0, 90.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, 1.0)

    def test_xyz_to_radec_basic(self):
        # (1, 0, 0) -> (0, 0)
        ra, dec = xyz_to_radec(1.0, 0.0, 0.0)
        self.assertAlmostEqual(ra, 0.0)
        self.assertAlmostEqual(dec, 0.0)

        # (0, 1, 0) -> (90, 0)
        ra, dec = xyz_to_radec(0.0, 1.0, 0.0)
        self.assertAlmostEqual(ra, 90.0)
        self.assertAlmostEqual(dec, 0.0)
        
        # (0, 0, 1) -> (0, 90) - RA is arbitrary at pole, but usually 0 or derived from atan2(0,0)=0
        ra, dec = xyz_to_radec(0.0, 0.0, 1.0)
        self.assertAlmostEqual(dec, 90.0)
        
    def test_roundtrip_radec_xyz(self):
        ra_in = 123.456
        dec_in = -45.678
        x, y, z = radec_to_xyz(ra_in, dec_in)
        ra_out, dec_out = xyz_to_radec(x, y, z)
        
        self.assertAlmostEqual(ra_in, ra_out)
        self.assertAlmostEqual(dec_in, dec_out)

    def test_rotation_matrix_identity(self):
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
        self.assertAlmostEqual(xn, 1.0)
        self.assertAlmostEqual(yn, 0.0)
        self.assertAlmostEqual(zn, 0.0)

    def test_apply_rotation_transpose(self):
        # Transpose should invert the rotation (since it's orthogonal)
        lon, lat = 45.0, 30.0
        mat = rotation_matrix(lon, lat)
        
        xn, yn, zn = 1.0, 0.0, 0.0 # Native center
        
        # Un-rotate
        xc, yc, zc = apply_rotation_transpose(mat, xn, yn, zn)
        
        # Should be celestial center
        xc_exp, yc_exp, zc_exp = radec_to_xyz(lon, lat)
        
        self.assertAlmostEqual(xc, xc_exp)
        self.assertAlmostEqual(yc, yc_exp)
        self.assertAlmostEqual(zc, zc_exp)

    def test_broadcasting(self):
        # Test array inputs
        ra = np.array([0.0, 90.0, 180.0])
        dec = np.array([0.0, 0.0, 0.0])
        
        x, y, z = radec_to_xyz(ra, dec)
        self.assertEqual(x.shape, (3,))
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[1], 0.0)
        self.assertAlmostEqual(x[2], -1.0)

if __name__ == '__main__':
    unittest.main()
