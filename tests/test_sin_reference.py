import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mapprojax import Sin

class TestSinReference(unittest.TestCase):
    def test_sin_reference(self):
        """Test SIN projection using parameters from Rust implementation."""
        # Define constants from sin.rs
        crpix1 = 382.00001513958
        crpix2 = 389.500015437603

        crval1 = 183.914583333
        crval2 = 36.3275

        cd11 = -2.7777777349544e-4
        cd22 = 2.77777773495436e-4

        # Prepare parameters for Sin class
        # crpix expected as [x, y]
        crpix = [crpix1, crpix2]

        # cd expected as matrix [[cd11, cd12], [cd21, cd22]]
        # The Rust code uses WcsImgXY2ProjXY::from_cd(crpix1, crpix2, cd11, 0.0, 0.0, cd22)
        # So off-diagonal elements are 0.0
        cd = [[cd11, 0.0], [0.0, cd22]]

        # crval expected as [ra, dec]
        crval = [crval1, crval2]

        # Create Sin projection object
        wcs = Sin(crpix, cd, crval)

        # Test 1: Projecting the center (CRVAL) should map to CRPIX
        # In the Rust test:
        # let img_coo_input = ImgXY::new(382.00001513958, 389.500015437603);
        # let lonlat = img2lonlat.img2lonlat(&img_coo_input).unwrap();
        # assert!((lonlat.lon() - proj_center.lon()).abs() < 1e-14);
        # assert!((lonlat.lat() - proj_center.lat()).abs() < 1e-14);

        # NOTE: wcs.unproj converts pixel to sky (img2lonlat)
        # wcs.proj converts sky to pixel (lonlat2img)

        # The Rust test checks that unprojecting CRPIX gives CRVAL (proj_center)
        ra_res, dec_res = wcs.unproj(crpix1, crpix2)

        self.assertAlmostEqual(ra_res, crval1, delta=1e-10)
        self.assertAlmostEqual(dec_res, crval2, delta=1e-10)

        # Test 2: Projecting CRVAL should map back to CRPIX
        # let img_coo_input = img2lonlat.lonlat2img(&lonlat).unwrap();
        # assert!((img_coo_input.x() - img_coo_input.x()).abs() < 1e-14);
        # Wait, the rust test comparison looks weird: (img_coo_input.x() - img_coo_input.x()).abs() < 1e-14
        # That's always true! 0 < 1e-14.
        # But generally, projection of center should correspond to CRPIX.

        x_res, y_res = wcs.proj(crval1, crval2)

        self.assertAlmostEqual(x_res, crpix1, delta=1e-10)
        self.assertAlmostEqual(y_res, crpix2, delta=1e-10)

if __name__ == '__main__':
    unittest.main()
