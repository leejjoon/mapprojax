from abc import ABC, abstractmethod
import numpy as np
from .utils import radec_to_xyz, xyz_to_radec, rotation_matrix, apply_rotation, apply_rotation_transpose

class WCSBase(ABC):
    xp = np

    def __init__(self, crpix, cd, crval):
        self.crpix = self.xp.array(crpix, dtype=float)
        self.cd = self.xp.array(cd, dtype=float)
        self.crval = self.xp.array(crval, dtype=float)
        
        # Precompute rotation matrix
        # crval is (lon, lat)
        self.r_matrix = rotation_matrix(self.xp.radians(self.crval[0]), self.xp.radians(self.crval[1]), xp=self.xp)

    @abstractmethod
    def _native_to_plane(self, xn, yn, zn):
        """Map native unit vector (xn, yn, zn) to projection plane (X, Y)."""
        pass

    @abstractmethod
    def _plane_to_native(self, X, Y):
        """Map projection plane (X, Y) to native unit vector (xn, yn, zn)."""
        pass

    def proj_xyz(self, x, y, z):
        """
        Forward projection: Celestial unit vector -> Pixel coordinates.
        """
        # 1. Celestial -> Native (Rotation)
        # v_n = M * v_c
        xn, yn, zn = apply_rotation(self.r_matrix, x, y, z)
        
        # 2. Native -> Plane (Projection specific)
        Xp, Yp = self._native_to_plane(xn, yn, zn)
        
        # 3. Plane -> Pixel (Linear)
        # Xp, Yp from projections are in radians (dimensionless ratios).
        # FITS CD matrix is usually in degrees.
        Xp_deg = self.xp.degrees(Xp)
        Yp_deg = self.xp.degrees(Yp)
        
        try:
            cd_inv = self.xp.linalg.inv(self.cd)
        except self.xp.linalg.LinAlgError:
            raise ValueError("CD matrix is singular/non-invertible")
            
        x_diff = cd_inv[0,0] * Xp_deg + cd_inv[0,1] * Yp_deg
        y_diff = cd_inv[1,0] * Xp_deg + cd_inv[1,1] * Yp_deg
        
        x_img = x_diff + self.crpix[0]
        y_img = y_diff + self.crpix[1]
        
        return x_img, y_img
        
    def proj(self, ra, dec):
        """
        Forward projection: Celestial (radians) -> Pixel coordinates.
        """
        # 1. Spherical -> Cartesian (Celestial)
        xc, yc, zc = radec_to_xyz(ra, dec, xp=self.xp)
        
        return self.proj_xyz(xc, yc, zc)

    def unproj_xyz(self, x, y):
        """
        Inverse projection: Pixel coordinates -> Celestial unit vector.
        """
        x = self.xp.asarray(x)
        y = self.xp.asarray(y)
        
        # 1. Pixel -> Plane (Linear)
        # (Xp, Yp) = CD * (x_img - crpix)
        x_diff = x - self.crpix[0]
        y_diff = y - self.crpix[1]
        
        Xp_deg = self.cd[0,0] * x_diff + self.cd[0,1] * y_diff
        Yp_deg = self.cd[1,0] * x_diff + self.cd[1,1] * y_diff
        
        # 2. Plane -> Native (Projection specific)
        # Convert to radians for projection logic
        Xp_rad = self.xp.radians(Xp_deg)
        Yp_rad = self.xp.radians(Yp_deg)
        
        xn, yn, zn = self._plane_to_native(Xp_rad, Yp_rad)
        
        # 3. Native -> Celestial (Inverse Rotation)
        # v_c = M.T * v_n
        xc, yc, zc = apply_rotation_transpose(self.r_matrix, xn, yn, zn)
        
        return xc, yc, zc

    def unproj(self, x, y):
        """
        Inverse projection: Pixel coordinates -> Celestial (radians).
        """
        xc, yc, zc = self.unproj_xyz(x, y)
        
        # 4. Cartesian -> Spherical
        ra_rad, dec_rad = xyz_to_radec(xc, yc, zc, xp=self.xp)
        return ra_rad, dec_rad

    def to_dict(self):
        """Basic serialization helper."""
        return {
            'crpix': self.crpix.tolist(),
            'cd': self.cd.tolist(),
            'crval': self.crval.tolist(),
            'type': self.__class__.__name__
        }
    
    @classmethod
    def from_dict(cls, data):
        # This needs to be handled by the specific subclass or a factory
        # For now, we assume calling on correct class
        return cls(data['crpix'], data['cd'], data['crval'])
    
    # ASDF and Custom Binary hooks can be added here or in subclasses


class WCSArrayBase(ABC):
    xp = np

    def __init__(self, crpix, cd, crvals):
        self.crpix = self.xp.array(crpix, dtype=float)
        self.cd = self.xp.array(cd, dtype=float)
        
        # crvals is expected to be (ra_array, dec_array) or similar
        # shape: (2, N) or tuple of arrays
        ra_arr, dec_arr = self.xp.asarray(crvals[0]), self.xp.asarray(crvals[1])
        if ra_arr.shape != dec_arr.shape:
             raise ValueError("RA and Dec arrays in crvals must have same shape")
             
        self.crvals = (ra_arr, dec_arr)
        
        # Precompute rotation matrices for all crvals
        # rotation_matrix util returns (3, 3, ...) if input is array?
        # Let's check our utils implementation.
        # In utils.py, we constructed it manually. 
        # If inputs are arrays shape S, the resulting 'mat' has shape (3, 3, S).
        self.r_matrices = rotation_matrix(self.xp.radians(ra_arr), self.xp.radians(dec_arr), xp=self.xp)
        # Shape is (3, 3) + crval_shape

    @abstractmethod
    def _native_to_plane(self, xn, yn, zn):
        pass

    @abstractmethod
    def _plane_to_native(self, X, Y):
        pass

    def proj_xyz(self, x, y, z):
        """
        Forward projection: Celestial unit vector -> Pixel coordinates.
        Supports broadcasting.
        """
        # apply_rotation expects matrix (3, 3, ...). 
        # It accesses matrix[0,0] which gives shape S_wcs.
        # It multiplies by x which has shape S_in.
        # Result shape: broadcast(S_wcs, S_in).
        xn, yn, zn = apply_rotation(self.r_matrices, x, y, z)
        
        Xp, Yp = self._native_to_plane(xn, yn, zn)
        
        # Linear transform part
        # CD is (2,2) constant for all WCS.
        # Convert Xp, Yp (radians) to degrees for CD matrix
        Xp_deg = self.xp.degrees(Xp)
        Yp_deg = self.xp.degrees(Yp)
        
        try:
            cd_inv = self.xp.linalg.inv(self.cd)
        except self.xp.linalg.LinAlgError:
            raise ValueError("CD matrix is singular")
            
        x_diff = cd_inv[0,0] * Xp_deg + cd_inv[0,1] * Yp_deg
        y_diff = cd_inv[1,0] * Xp_deg + cd_inv[1,1] * Yp_deg
        
        x_img = x_diff + self.crpix[0]
        y_img = y_diff + self.crpix[1]
        
        return x_img, y_img

    def proj(self, ra, dec):
        """
        Forward projection: Celestial (radians) -> Pixel coordinates.
        Supports broadcasting.
        """
        xc, yc, zc = radec_to_xyz(ra, dec, xp=self.xp)
        return self.proj_xyz(xc, yc, zc)

    def unproj_xyz(self, x, y):
        """
        Inverse projection: Pixel coordinates -> Celestial unit vector.
        Supports broadcasting.
        """
        x = self.xp.asarray(x)
        y = self.xp.asarray(y)
        
        x_diff = x - self.crpix[0]
        y_diff = y - self.crpix[1]
        
        Xp_deg = self.cd[0,0] * x_diff + self.cd[0,1] * y_diff
        Yp_deg = self.cd[1,0] * x_diff + self.cd[1,1] * y_diff
        
        # Convert to radians for projection logic
        Xp_rad = self.xp.radians(Xp_deg)
        Yp_rad = self.xp.radians(Yp_deg)
        
        xn, yn, zn = self._plane_to_native(Xp_rad, Yp_rad)
        
        xc, yc, zc = apply_rotation_transpose(self.r_matrices, xn, yn, zn)
        
        return xc, yc, zc

    def unproj(self, x, y):
        """
        Inverse projection: Pixel coordinates -> Celestial (radians).
        Supports broadcasting.
        """
        xc, yc, zc = self.unproj_xyz(x, y)
        
        ra_rad, dec_rad = xyz_to_radec(xc, yc, zc, xp=self.xp)
        return ra_rad, dec_rad

    def to_dict(self):
        return {
            'crpix': self.crpix.tolist(),
            'cd': self.cd.tolist(),
            'crvals': [self.xp.asarray(c).tolist() for c in self.crvals],
            'type': self.__class__.__name__
        }
