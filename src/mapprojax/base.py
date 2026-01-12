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
        
    def proj(self, ra, dec):
        """
        Forward projection: Celestial (radians) -> Pixel coordinates.
        """
        # 1. Spherical -> Cartesian (Celestial)
        xc, yc, zc = radec_to_xyz(ra, dec, xp=self.xp)
        
        # 2. Celestial -> Native (Rotation)
        # v_n = M * v_c
        xn, yn, zn = apply_rotation(self.r_matrix, xc, yc, zc)
        
        # 3. Native -> Plane (Projection specific)
        Xp, Yp = self._native_to_plane(xn, yn, zn)
        
        # 4. Plane -> Pixel (Linear)
        # (x_img - crpix) = CD_inv * (Xp, Yp) ?
        # Wait, implementation.md says:
        # (Xp, Yp) = CD * (x_img - crpix)
        # So:
        # (Xp, Yp) = CD1_1*(x-c1) + CD1_2*(y-c2)
        # We need inverse:
        # (x_img - crpix) = CD_inv * (Xp, Yp)
        
        try:
            cd_inv = self.xp.linalg.inv(self.cd)
        except self.xp.linalg.LinAlgError:
            raise ValueError("CD matrix is singular/non-invertible")
            
        # Linear algebra:
        # P = [Xp, Yp]
        # P = CD * (p - crpix)  => CD_inv * P = p - crpix => p = CD_inv * P + crpix
        
        # CD_inv is 2x2. Xp, Yp might be arrays.
        # x_diff = cd_inv[0,0]*Xp + cd_inv[0,1]*Yp
        # y_diff = cd_inv[1,0]*Xp + cd_inv[1,1]*Yp
        
        x_diff = cd_inv[0,0] * Xp + cd_inv[0,1] * Yp
        y_diff = cd_inv[1,0] * Xp + cd_inv[1,1] * Yp
        
        x_img = x_diff + self.crpix[0]
        y_img = y_diff + self.crpix[1]
        
        return x_img, y_img

    def unproj(self, x, y):
        """
        Inverse projection: Pixel coordinates -> Celestial (radians).
        """
        x = self.xp.asarray(x)
        y = self.xp.asarray(y)
        
        # 1. Pixel -> Plane (Linear)
        # (Xp, Yp) = CD * (x_img - crpix)
        x_diff = x - self.crpix[0]
        y_diff = y - self.crpix[1]
        
        Xp = self.cd[0,0] * x_diff + self.cd[0,1] * y_diff
        Yp = self.cd[1,0] * x_diff + self.cd[1,1] * y_diff
        
        # 2. Plane -> Native (Projection specific)
        xn, yn, zn = self._plane_to_native(Xp, Yp)
        
        # 3. Native -> Celestial (Inverse Rotation)
        # v_c = M.T * v_n
        xc, yc, zc = apply_rotation_transpose(self.r_matrix, xn, yn, zn)
        
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

    def proj(self, ra, dec):
        """
        Forward projection: Celestial (radians) -> Pixel coordinates.
        Supports broadcasting.
        """
        # Broadcasting strategy:
        # Input RA/Dec shape: S_in
        # WCS Array shape: S_wcs (from crvals)
        # We need to broadcast S_in and S_wcs against each other.
        # But wait, the standard numpy behavior only works if dimensions align on the right.
        # User Manual Scenario C: 
        #   WCS shape (3,) -> (3, 1)
        #   Points shape (5,) -> (1, 5)
        #   Result (3, 5)
        # This implies standard broadcasting rules apply to the *result* calculation.
        
        # r_matrices shape: (3, 3, S_wcs...)
        # input ra/dec -> xc, yc, zc shape S_in
        
        # We need to perform matrix multiplication: M * v
        # M: (3, 3, S_wcs...)
        # v: (3, S_in...) (conceptually vector is axis 0 or last?)
        
        # In apply_rotation:
        # xn = m00*x + m01*y + m02*z
        # m00 shape is S_wcs. x shape is S_in.
        # xn shape will be broadcast(S_wcs, S_in).
        # This works naturally with NumPy broadcasting!
        
        xc, yc, zc = radec_to_xyz(ra, dec, xp=self.xp)
        
        # apply_rotation expects matrix (3, 3, ...). 
        # It accesses matrix[0,0] which gives shape S_wcs.
        # It multiplies by x which has shape S_in.
        # Result shape: broadcast(S_wcs, S_in).
        xn, yn, zn = apply_rotation(self.r_matrices, xc, yc, zc)
        
        Xp, Yp = self._native_to_plane(xn, yn, zn)
        
        # Linear transform part
        # CD is (2,2) constant for all WCS.
        # Xp, Yp shape is broadcast(S_wcs, S_in).
        try:
            cd_inv = self.xp.linalg.inv(self.cd)
        except self.xp.linalg.LinAlgError:
            raise ValueError("CD matrix is singular")
            
        x_diff = cd_inv[0,0] * Xp + cd_inv[0,1] * Yp
        y_diff = cd_inv[1,0] * Xp + cd_inv[1,1] * Yp
        
        x_img = x_diff + self.crpix[0]
        y_img = y_diff + self.crpix[1]
        
        return x_img, y_img

    def unproj(self, x, y):
        """
        Inverse projection: Pixel coordinates -> Celestial (radians).
        Supports broadcasting.
        """
        x = self.xp.asarray(x)
        y = self.xp.asarray(y)
        
        x_diff = x - self.crpix[0]
        y_diff = y - self.crpix[1]
        
        Xp = self.cd[0,0] * x_diff + self.cd[0,1] * y_diff
        Yp = self.cd[1,0] * x_diff + self.cd[1,1] * y_diff
        
        xn, yn, zn = self._plane_to_native(Xp, Yp)
        
        xc, yc, zc = apply_rotation_transpose(self.r_matrices, xn, yn, zn)
        
        ra_rad, dec_rad = xyz_to_radec(xc, yc, zc, xp=self.xp)
        return ra_rad, dec_rad

    def to_dict(self):
        return {
            'crpix': self.crpix.tolist(),
            'cd': self.cd.tolist(),
            'crvals': [self.xp.asarray(c).tolist() for c in self.crvals],
            'type': self.__class__.__name__
        }
