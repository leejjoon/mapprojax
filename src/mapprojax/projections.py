import numpy as np
from .base import WCSBase, WCSArrayBase

class TanMixin:
    """Gnomonic Projection Logic"""
    def _native_to_plane(self, x, y, z):
        # x > 0 check is implicit, usually resulting in valid values on front hemisphere
        # If x <= 0, it diverges.
        # We assume valid inputs for simple implementation or let numpy warn
        return y / x, z / x

    def _plane_to_native(self, X, Y):
        r = np.sqrt(1 + X**2 + Y**2)
        # x = 1/r, y = X/r, z = Y/r
        return 1.0/r, X/r, Y/r

class SinMixin:
    """Orthographic Projection Logic"""
    def _native_to_plane(self, x, y, z):
        # x >= 0 check needed?
        return y, z

    def _plane_to_native(self, X, Y):
        r2 = X**2 + Y**2
        # If r2 > 1, undefined.
        # We can use np.nan for invalid pixels
        
        # Calculate x: sqrt(1 - r^2)
        # Use np.sqrt handling negative values -> nan
        
        # To avoid RuntimeWarning for invalid pixels, we might mask, but raw numpy is requested
        x = np.sqrt(1 - r2) 
        # y = X, z = Y
        return x, X, Y


class Tan(TanMixin, WCSBase):
    pass

class TanArray(TanMixin, WCSArrayBase):
    pass

class Sin(SinMixin, WCSBase):
    pass

class SinArray(SinMixin, WCSArrayBase):
    pass
