"""Backend-agnostic TAN and SIN projection mixins.

The public precision-explicit classes live in :mod:`mapprojax.wcs`. These
mixins remain as implementation pieces shared by those classes.
"""


class TanMixin:
    """Gnomonic projection."""

    def _native_to_plane(self, x, y, z):
        valid = x > 0
        x_safe = self.xp.where(valid, x, self.xp.ones_like(x))
        return y / x_safe, z / x_safe, valid

    def _plane_to_native(self, X, Y):
        r = self.xp.sqrt(1 + X**2 + Y**2)
        return 1.0 / r, X / r, Y / r, self.xp.ones_like(X, dtype=bool)


class SinMixin:
    """Orthographic projection."""

    def _native_to_plane(self, x, y, z):
        return y, z, x >= 0

    def _plane_to_native(self, X, Y):
        r2 = X**2 + Y**2
        valid = r2 <= 1
        x = self.xp.sqrt(self.xp.where(valid, 1 - r2, self.xp.zeros_like(r2)))
        return x, X, Y, valid
