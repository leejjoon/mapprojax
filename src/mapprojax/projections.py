"""Projection mixins.

Each mixin implements ``_native_to_plane`` and ``_plane_to_native`` against
``self.xp`` and returns a validity flag alongside the coordinates:

- ``_native_to_plane(xn, yn, zn) -> (X, Y, valid)``
- ``_plane_to_native(X, Y) -> (xn, yn, zn, valid)``

The returned coordinates are **finite everywhere**, including outside the
projection's domain: invalid regions are substituted with a safe value so the
CD multiply and the SIP evaluation downstream never see a NaN. That matters for
gradients -- a NaN that reaches an arithmetic step poisons its pullback
(``0 * NaN = NaN``) even when the caller masks the invalid sample out of the
loss. ``WCSBase`` combines the flags and restores NaN at the public outputs
only.
"""

from .base import WCSBase, WCSArrayBase


class TanMixin:
    """Gnomonic projection.

    Valid where the native vector points into the forward hemisphere,
    ``xn > 0``; at ``xn == 0`` the tangent plane is at infinity.
    """

    def _native_to_plane(self, x, y, z):
        valid = x > 0
        # Substituting before the division keeps X, Y finite for xn <= 0.
        x_safe = self.xp.where(valid, x, self.xp.ones_like(x))
        return y / x_safe, z / x_safe, valid

    def _plane_to_native(self, X, Y):
        # Every finite (X, Y) maps to a unit vector on the forward hemisphere.
        r = self.xp.sqrt(1 + X**2 + Y**2)
        return 1.0 / r, X / r, Y / r, self.xp.ones_like(X, dtype=bool)


class SinMixin:
    """Orthographic projection.

    Valid where ``xn >= 0`` (the back hemisphere folds onto the front) and,
    for the inverse, where ``X**2 + Y**2 <= 1``.
    """

    def _native_to_plane(self, x, y, z):
        return y, z, x >= 0

    def _plane_to_native(self, X, Y):
        r2 = X**2 + Y**2
        valid = r2 <= 1
        # clamp before the sqrt so the invalid region is 0, not NaN
        x = self.xp.sqrt(self.xp.where(valid, 1 - r2, self.xp.zeros_like(r2)))
        return x, X, Y, valid


# NumPy-backed concrete classes. The mixins are backend-agnostic; the JAX
# variants in ``jax_projections.py`` pair the same mixins with the JAX bases.
class Tan(TanMixin, WCSBase):
    pass


class TanArray(TanMixin, WCSArrayBase):
    pass


class Sin(SinMixin, WCSBase):
    pass


class SinArray(SinMixin, WCSArrayBase):
    pass
