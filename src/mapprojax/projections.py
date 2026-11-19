"""Backend-agnostic zenithal projection mixins: TAN, SIN, ZEA, STG and ARC.

The public precision-explicit classes live in :mod:`mapprojax.wcs`. These
mixins remain as implementation pieces shared by those classes.

Conventions. The native frame puts the reference point at the unit vector
``(1, 0, 0)`` with native north up, so the angular distance ``rho`` from the
reference has ``cos(rho) = x`` and ``sin(rho) = s = sqrt(y**2 + z**2)``. Every
projection here keeps the direction ``(y, z) / s`` and only changes the radial
function ``R(rho)`` of Calabretta & Greisen (2002). Plane coordinates are in
radians.

Every method returns finite values even where its validity flag is False,
and keeps every *unselected* branch finite too (the "double ``where``"): an
outer ``where`` alone does not stop an infinite or NaN derivative in the
discarded branch from poisoning the pullback (``0 * inf = nan``). Constants
that enter a ``where`` are built with ``*_like`` so that an f32 trace never
picks up a float64 literal.
"""

import numpy as np


def _is_f32(value):
    return np.dtype(value.dtype) == np.float32


def _series(t, coefficients):
    """``c0 + c1*t + c2*t**2 + ...`` by Horner's rule."""
    result = coefficients[-1]
    for c in reversed(coefficients[:-1]):
        result = c + t * result
    return result


#: ARC switches to a series below these squared radii. With terms through
#: sixth order in the radius, the truncation error there is ~1e-9 relative in
#: float32 and ~1e-17 in float64, below each type's rounding.
_ARC_SERIES_LIMIT = {True: 1e-2, False: 1e-4}  # keyed by "is float32"

#: atan(u)/u, sin(R)/R and cos(R), as series in u**2 or R**2.
_ATAN_OVER_U = (1.0, -1.0 / 3.0, 1.0 / 5.0, -1.0 / 7.0)
_SIN_OVER_R = (1.0, -1.0 / 6.0, 1.0 / 120.0, -1.0 / 5040.0)
_COS = (1.0, -1.0 / 2.0, 1.0 / 24.0, -1.0 / 720.0, 1.0 / 40320.0)


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


class _BeyondHorizonMixin:
    """Shared forward pieces for projections defined up to the antipode.

    ``valid`` excludes only the exact antipode ``(-1, 0, 0)``, where the
    native longitude -- and so the direction ``(y, z) / s`` -- is undefined.
    It is tested on ``s**2`` rather than ``x > -1``: in float32 ``x`` rounds to
    -1 within about 0.014 degrees of the antipode, which would wrongly mask
    that whole disc.
    """

    def _antipode_split(self, x, y, z):
        """``(s2, one_plus_x, valid)``; ``one_plus_x`` is finite and > 0
        everywhere (1 on the invalid antipode)."""
        xp = self.xp
        s2 = y**2 + z**2
        tiny = xp.finfo(xp.asarray(x).dtype).tiny
        valid = (x > 0) | (s2 > tiny)
        # 1 + x cancels for x near -1; there it equals s**2 / (1 - x), whose
        # denominator is >= 1 on that branch. Both branches stay finite.
        back = x < 0
        denom = xp.where(back, 1 - x, xp.ones_like(x))
        one_plus_x = xp.where(back, s2 / denom, 1 + x)
        one_plus_x = xp.where(valid, one_plus_x, xp.ones_like(x))
        return s2, one_plus_x, valid


class ZeaMixin(_BeyondHorizonMixin):
    """Zenithal equal-area projection, ``R = 2 sin(rho / 2)``."""

    def _native_to_plane(self, x, y, z):
        _, one_plus_x, valid = self._antipode_split(x, y, z)
        k = self.xp.sqrt(2 / one_plus_x)
        return y * k, z * k, valid

    def _plane_to_native(self, X, Y):
        xp = self.xp
        r2 = X**2 + Y**2
        valid = r2 <= 4
        r2 = xp.where(valid, r2, xp.zeros_like(r2))
        # r2 / 4 is exact, so 1 - r2 / 4 >= 0 on the valid branch. The limb
        # R = 2 is valid with an infinite inverse derivative, as SIN's is.
        k = xp.sqrt(1 - r2 / 4)
        return 1 - r2 / 2, X * k, Y * k, valid


class StgMixin(_BeyondHorizonMixin):
    """Stereographic projection, ``R = 2 tan(rho / 2)``."""

    def _native_to_plane(self, x, y, z):
        _, one_plus_x, valid = self._antipode_split(x, y, z)
        k = 2 / one_plus_x
        return y * k, z * k, valid

    def _plane_to_native(self, X, Y):
        # Every finite plane point is valid. The k form stays finite when
        # r2 overflows (k -> 0, x -> -1), where (4 - r2) / (4 + r2) is inf/inf.
        r2 = X**2 + Y**2
        k = 1 / (1 + r2 / 4)
        return 2 * k - 1, X * k, Y * k, self.xp.ones_like(X, dtype=bool)


class ArcMixin(_BeyondHorizonMixin):
    """Zenithal equidistant projection, ``R = rho``.

    ``atan2(s, x) / s``, ``sin(R) / R`` and ``cos(sqrt(R**2))`` all have a
    removable singularity at the reference point, which is also where every
    invalid sample's stand-in lands. Each switches to a series there.
    """

    def _native_to_plane(self, x, y, z):
        xp = self.xp
        s2, _, valid = self._antipode_split(x, y, z)
        limit = _ARC_SERIES_LIMIT[_is_f32(xp.asarray(x))]
        # x > 0 keeps the series to the reference side; near the antipode
        # s2 is small too, but atan2(s, x) is close to pi there, not s / x.
        small = (x > 0) & (s2 <= limit)
        ones = xp.ones_like(x)

        x_pos = xp.where(small, x, ones)
        u2 = s2 / x_pos**2
        k_series = _series(u2, _ATAN_OVER_U) / x_pos

        s = xp.sqrt(xp.where(small | ~valid, ones, s2))
        k_exact = xp.arctan2(s, x) / s

        k = xp.where(small, k_series, k_exact)
        return y * k, z * k, valid

    def _plane_to_native(self, X, Y):
        xp = self.xp
        r2 = X**2 + Y**2
        valid = r2 <= np.pi**2
        r2 = xp.where(valid, r2, xp.zeros_like(r2))
        limit = _ARC_SERIES_LIMIT[_is_f32(xp.asarray(r2))]
        small = r2 <= limit

        cos_series = _series(r2, _COS)
        sinc_series = _series(r2, _SIN_OVER_R)

        r = xp.sqrt(xp.where(small, xp.ones_like(r2), r2))
        cos_exact = xp.cos(r)
        sinc_exact = xp.sin(r) / r

        x = xp.where(small, cos_series, cos_exact)
        k = xp.where(small, sinc_series, sinc_exact)
        return x, X * k, Y * k, valid
