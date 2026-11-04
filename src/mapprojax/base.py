from abc import ABC, abstractmethod

import jax
import numpy as np
from .utils import (
    radec_to_xyz, xyz_to_radec, apply_rotation, apply_rotation_transpose,
    _check_sequence_components,
)


def _check_cd(cd):
    """Reject a singular or non-finite concrete CD at construction.

    Tracers cannot be inspected, so they are skipped here and handled by the
    output mask in ``native_to_pix``.
    """
    if _is_tracer(cd):
        return
    _, valid = _cd_inverse(cd, xp=np)
    if not bool(np.all(valid)):
        raise ValueError("CD matrix is singular or has non-finite entries")


def _is_tracer(value):
    """True for a JAX tracer (which cannot be inspected eagerly)."""
    return isinstance(value, jax.core.Tracer)


def _cd_is_valid(cd, xp=np):
    """Scale-invariant CD validity, without forming the inverse.

    ``pix_to_native`` multiplies by ``cd`` rather than inverting it, so it
    needs the same test but not the adjugate.
    """
    return _cd_inverse(cd, xp=xp)[1]


def _cd_finite(cd, xp=np):
    """``(cd_safe, finite)``: ``cd`` with any non-finite matrix replaced by
    the identity, and the flag saying which matrices were replaced.

    A selection, not arithmetic, so an infinite or NaN entry never reaches a
    multiply whose pullback would turn ``0 * inf`` into NaN.
    """
    cd = xp.asarray(cd)
    finite = xp.all(xp.isfinite(cd), axis=(-2, -1))
    eye = xp.broadcast_to(xp.eye(2, dtype=cd.dtype), cd.shape)
    return xp.where(finite[..., None, None], cd, eye), finite


def _cd_inverse(cd, xp=np):
    """Guarded closed-form 2x2 inverse of a CD matrix.

    Returns ``(cd_inv, valid)``. ``valid`` is a boolean (scalar or array
    shaped like the batch) that is False when the matrix is singular or any
    entry is non-finite; ``cd_inv`` is finite on that branch so downstream
    arithmetic never sees a NaN.

    The determinant is taken after dividing by the largest absolute entry, so
    the test is scale invariant: ``diag(1e-25, 1e-25)`` underflows an f32
    raw determinant to 0 and ``diag(1e25, 1e25)`` overflows it to inf,
    although both are perfectly invertible. A raw ``det == 0`` test would
    reject the first and, because ``d / det`` rounds to 0, silently zero the
    second.

    ``cd_inv`` is built as ``adjugate / (scaled_det * scale)``: dividing the
    adjugate by the scaled determinant first keeps the intermediate inside
    the dtype's range, so the scaling cancels instead of overflowing.
    """
    cd_safe, finite = _cd_finite(cd, xp)
    scale = xp.max(xp.abs(cd_safe), axis=(-2, -1))

    # where() rather than a division: a zero scale must not reach the
    # arithmetic that feeds the gradients.
    safe_scale = xp.where(scale > 0, scale, xp.ones_like(scale))
    scaled = cd_safe / safe_scale[..., None, None]
    det = scaled[..., 0, 0] * scaled[..., 1, 1] - scaled[..., 0, 1] * scaled[..., 1, 0]
    valid = finite & (det != 0) & xp.isfinite(det)

    safe_det = xp.where(valid, det, xp.ones_like(det))
    adjugate_scale = xp.where(valid, safe_scale, xp.ones_like(safe_scale))
    cd_inv = xp.stack([
        xp.stack([scaled[..., 1, 1], -scaled[..., 0, 1]], axis=-1),
        xp.stack([-scaled[..., 1, 0], scaled[..., 0, 0]], axis=-1),
    ], axis=-2) / safe_det[..., None, None] / adjugate_scale[..., None, None]
    return cd_inv, valid


def _host_array(value, name):
    """Transfer a concrete value to NumPy, rejecting traced host work."""
    if isinstance(value, jax.core.Tracer):
        raise TypeError(f"{name} is host-only and cannot be used with a JAX tracer")
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a concrete host value") from exc


def _host_crval(crval):
    _check_sequence_components(crval, "crval")
    crval = _host_array(crval, "crval")
    if crval.shape != (2,):
        raise ValueError(f"crval must have shape (2,), got {crval.shape}")
    if crval.dtype != np.dtype(np.float64):
        raise TypeError(f"crval must be float64, got {crval.dtype}")
    return crval


def _host_offset_scale(offset_scale):
    offset_scale = _host_array(offset_scale, "offset_scale")
    if offset_scale.shape != ():
        raise ValueError(
            f"offset_scale must be a scalar, got shape {offset_scale.shape}"
        )
    if offset_scale.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(
            "offset_scale must have float32 or float64 dtype, "
            f"got {offset_scale.dtype}"
        )
    return offset_scale


def _host_pair(values, name):
    try:
        first, second = values
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two components") from exc
    return first, second


def _host_offset_component(value, name, dtype):
    value = _host_array(value, name)
    if value.dtype != dtype:
        raise TypeError(
            f"{name} dtype {value.dtype} does not match offset_scale dtype {dtype}"
        )
    return value


def _host_sky_component(value, name):
    value = _host_array(value, name)
    if value.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric coordinates")
    return value.astype(np.float64, copy=False)


def absolute_from_offset(offset, *, crval, offset_scale):
    """Compose a producer-typed sky offset into absolute host coordinates.

    ``offset`` is the ``(Delta_lon * cos(lat0), Delta_lat)`` pair in radians
    returned by ``unproj_offset``. The keyword-only ``crval`` is a concrete
    float64 ``(lon, lat)`` pair in degrees, and ``offset_scale`` is the
    producer's ``offset_scale()``. Both offset components must have that
    scale's float dtype. Passing the scale is important near the
    poles: recomputing a float64 cosine would not undo the scaling actually
    used by a float32 producer. Python float offsets are therefore accepted
    only with a float64 scale.

    The result is a pair of NumPy float64 arrays, with longitude wrapped to
    ``[0, 2*pi)``. This host transfer is not JIT- or autodiff-compatible. At
    an exactly zero scale, longitude is undefined and returned as NaN rather
    than being fabricated; latitude remains available.
    """
    reference = _host_crval(crval)
    scale = _host_offset_scale(offset_scale)
    scale_dtype = scale.dtype
    dlon_coslat, dlat = _host_pair(offset, "offset")
    dlon_coslat = _host_offset_component(dlon_coslat, "longitude offset", scale_dtype)
    dlat = _host_offset_component(dlat, "latitude offset", scale_dtype)
    dlon_coslat, dlat = np.broadcast_arrays(dlon_coslat, dlat)

    valid = (
        np.isfinite(dlon_coslat)
        & np.isfinite(dlat)
        & np.isfinite(reference).all()
        & np.isfinite(scale)
    )
    safe_dlon_coslat = np.where(valid, dlon_coslat, np.asarray(0, dtype=scale_dtype))
    safe_dlat = np.where(valid, dlat, np.asarray(0, dtype=scale_dtype))
    has_longitude_scale = bool(scale != 0)
    safe_scale = scale if has_longitude_scale else np.asarray(1, dtype=scale_dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlon = (
            safe_dlon_coslat.astype(np.float64)
            / np.asarray(safe_scale, dtype=np.float64)
        )

    lon0, lat0 = np.radians(reference)
    lon = np.mod(lon0 + dlon.astype(np.float64), 2 * np.pi)
    lat = lat0 + safe_dlat.astype(np.float64)
    lon = np.where(valid & has_longitude_scale, lon, np.nan)
    lat = np.where(valid, lat, np.nan)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def offset_from_absolute(absolute, *, crval, offset_scale):
    """Convert absolute host sky coordinates to producer-typed offsets.

    ``absolute`` is the ``(lon, lat)`` pair in radians. The keyword-only
    ``crval`` is a concrete float64 ``(lon, lat)`` pair in degrees, and
    ``offset_scale`` is the consuming WCS's ``offset_scale()``. Longitude wrapping and
    the subtraction from ``crval`` happen in float64 before the differences
    are cast to ``offset_scale``'s dtype. This permits float64 absolute inputs
    with a float32 producer scale while preserving the producer's actual
    signed scale expression. The returned pair has the scale's dtype.

    This host transfer is not JIT- or autodiff-compatible. Non-finite input
    coordinates remain invalid in both output components.
    """
    reference = _host_crval(crval)
    scale = _host_offset_scale(offset_scale)
    lon, lat = _host_pair(absolute, "absolute")
    lon = _host_sky_component(lon, "longitude")
    lat = _host_sky_component(lat, "latitude")
    lon, lat = np.broadcast_arrays(lon, lat)

    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(reference).all()
        & np.isfinite(scale)
    )
    lon0, lat0 = np.radians(reference)
    safe_lon = np.where(valid, lon, lon0)
    safe_lat = np.where(valid, lat, lat0)
    dlon = np.mod(safe_lon - lon0 + np.pi, 2 * np.pi) - np.pi
    dlat = safe_lat - lat0

    scale_dtype = scale.dtype
    dlon = dlon.astype(scale_dtype)
    dlat = dlat.astype(scale_dtype)
    dlon_coslat = dlon * scale
    dlon_coslat = np.where(valid, dlon_coslat, np.asarray(np.nan, dtype=scale_dtype))
    dlat = np.where(valid, dlat, np.asarray(np.nan, dtype=scale_dtype))
    return dlon_coslat, dlat


class _PipelineBase(ABC):
    """Pixel <-> native <-> celestial pipeline shared by ``Tan``/``Sin``
    (and the reprojection legs).

    Every stage has an internal ``_..._valid`` form that returns finite
    values plus a validity flag; invalid samples carry a safe substitute.
    The public methods are thin wrappers that turn the flag into NaN at the
    very end. Inserting NaN earlier would let it reach a later multiply --
    the CD inverse, the SIP polynomial, the celestial rotation, the offset
    trig -- whose pullback then gives ``0 * NaN = NaN`` for the parameter
    gradients even when the caller masks the invalid sample out of the loss.
    """

    xp = np

    @abstractmethod
    def _native_to_plane(self, xn, yn, zn):
        """Map native unit vector (xn, yn, zn) to ``(X, Y, valid)``.

        ``X``/``Y`` must be finite even where ``valid`` is False; see the
        module docstring in ``projections.py``.
        """

    @abstractmethod
    def _plane_to_native(self, X, Y):
        """Map projection plane (X, Y) to ``(xn, yn, zn, valid)``."""

    def _mask(self, *values):
        """NaN out the invalid samples of ``values``, selected not multiplied.

        ``finite * nan`` poisons the pullback with ``0 * nan = nan``, so a
        multiplicative mask defeats the purpose of masking. ``where`` selects,
        and its gradient is the identity on the selected branch.

        The validity flag is the last positional argument.
        """
        *values, valid = values
        xp = self.xp
        # NaN in each output's own dtype: an untyped NaN is a weak f64
        # constant that would put a (folded-away) f64 op into an f32 trace.
        return tuple(xp.where(valid, v, xp.asarray(np.nan, dtype=xp.asarray(v).dtype))
                     for v in values)

    def _standin(self, valid, *pairs):
        """Replace invalid samples by a fixed, non-singular stand-in right
        before a step with a singular derivative (atan2, hypot).

        The stand-in values themselves are finite but arbitrary: an invalid
        SIN sample, say, lands at native (0, 0, 2.09), and after the rotation
        that can sit exactly on the pole of the spherical conversion, where
        the local derivative is infinite and ``0 * inf`` poisons the crval
        gradient. ``pairs`` is ``(value, fill), ...``.
        """
        xp = self.xp
        return tuple(xp.where(valid, v, xp.asarray(f, dtype=xp.asarray(v).dtype))
                     for v, f in pairs)

    def _finite_inputs(self, *values, fill=0.0):
        """``(*safe_values, valid)``: non-finite inputs (a caller's NaN, or an
        upstream stage's masked output) replaced by ``fill`` so they never
        reach the arithmetic below. ``fill`` may be a tuple, one per value."""
        xp = self.xp
        values = [xp.asarray(v) for v in values]
        valid = xp.isfinite(values[0])
        for v in values[1:]:
            valid = valid & xp.isfinite(v)
        fills = fill if isinstance(fill, tuple) else (fill,) * len(values)
        safe = [xp.where(valid, v, xp.asarray(f, dtype=v.dtype)) for v, f in zip(values, fills)]
        return (*safe, valid)

    # --- native <-> pixel -------------------------------------------------

    def _native_to_pix_valid(self, xn, yn, zn, valid=None):
        """``(x, y, valid)`` with finite ``x``/``y`` everywhere.

        ``valid`` is the validity accumulated upstream (e.g. the source leg
        of a ``Reprojector``). Every stage swaps the samples that are invalid
        *so far* for a regular point -- native (1, 0, 0), plane (0, 0) --
        before its own arithmetic. Substituting only on this stage's own
        domain test is not enough: an upstream-invalid sample can be in this
        stage's domain yet sit where the arithmetic overflows (a huge TAN
        coordinate fed to SIP) or a derivative is infinite (SIN's limb).
        """
        xn, yn, zn, ok = self._finite_inputs(xn, yn, zn, fill=(1.0, 0.0, 0.0))
        valid = ok if valid is None else valid & ok
        xn, yn, zn = self._standin(valid, (xn, 1.0), (yn, 0.0), (zn, 0.0))

        # Native -> Plane (projection specific).
        Xp, Yp, in_domain = self._native_to_plane(xn, yn, zn)
        cd_inv, cd_valid = _cd_inverse(self.cd, xp=self.xp)
        valid = valid & in_domain & cd_valid
        Xp, Yp = self._standin(valid, (Xp, 0.0), (Yp, 0.0))

        # Plane (radians) -> intermediate world (degrees) -> pixel.
        Xp_deg = self.xp.degrees(Xp)
        Yp_deg = self.xp.degrees(Yp)
        U = cd_inv[..., 0, 0] * Xp_deg + cd_inv[..., 0, 1] * Yp_deg
        V = cd_inv[..., 1, 0] * Xp_deg + cd_inv[..., 1, 1] * Yp_deg

        if self.sip is not None:
            U, V, converged = self.sip._foc_to_pix_valid(U, V, xp=self.xp)
            valid = valid & converged

        return U + self.crpix[0], V + self.crpix[1], valid

    def native_to_pix(self, xn, yn, zn):
        """
        Map native unit vector to pixel coordinates.
        Does NOT apply the celestial rotation.

        Returns NaN outside the projection's domain.
        """
        return self._mask(*self._native_to_pix_valid(xn, yn, zn))

    def _pix_to_native_valid(self, x, y):
        """``(xn, yn, zn, valid)`` with a finite native vector everywhere.

        As in ``_native_to_pix_valid``, samples invalid so far are swapped
        for a regular point (relative pixel 0, plane 0) before each stage:
        a stand-in pixel could otherwise land exactly on SIN's limb, where
        the sqrt has an infinite derivative.
        """
        x, y, valid = self._finite_inputs(x, y)

        # Pixel -> Plane (linear + distortion)
        u, v = self._standin(valid, (x - self.crpix[0], 0.0), (y - self.crpix[1], 0.0))
        if self.sip is not None:
            u, v = self.sip.pix_to_foc(u, v, xp=self.xp)

        # Multiplying by a singular CD gives a finite but meaningless native
        # vector, so the matrix itself has to be checked; a non-finite one is
        # replaced by the identity before the multiply.
        cd, _ = _cd_finite(self.cd, xp=self.xp)
        valid = valid & _cd_is_valid(self.cd, xp=self.xp)
        Xp_deg = cd[..., 0, 0] * u + cd[..., 0, 1] * v
        Yp_deg = cd[..., 1, 0] * u + cd[..., 1, 1] * v
        Xp, Yp = self._standin(valid, (self.xp.radians(Xp_deg), 0.0),
                               (self.xp.radians(Yp_deg), 0.0))

        # Plane -> Native (projection specific), in radians.
        xn, yn, zn, in_domain = self._plane_to_native(Xp, Yp)
        return xn, yn, zn, valid & in_domain

    def pix_to_native(self, x, y):
        """
        Map pixel coordinates to native unit vector.
        Does NOT apply the celestial rotation.

        Returns NaN outside the projection's domain.
        """
        return self._mask(*self._pix_to_native_valid(x, y))

    # --- celestial <-> native ---------------------------------------------

    def _rotation(self):
        """Celestial -> native rotation (one matrix, or one per element)."""
        raise NotImplementedError

    def _check_rotation_precision(self):
        """Hook for the JAX classes, whose rotation must run in float64."""

    def _celestial_to_native(self, x, y, z):
        self._check_rotation_precision()
        return apply_rotation(self._rotation(), x, y, z)

    def _native_to_celestial(self, xn, yn, zn):
        self._check_rotation_precision()
        return apply_rotation_transpose(self._rotation(), xn, yn, zn)

    def _proj_xyz_valid(self, x, y, z):
        x, y, z, valid = self._finite_inputs(x, y, z, fill=(1.0, 0.0, 0.0))
        xn, yn, zn = self._celestial_to_native(x, y, z)
        return self._native_to_pix_valid(xn, yn, zn, valid)

    def _unproj_xyz_valid(self, x, y):
        xn, yn, zn, valid = self._pix_to_native_valid(x, y)
        return (*self._native_to_celestial(xn, yn, zn), valid)

    def proj_xyz(self, x, y, z):
        """
        Forward projection: Celestial unit vector -> Pixel coordinates.

        Returns NaN where the position falls outside the projection's domain.
        """
        return self._mask(*self._proj_xyz_valid(x, y, z))

    def proj(self, ra, dec):
        """
        Forward projection: Celestial (radians) -> Pixel coordinates.

        Returns NaN where the position falls outside the projection's domain.
        """
        ra, dec, valid = self._finite_inputs(ra, dec)
        xc, yc, zc = radec_to_xyz(ra, dec, xp=self.xp)
        x, y, z, ok = self._finite_inputs(xc, yc, zc, fill=(1.0, 0.0, 0.0))
        xn, yn, zn = self._celestial_to_native(x, y, z)
        return self._mask(*self._native_to_pix_valid(xn, yn, zn, valid & ok))

    def unproj_xyz(self, x, y):
        """
        Inverse projection: Pixel coordinates -> Celestial unit vector.

        Returns NaN outside the projection's domain.
        """
        return self._mask(*self._unproj_xyz_valid(x, y))

    def unproj(self, x, y):
        """
        Inverse projection: Pixel coordinates -> Celestial (radians).

        Returns NaN outside the projection's domain.
        """
        xc, yc, zc, valid = self._unproj_xyz_valid(x, y)
        xc, yc, zc = self._standin(valid, (xc, 1.0), (yc, 0.0), (zc, 0.0))
        ra_rad, dec_rad = xyz_to_radec(xc, yc, zc, xp=self.xp)
        return self._mask(ra_rad, dec_rad, valid)
