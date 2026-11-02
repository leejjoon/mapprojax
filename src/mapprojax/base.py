from abc import ABC, abstractmethod

import jax
import numpy as np
from .utils import radec_to_xyz, xyz_to_radec, rotation_matrix, apply_rotation, apply_rotation_transpose
from .sip import Sip


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


class _PipelineBase(ABC):
    """Pixel <-> native <-> celestial pipeline shared by the scalar and the
    array WCS bases (and the reprojection legs).

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


class WCSBase(_PipelineBase):
    xp = np

    def __init__(self, crpix, cd, crval, sip=None):
        self.crpix = self.xp.array(crpix, dtype=float)
        self.cd = self.xp.array(cd, dtype=float)
        self.crval = self.xp.array(crval, dtype=float)
        self.sip = sip
        _check_cd(self.cd)

    @property
    def r_matrix(self):
        """Celestial -> native rotation, derived from ``crval``.

        Never stored: a cached matrix would stop following the ``crval`` leaf
        across a pytree flatten/unflatten and silently zero the ``crval``
        gradient. Recomputed on access, which is a 3x3 trig evaluation, not
        per-pixel work.
        """
        return rotation_matrix(
            self.xp.radians(self.crval[0]),
            self.xp.radians(self.crval[1]),
            xp=self.xp,
        )

    def _rotation(self):
        return self.r_matrix

    def unproj_offset(self, x, y):
        """Inverse projection: pixel -> small sky offset from CRVAL,
        ``(Delta_alpha * cos(dec0), Delta_delta)`` (radians).

        Unlike :meth:`unproj`, this never forms an absolute (O(1)) RA/Dec
        value: every intermediate quantity here is field-of-view-sized,
        computed via spherical trig (``fo64_report_and_plan.md`` Phase 2,
        dfreproject-style), so this is accurate in float32 regardless of
        ``jax_enable_x64`` -- see :meth:`unproj_absolute` for the O(1)
        epilogue that recovers true RA/Dec, and DEVELOPER.md's Precision
        Strategy section for why ``unproj``/``unproj_xyz`` need float64
        but this does not.

        The ``cos(dec0)`` scaling is the standard astrometric convention
        (RA offsets compress toward the pole); it degenerates as CRVAL's
        declination approaches +/-90 deg, same as any RA-based offset.

        Returns
        -------
        dalpha_cosdec, ddelta : same shape as x, y
        """
        xn, yn, zn, valid = self._pix_to_native_valid(x, y)
        p = self.xp.radians(self.crval[1])
        cos_p, sin_p = self.xp.cos(p), self.xp.sin(p)

        # w, zc: the native vector rotated by CRVAL's declination alone
        # (no dependence on CRVAL's RA) -- safe to compute in fp32 since
        # xn is O(1) (near the native pole) and yn/zn are field-sized, so
        # this is "large + small", never a difference of two O(1) terms.
        w = cos_p * xn - sin_p * zn
        zc = sin_p * xn + cos_p * zn
        # Invalid samples get a stand-in before the atan2/hypot below.
        yn, w, zc = self._standin(valid, (yn, 0.0), (w, 1.0), (zc, 0.0))

        # Delta_alpha = atan2(yn, w) is *exact* (not an approximation):
        # yn = cos(delta)*sin(Delta_alpha) and w = cos(delta)*cos(Delta_alpha)
        # are themselves exact identities, so their ratio's atan2 recovers
        # Delta_alpha exactly modulo ordinary fp32 rounding -- no
        # cancellation, regardless of field-center separation.
        dalpha = self.xp.arctan2(yn, w)

        # Delta_delta via the sin/cos-difference formulas, with cos(delta)
        # recovered safely as hypot(yn, w) (exact; no cancellation) rather
        # than sqrt(1 - zc**2). A naive "arcsin(zc) - dec0" would subtract
        # two O(1) angles (recreating the exact problem this function
        # exists to avoid); computing sin(Delta_delta)/cos(Delta_delta)
        # this way and wrapping with atan2 keeps the fp32 error at the
        # ~1e-7 rad floor (ordinary fp32 rounding) across separations from
        # sub-degree to >45 deg, instead of growing with separation
        # (measured: atan2(zn, xn) alone -- the naive small-angle
        # shortcut -- grows to ~0.08 rad error by 20 deg; this formula
        # stays ~1e-7).
        cos_delta = self.xp.hypot(yn, w)
        sin_ddelta = zc * cos_p - cos_delta * sin_p
        cos_ddelta = cos_delta * cos_p + zc * sin_p
        ddelta = self.xp.arctan2(sin_ddelta, cos_ddelta)

        return self._mask(dalpha * cos_p, ddelta, valid)

    def unproj_absolute(self, x, y):
        """Inverse projection: pixel -> true celestial (RA, Dec) in
        radians, RA wrapped to ``[0, 2*pi)``.

        The bulk math (:meth:`unproj_offset`) runs in whatever precision
        ``self.xp`` gives it; only the final ``crval + offset`` add and
        the ``% 2*pi`` wrap -- negligible cost -- need float64, which is
        why this requires a NumPy-backed WCS (``Tan``/``Sin``): a JAX
        WCS's own ``crval`` may already be truncated to float32 at
        *construction* time if ``jax_enable_x64`` was off then, and this
        method cannot recover precision that was already discarded.
        """
        if self.xp is not np:
            raise TypeError(
                "unproj_absolute requires a NumPy-backed WCS (e.g. Tan/Sin): "
                "the float64 epilogue needs crval's true precision, and a JAX "
                "WCS's crval may already be truncated to float32 (if "
                "jax_enable_x64 was off when it was constructed) before this "
                "method ever runs. Use unproj_offset directly if crval-scale "
                "float64 doesn't matter for your use case."
            )
        dalpha_cosdec, ddelta = self.unproj_offset(x, y)
        dec0 = np.radians(self.crval[1])
        ra0 = np.radians(self.crval[0])
        dalpha = dalpha_cosdec / np.cos(dec0)
        ra = np.mod(ra0 + dalpha, 2 * np.pi)
        dec = dec0 + ddelta
        return ra, dec

    def proj_offset(self, dalpha_cosdec, ddelta):
        """Forward projection: a small sky offset from CRVAL,
        ``(Delta_alpha * cos(dec0), Delta_delta)`` (radians) -> pixel
        coordinates. Inverse of :meth:`unproj_offset`; see its docstring
        for the accuracy/precision rationale.
        """
        dalpha_cosdec, ddelta, valid = self._finite_inputs(dalpha_cosdec, ddelta)
        p = self.xp.radians(self.crval[1])
        cos_p, sin_p = self.xp.cos(p), self.xp.sin(p)
        dalpha = dalpha_cosdec / cos_p

        cos_dd, sin_dd = self.xp.cos(ddelta), self.xp.sin(ddelta)
        cos_da, sin_da = self.xp.cos(dalpha), self.xp.sin(dalpha)

        # Inverse of unproj_offset's rotation-by-dec0 step: cos(delta) and
        # sin(delta) recovered via the safe sum formulas (large + small,
        # since Delta_delta is field-sized), then rotated back by -dec0.
        cos_delta = cos_p * cos_dd - sin_p * sin_dd
        zc = sin_p * cos_dd + cos_p * sin_dd
        w = cos_delta * cos_da
        yn = cos_delta * sin_da
        xn = w * cos_p + zc * sin_p
        zn = -w * sin_p + zc * cos_p

        return self._mask(*self._native_to_pix_valid(xn, yn, zn, valid))

    def proj_absolute(self, ra, dec):
        """Forward projection: true celestial (RA, Dec) in radians ->
        pixel coordinates.

        Wraps ``ra - ra0`` to ``(-pi, pi]`` in float64 (cheap, exact for
        any input range including RA near the 0/2*pi boundary) before
        handing the now-small offset to :meth:`proj_offset`'s float32
        bulk math. See :meth:`unproj_absolute` for why this requires a
        NumPy-backed WCS.
        """
        if self.xp is not np:
            raise TypeError(
                "proj_absolute requires a NumPy-backed WCS (e.g. Tan/Sin) -- "
                "see unproj_absolute's docstring for why."
            )
        ra0 = np.radians(self.crval[0])
        dec0 = np.radians(self.crval[1])
        dalpha = np.mod(np.asarray(ra) - ra0 + np.pi, 2 * np.pi) - np.pi
        ddelta = np.asarray(dec) - dec0
        dalpha_cosdec = dalpha * np.cos(dec0)
        return self.proj_offset(dalpha_cosdec, ddelta)

    def to_dict(self):
        """Basic serialization helper."""
        return {
            'crpix': self.crpix.tolist(),
            'cd': self.cd.tolist(),
            'crval': self.crval.tolist(),
            'sip': self.sip.to_dict() if self.sip else None,
            'type': self.__class__.__name__
        }
    
    @classmethod
    def from_dict(cls, data):
        sip_data = data.get('sip')
        sip = Sip.from_dict(sip_data) if sip_data else None
        return cls(data['crpix'], data['cd'], data['crval'], sip=sip)


class WCSArrayBase(_PipelineBase):
    xp = np

    def __init__(self, crpix, cd, crvals, sip=None):
        self.crpix = self.xp.array(crpix, dtype=float)
        self.cd = self.xp.array(cd, dtype=float)
        self.sip = sip
        _check_cd(self.cd)
        
        # crvals is expected to be (ra_array, dec_array) or similar
        # shape: (2, N) or tuple of arrays
        ra_arr, dec_arr = self.xp.asarray(crvals[0]), self.xp.asarray(crvals[1])
        if ra_arr.shape != dec_arr.shape:
             raise ValueError("RA and Dec arrays in crvals must have same shape")
             
        self.crvals = (ra_arr, dec_arr)

    @property
    def r_matrices(self):
        """Celestial -> native rotation per element, derived from ``crvals``.
        See :attr:`WCSBase.r_matrix` for why this is not stored.
        """
        ra_arr, dec_arr = self.crvals
        return rotation_matrix(
            self.xp.radians(ra_arr),
            self.xp.radians(dec_arr),
            xp=self.xp,
        )

    def _rotation(self):
        return self.r_matrices

    def to_dict(self):
        return {
            'crpix': self.crpix.tolist(),
            'cd': self.cd.tolist(),
            'crvals': [self.xp.asarray(c).tolist() for c in self.crvals],
            'sip': self.sip.to_dict() if self.sip else None,
            'type': self.__class__.__name__
        }