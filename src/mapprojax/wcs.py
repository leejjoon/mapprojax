"""Precision-explicit JAX TAN and SIN WCS classes.

These classes share the established pixel/native/celestial pipeline while
keeping their numerical precision policy and PyTree reconstruction explicit.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from .base import WCSBase, _check_cd
from .jax_base import _effective_f64, _require_x64
from .projections import SinMixin, TanMixin
from .sip import Sip
from .utils import apply_rotation, apply_rotation_transpose, rotation_matrix


_PRECISIONS = {"f32", "mixed", "f64"}


def _is_array_or_tracer(value):
    return isinstance(value, (jax.Array, jax.core.Tracer, np.ndarray))


def _sequence_float_dtype(value):
    """Return a sequence component's floating dtype, if it has one."""
    if isinstance(value, float):
        return np.dtype(np.float64)
    if isinstance(value, np.floating):
        return value.dtype
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        dtype = np.dtype(dtype)
        if dtype.kind == "f":
            return dtype
    return None


def _normalize_crval(crval):
    """Keep array/tracer references intact; pack coordinate sequences once."""
    if _is_array_or_tracer(crval):
        result = crval
    elif isinstance(crval, Sequence) and not isinstance(crval, (str, bytes)):
        items = tuple(crval)
        on_device = any(isinstance(item, (jax.Array, jax.core.Tracer)) for item in items)
        if on_device:
            result = jnp.stack(tuple(jnp.asarray(item) for item in items))
        else:
            result = np.asarray(items)
        kind = np.dtype(result.dtype).kind
        if kind in "biu":
            # Integer degrees lose nothing in float64; an integer leaf would
            # otherwise fail the reference checks with a misleading message.
            result = (jnp if on_device else np).asarray(result, dtype=np.float64)
        elif kind == "f":
            # Reject assembly that changed any component's own floating
            # dtype: [np.float32(x), 30] would otherwise become a float64
            # reference that hides the truncated component.
            assembled = np.dtype(result.dtype)
            for item in items:
                dtype = _sequence_float_dtype(item)
                if dtype is not None and dtype != assembled:
                    raise TypeError(
                        "crval sequence components must use one floating "
                        f"precision; a {dtype} component would be stored as "
                        f"{assembled}"
                    )
    else:
        result = np.asarray(crval)

    if getattr(result, "shape", None) != (2,):
        raise ValueError(f"crval must have shape (2,), got {getattr(result, 'shape', None)}")
    return result


def _cast_sip(sip, dtype):
    if sip is None:
        return None
    if not isinstance(sip, Sip):
        raise TypeError("sip must be a Sip instance or None")
    return Sip(
        a=None if sip.a is None else jnp.asarray(sip.a, dtype=dtype),
        b=None if sip.b is None else jnp.asarray(sip.b, dtype=dtype),
        ap=None if sip.ap is None else jnp.asarray(sip.ap, dtype=dtype),
        bp=None if sip.bp is None else jnp.asarray(sip.bp, dtype=dtype),
        inverse_steps=sip.inverse_steps,
    )


class _PrecisionWCSBase(WCSBase):
    """Shared implementation for the two precision-explicit projections."""

    xp = jnp

    def __init__(self, crpix, cd, crval, sip=None, *, precision: str):
        if not isinstance(precision, str) or precision not in _PRECISIONS:
            raise ValueError(
                f"precision must be one of {sorted(_PRECISIONS)}, got {precision!r}"
            )
        if precision in ("mixed", "f64"):
            _require_x64(f"precision={precision!r}")

        self.precision = precision
        dtype = self._pipeline_dtype
        self.crpix = self.xp.asarray(crpix, dtype=dtype)
        self.cd = self.xp.asarray(cd, dtype=dtype)
        if self.crpix.shape != (2,):
            raise ValueError(f"crpix must have shape (2,), got {self.crpix.shape}")
        if self.cd.shape != (2, 2):
            raise ValueError(f"cd must have shape (2, 2), got {self.cd.shape}")
        self.crval = _normalize_crval(crval)
        self.sip = _cast_sip(sip, dtype)
        # Validate the stored, post-cast matrix. _check_cd deliberately skips
        # tracers, whose validity is carried through the pipeline at runtime.
        _check_cd(self.cd)

    @property
    def _pipeline_dtype(self):
        return jnp.float64 if self.precision == "f64" else jnp.float32

    @property
    def _reference_dtype(self):
        return jnp.float32 if self.precision == "f32" else jnp.float64

    def _require_reference_f64(self, operation):
        _require_x64(operation)
        if not _effective_f64(self.crval):
            raise TypeError(
                f"{operation} requires a true float64 crval leaf; "
                "the constructor preserves crval dtype and cannot restore precision"
            )

    def _check_rotation_precision(self):
        if self.precision == "f32":
            raise TypeError(
                "precision='f32' supports the offset pipeline, not absolute "
                "celestial rotation; use precision='mixed' or 'f64' with x64 enabled"
            )
        self._require_reference_f64("absolute celestial rotation")

    def _check_native_precision(self):
        if self.precision == "f64":
            _require_x64("precision='f64' native operations")

    def _check_offset_precision(self):
        self._check_native_precision()
        if self.precision != "f32":
            self._require_reference_f64("offset reference trigonometry")

    def _pix_to_native_valid(self, x, y):
        self._check_native_precision()
        x = self.xp.asarray(x, dtype=self._pipeline_dtype)
        y = self.xp.asarray(y, dtype=self._pipeline_dtype)
        return super()._pix_to_native_valid(x, y)

    def _native_to_pix_valid(self, xn, yn, zn, valid=None):
        self._check_native_precision()
        dtype = self._pipeline_dtype
        xn = self.xp.asarray(xn, dtype=dtype)
        yn = self.xp.asarray(yn, dtype=dtype)
        zn = self.xp.asarray(zn, dtype=dtype)
        return super()._native_to_pix_valid(xn, yn, zn, valid)

    @property
    def r_matrix(self):
        self._check_rotation_precision()
        if self.precision == "f32":
            reference = self.xp.asarray(self.crval, dtype=jnp.float32)
        else:
            self._require_reference_f64("celestial rotation")
            reference = self.xp.asarray(self.crval, dtype=jnp.float64)
        return rotation_matrix(
            self.xp.radians(reference[0]),
            self.xp.radians(reference[1]),
            xp=self.xp,
        )

    def _celestial_to_native(self, x, y, z):
        self._check_rotation_precision()
        x = self.xp.asarray(x, dtype=jnp.float64)
        y = self.xp.asarray(y, dtype=jnp.float64)
        z = self.xp.asarray(z, dtype=jnp.float64)
        xn, yn, zn = apply_rotation(self.r_matrix, x, y, z)
        if self.precision == "mixed":
            return xn.astype(jnp.float32), yn.astype(jnp.float32), zn.astype(jnp.float32)
        return xn, yn, zn

    def _native_to_celestial(self, xn, yn, zn):
        self._check_rotation_precision()
        xn = self.xp.asarray(xn, dtype=jnp.float64)
        yn = self.xp.asarray(yn, dtype=jnp.float64)
        zn = self.xp.asarray(zn, dtype=jnp.float64)
        return apply_rotation_transpose(self.r_matrix, xn, yn, zn)

    def proj(self, ra, dec):
        self._check_rotation_precision()
        return super().proj(
            self.xp.asarray(ra, dtype=jnp.float64),
            self.xp.asarray(dec, dtype=jnp.float64),
        )

    def proj_xyz(self, x, y, z):
        self._check_rotation_precision()
        return super().proj_xyz(
            self.xp.asarray(x, dtype=jnp.float64),
            self.xp.asarray(y, dtype=jnp.float64),
            self.xp.asarray(z, dtype=jnp.float64),
        )

    def _reference_trig(self):
        """Return reference declination sine/cosine in the offset dtype."""
        if self.precision == "f32":
            dec = self.xp.asarray(self.crval[1], dtype=jnp.float32)
        else:
            self._require_reference_f64("offset reference trigonometry")
            dec = self.xp.asarray(self.crval[1], dtype=jnp.float64)
        p = self.xp.radians(dec)
        return self.xp.cos(p), self.xp.sin(p)

    def offset_scale(self):
        """Return the longitude scale used by the offset methods."""
        cos_p, _ = self._reference_trig()
        return cos_p

    def unproj_offset(self, x, y):
        """Map pixels to ``(Delta_lon*cos(lat0), Delta_lat)`` in radians."""
        self._check_native_precision()
        x = self.xp.asarray(x, dtype=self._pipeline_dtype)
        y = self.xp.asarray(y, dtype=self._pipeline_dtype)
        xn, yn, zn, valid = self._pix_to_native_valid(x, y)
        cos_p, sin_p = self._reference_trig()

        # Preserve Stage 2's stable large-plus-small spherical difference.
        if self.precision == "mixed":
            xn = xn.astype(jnp.float64)
            yn = yn.astype(jnp.float64)
            zn = zn.astype(jnp.float64)
        w = cos_p * xn - sin_p * zn
        zc = sin_p * xn + cos_p * zn
        yn, w, zc = self._standin(valid, (yn, 0.0), (w, 1.0), (zc, 0.0))
        dalpha = self.xp.arctan2(yn, w)
        cos_delta = self.xp.hypot(yn, w)
        sin_ddelta = zc * cos_p - cos_delta * sin_p
        cos_ddelta = cos_delta * cos_p + zc * sin_p
        ddelta = self.xp.arctan2(sin_ddelta, cos_ddelta)
        return self._mask(dalpha * cos_p, ddelta, valid)

    def proj_offset(self, dalpha_cosdec, ddelta):
        """Map ``(Delta_lon*cos(lat0), Delta_lat)`` in radians to pixels."""
        self._check_offset_precision()
        dtype = self._reference_dtype
        dalpha_cosdec = self.xp.asarray(dalpha_cosdec, dtype=dtype)
        ddelta = self.xp.asarray(ddelta, dtype=dtype)
        dalpha_cosdec, ddelta, valid = self._finite_inputs(dalpha_cosdec, ddelta)
        cos_p, sin_p = self._reference_trig()
        dalpha = dalpha_cosdec / cos_p

        cos_dd, sin_dd = self.xp.cos(ddelta), self.xp.sin(ddelta)
        cos_da, sin_da = self.xp.cos(dalpha), self.xp.sin(dalpha)
        cos_delta = cos_p * cos_dd - sin_p * sin_dd
        zc = sin_p * cos_dd + cos_p * sin_dd
        w = cos_delta * cos_da
        yn = cos_delta * sin_da
        xn = w * cos_p + zc * sin_p
        zn = -w * sin_p + zc * cos_p

        if self.precision == "mixed":
            xn, yn, zn = (v.astype(jnp.float32) for v in (xn, yn, zn))
        return self._mask(*self._native_to_pix_valid(xn, yn, zn, valid))

    def with_precision(self, precision: str):
        """Return this projection with numerical leaves converted to a mode."""
        return type(self)(
            self.crpix, self.cd, self.crval, sip=self.sip, precision=precision
        )

    def tree_flatten(self):
        children = (self.crpix, self.cd, self.crval, self.sip)
        sip_steps = None if self.sip is None else self.sip.inverse_steps
        return children, (self.precision, sip_steps)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        precision, _sip_steps = aux_data
        obj = cls.__new__(cls)
        obj.crpix, obj.cd, obj.crval, obj.sip = children
        obj.precision = precision
        return obj

    @property
    def crval_dtype(self):
        return str(self.crval.dtype)

    def to_dict(self):
        """Serialize a concrete WCS, preserving the normalized reference dtype."""
        leaves = jax.tree_util.tree_leaves(
            (self.crpix, self.cd, self.crval, self.sip)
        )
        if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
            raise TypeError("cannot serialize a traced WCS")
        expected = np.dtype("float64" if self.precision == "f64" else "float32")
        for name in ("crpix", "cd"):
            if np.asarray(getattr(self, name)).dtype != expected:
                raise ValueError(
                    f"{name} dtype is inconsistent with precision={self.precision!r}"
                )
        if self.sip is not None:
            for name in ("a", "b", "ap", "bp"):
                value = getattr(self.sip, name)
                if value is not None and np.asarray(value).dtype != expected:
                    raise ValueError(
                        f"SIP {name} dtype is inconsistent with "
                        f"precision={self.precision!r}"
                    )

        def to_list(value):
            return None if value is None else np.asarray(value).tolist()

        sip_data = None
        if self.sip is not None:
            sip_data = {
                "a": to_list(self.sip.a),
                "b": to_list(self.sip.b),
                "ap": to_list(self.sip.ap),
                "bp": to_list(self.sip.bp),
                "inverse_steps": self.sip.inverse_steps,
            }
        return {
            "type": type(self).__name__,
            "precision": self.precision,
            "crpix": to_list(self.crpix),
            "cd": to_list(self.cd),
            "crval": to_list(self.crval),
            "crval_dtype": self.crval_dtype,
            "sip": sip_data,
        }

    @classmethod
    def from_dict(cls, data):
        """Load only this class's complete precision-aware JSON contract."""
        required = {
            "type", "precision", "crpix", "cd", "crval", "crval_dtype", "sip"
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("WCS data must contain exactly the precision-aware contract fields")
        if data["type"] != cls.__name__:
            raise ValueError(
                f"expected WCS type {cls.__name__!r}, got {data['type']!r}"
            )
        if not isinstance(data["crval_dtype"], str):
            raise ValueError("crval_dtype must be a dtype name")
        try:
            crval_dtype = np.dtype(data["crval_dtype"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid crval_dtype: {data['crval_dtype']!r}") from exc
        crval = np.asarray(data["crval"], dtype=crval_dtype)

        sip_data = data["sip"]
        sip = None
        if sip_data is not None:
            sip_fields = {"a", "b", "ap", "bp", "inverse_steps"}
            if not isinstance(sip_data, dict) or set(sip_data) != sip_fields:
                raise ValueError("SIP data must include coefficients and inverse_steps")
            sip = Sip.from_dict(sip_data)
        return cls(
            data["crpix"], data["cd"], crval, sip=sip,
            precision=data["precision"],
        )


@register_pytree_node_class
class Tan(TanMixin, _PrecisionWCSBase):
    """Tangent-plane JAX WCS with explicit f32, mixed, or f64 precision."""


@register_pytree_node_class
class Sin(SinMixin, _PrecisionWCSBase):
    """Orthographic JAX WCS with explicit f32, mixed, or f64 precision."""

