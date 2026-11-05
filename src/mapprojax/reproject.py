"""Fused src-pixel -> tgt-pixel reprojection.

The per-pixel path uses only each projection's native-frame methods and a
relative rotation matrix. ``Reprojector`` accepts any projection implementing
the precision-conversion and finite-validity protocol; it does not depend on
projection class names or inheritance.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from .utils import apply_rotation, compose_relative_rotation
from .sip import Sip

SUPPORTED_MATRIX_PRECISIONS = ("f32", "f64")


def _cast_sip(sip, dtype):
    """Cast SIP coefficients while preserving its static Newton step count."""
    if sip is None:
        return None
    cast = lambda value: None if value is None else jnp.asarray(value, dtype=dtype)
    return Sip(a=cast(sip.a), b=cast(sip.b), ap=cast(sip.ap), bp=cast(sip.bp),
               inverse_steps=sip.inverse_steps)


@register_pytree_node_class
class Reprojector:
    """Fused pixel(``wcs_src``) -> pixel(``wcs_tgt``) reprojection.

    Parameters
    ----------
    wcs_src, wcs_tgt
        Projection objects implementing ``with_precision``,
        ``_pix_to_native_valid`` and ``_native_to_pix_valid`` and exposing a
        concrete, float64, degree-valued ``crval`` for host composition.
    frame_rotation : array, shape (3, 3), optional
        Optional host rotation applied between the source and target frames.
    matrix_precision : {"f32", "f64"}
        Precision used for the composed rotation matrix *and* the
        per-pixel ``pix_to_native``/``native_to_pix`` legs:

        - ``"f32"`` (default): plain float32 throughout the per-pixel
          path, independent of ``jax_enable_x64``. Accurate to ~1e-4 px
          for overlapping fields (separations up to a few degrees at
          arcsecond plate scales); error grows with field-center
          separation and with finer plate scales -- see
          ``fo64_report_and_plan.md``.
        - ``"f64"``: float64 throughout; requires
          ``jax_enable_x64``.

    Use :meth:`from_matrix` when ``m_rel`` is dynamic or depends on live
    source/target reference values. Its matrix is a JAX leaf and remains
    differentiable; no host reference composition occurs on that path.
    """

    def __init__(self, wcs_src, wcs_tgt, frame_rotation=None, matrix_precision="f32"):
        self._init_legs(wcs_src, wcs_tgt, matrix_precision)
        self.m_rel = jnp.asarray(
            compose_relative_rotation(wcs_src, wcs_tgt, frame_rotation),
            dtype=self._dtype,
        )

    def _init_legs(self, wcs_src, wcs_tgt, matrix_precision):
        if matrix_precision not in SUPPORTED_MATRIX_PRECISIONS:
            raise ValueError(
                f"matrix_precision must be one of {SUPPORTED_MATRIX_PRECISIONS}, "
                f"got {matrix_precision!r}"
            )
        if matrix_precision == "f64" and not jax.config.jax_enable_x64:
            raise ValueError(
                "Reprojector(matrix_precision='f64') requires float64: call "
                "jax.config.update('jax_enable_x64', True)."
            )
        precision = matrix_precision
        self.matrix_precision = matrix_precision
        self._dtype = jnp.float64 if matrix_precision == "f64" else jnp.float32
        try:
            self._src_leg = wcs_src.with_precision(precision)
            self._tgt_leg = wcs_tgt.with_precision(precision)
        except AttributeError as exc:
            raise TypeError(
                "Reprojector projections must implement nonmutating "
                "with_precision(precision)"
            ) from exc

    @classmethod
    def from_matrix(cls, wcs_src, wcs_tgt, m_rel, *, matrix_precision="f32"):
        """Build a Reprojector from a live native(src)->native(tgt) matrix.

        ``m_rel`` is cast to the kernel precision but is never converted to
        the host, so callers can differentiate through the matrix and its
        source/target reference dependencies.
        """
        obj = cls.__new__(cls)
        obj._init_legs(wcs_src, wcs_tgt, matrix_precision)
        obj.m_rel = jnp.asarray(m_rel, dtype=obj._dtype)
        if obj.m_rel.shape != (3, 3):
            raise ValueError(f"m_rel must have shape (3, 3), got {obj.m_rel.shape}")
        return obj

    @property
    def tgt_leg(self):
        """The ``wcs_tgt``-side leg object (exposes ``pix_to_native``,
        ``native_to_pix``, ``crpix``, ``cd``, ``sip``, all at this
        ``Reprojector``'s dtype) -- for callers that need to post-process
        the native-frame vector themselves before finishing
        ``native_to_pix`` (e.g. refining an approximate SIP inverse with
        their own Newton iteration; that refinement is itself entirely in
        the relative/native regime, so it composes cleanly with this
        class's fp32 path). See also :meth:`pix_to_native_tgt`.
        """
        return self._tgt_leg

    @property
    def src_leg(self):
        """The ``wcs_src``-side leg object. See :attr:`tgt_leg`."""
        return self._src_leg

    def pix_to_native_tgt(self, x, y):
        """Map pixel coordinates in ``wcs_src`` to the native-frame unit
        vector components ``(xn, yn, zn)`` in ``wcs_tgt``'s frame --
        :meth:`pix_to_pix` stops one step short of this, returning
        ``self.tgt_leg.native_to_pix(*this)``. Exposed for callers that
        need that intermediate vector directly.
        """
        xn, yn, zn, valid = self._pix_to_native_tgt_valid(x, y)
        return self._mask(xn, yn, zn, valid)

    def _pix_to_native_tgt_valid(self, x, y):
        x = jnp.asarray(x, dtype=self._dtype)
        y = jnp.asarray(y, dtype=self._dtype)

        xn, yn, zn, valid = self._src_leg._pix_to_native_valid(x, y)
        finite_matrix = jnp.all(jnp.isfinite(self.m_rel))
        eye = jnp.eye(3, dtype=self._dtype)
        matrix = jnp.where(finite_matrix, self.m_rel, eye)
        valid = valid & finite_matrix
        xt, yt, zt = apply_rotation(matrix, xn, yn, zn)
        finite_vector = jnp.isfinite(xt) & jnp.isfinite(yt) & jnp.isfinite(zt)
        valid = valid & finite_vector
        xt = jnp.where(valid, xt, jnp.ones_like(xt))
        yt = jnp.where(valid, yt, jnp.zeros_like(yt))
        zt = jnp.where(valid, zt, jnp.zeros_like(zt))
        return xt, yt, zt, valid

    def pix_to_pix(self, x, y):
        """Map pixel coordinates in ``wcs_src`` to the corresponding pixel
        coordinates in ``wcs_tgt``.

        Returns NaN where either leg is outside its domain. Validity is
        carried between the legs rather than passed as NaN, so masked-loss
        gradients for both legs' CD and SIP stay finite.
        """
        xt, yt, zt, valid = self._pix_to_native_tgt_valid(x, y)
        x, y, valid = self._tgt_leg._native_to_pix_valid(xt, yt, zt, valid)
        return self._mask(x, y, valid)

    def pixel_map(self, x, y, *, bounds_shape=None):
        """Map source pixels to finite target pixels and return validity.

        The returned coordinates use this Reprojector's kernel dtype, with
        zero stand-ins wherever ``valid`` is false. When ``bounds_shape`` is
        provided, it must be a static ``(ny, nx)`` tuple of positive Python
        integers and the target pixel must lie in
        ``[-0.5, nx - 0.5) x [-0.5, ny - 0.5)``. Bounds classify the computed
        kernel-dtype coordinates without clipping or promotion; near an edge,
        float32 can therefore classify a coordinate differently from float64.

        Coordinate values may be traced. Keep ``bounds_shape`` static when
        jitting this method.
        """
        if bounds_shape is not None:
            if not isinstance(bounds_shape, tuple):
                raise TypeError("bounds_shape must be a tuple of two Python integers")
            if len(bounds_shape) != 2:
                raise ValueError("bounds_shape must contain exactly two dimensions")
            if any(type(size) is not int for size in bounds_shape):
                raise TypeError("bounds_shape must be a tuple of two Python integers")
            if any(size <= 0 for size in bounds_shape):
                raise ValueError("bounds_shape dimensions must be positive")

        xt, yt, zt, valid = self._pix_to_native_tgt_valid(x, y)
        x_mapped, y_mapped, valid = self._tgt_leg._native_to_pix_valid(
            xt, yt, zt, valid
        )
        x_mapped = jnp.asarray(x_mapped, dtype=self._dtype)
        y_mapped = jnp.asarray(y_mapped, dtype=self._dtype)
        valid = valid & jnp.isfinite(x_mapped) & jnp.isfinite(y_mapped)

        if bounds_shape is not None:
            ny, nx = bounds_shape
            half = jnp.asarray(0.5, dtype=self._dtype)
            x_upper = jnp.asarray(nx, dtype=self._dtype) - half
            y_upper = jnp.asarray(ny, dtype=self._dtype) - half
            valid = (valid & (x_mapped >= -half) & (x_mapped < x_upper)
                     & (y_mapped >= -half) & (y_mapped < y_upper))

        x_mapped = jnp.where(valid, x_mapped, jnp.zeros_like(x_mapped))
        y_mapped = jnp.where(valid, y_mapped, jnp.zeros_like(y_mapped))
        return x_mapped, y_mapped, valid

    @staticmethod
    def _mask(*values):
        *outputs, valid = values
        return tuple(
            jnp.where(valid, value, jnp.asarray(jnp.nan, dtype=jnp.asarray(value).dtype))
            for value in outputs
        )

    def tree_flatten(self):
        children = (self._src_leg, self._tgt_leg, self.m_rel)
        aux_data = (self.matrix_precision, self._dtype)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj._src_leg, obj._tgt_leg, obj.m_rel = children
        obj.matrix_precision, obj._dtype = aux_data
        return obj
