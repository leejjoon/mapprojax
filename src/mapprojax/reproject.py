"""Fused src-pixel -> tgt-pixel reprojection with no fp64 on the per-pixel
path (``fo64_report_and_plan.md``, Phase 1).

The fp64 requirement in ordinary WCS math comes from routing through an
absolute celestial unit vector between two WCSes' native frames: each WCS's
own celestial<->native rotation is an O(1)-to-O(1) rotation that
catastrophically cancels into tiny field offsets when only fp32 is
available (see DEVELOPER.md's Precision Strategy section).
:class:`Reprojector` sidesteps that entirely for the reprojection use case
(map a pixel in one WCS directly to the corresponding pixel in another):
it composes the two WCSes' rotations into a single native(src) ->
native(tgt) matrix, in fp64, *once*, on the host
(:func:`mapprojax.utils.compose_relative_rotation`); every per-pixel call
then runs ``pix_to_native -> apply that matrix -> native_to_pix`` in fp32
(or fp64), never forming an absolute celestial vector.
"""

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .base import WCSBase
from .projections import TanMixin, SinMixin
from .sip import Sip
from .utils import apply_rotation, compose_relative_rotation

SUPPORTED_MATRIX_PRECISIONS = ("f32", "f64")


class _ReprojectionLegBase(WCSBase):
    """WCS-like object exposing only ``pix_to_native``/``native_to_pix``,
    holding ``crpix``/``cd``/``sip`` at exactly the caller-given dtype.

    Unlike plain ``TanJax``/``SinJax``, whose inherited
    ``WCSBase.__init__`` hardcodes ``dtype=float`` (i.e. always promotes
    to the highest available precision whenever ``jax_enable_x64`` is on,
    regardless of the dtype of the arrays passed in -- exactly why
    ``MixedPrecisionWCSJaxMixin`` needs its own ``__init__`` override),
    this class never re-casts, including across a pytree
    flatten/unflatten round-trip (e.g. inside ``jax.jit``/``vmap``). Used
    internally by :class:`Reprojector`; not part of the public API.

    ``crval``/``r_matrix`` are intentionally absent: ``Reprojector``'s
    precomputed ``m_rel`` already absorbs the crval-driven rotation for
    both legs, and ``native_to_pix``/``pix_to_native`` never reference
    ``r_matrix``.
    """

    xp = jnp

    def __init__(self, crpix, cd, sip=None):
        self.crpix = crpix
        self.cd = cd
        self.sip = sip

    def tree_flatten(self):
        return (self.crpix, self.cd, self.sip), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        crpix, cd, sip = children
        return cls(crpix, cd, sip=sip)


@register_pytree_node_class
class _TanLeg(TanMixin, _ReprojectionLegBase):
    pass


@register_pytree_node_class
class _SinLeg(SinMixin, _ReprojectionLegBase):
    pass


_LEG_CLS_FOR_MIXIN = {TanMixin: _TanLeg, SinMixin: _SinLeg}


def _leg_cls_for(wcs):
    for mixin, leg_cls in _LEG_CLS_FOR_MIXIN.items():
        if isinstance(wcs, mixin):
            return leg_cls
    supported = [m.__name__ for m in _LEG_CLS_FOR_MIXIN]
    raise TypeError(f"unsupported projection type {type(wcs).__name__!r}; Reprojector supports {supported}")


def _cast_sip(sip, dtype):
    if sip is None:
        return None

    def c(arr):
        return None if arr is None else jnp.asarray(arr, dtype=dtype)

    return Sip(a=c(sip.a), b=c(sip.b), ap=c(sip.ap), bp=c(sip.bp))


def _leg(wcs, dtype):
    leg_cls = _leg_cls_for(wcs)
    crpix = jnp.asarray(wcs.crpix, dtype=dtype)
    cd = jnp.asarray(wcs.cd, dtype=dtype)
    sip = _cast_sip(wcs.sip, dtype)
    return leg_cls(crpix, cd, sip=sip)


@register_pytree_node_class
class Reprojector:
    """Fused pixel(``wcs_src``) -> pixel(``wcs_tgt``) reprojection.

    Parameters
    ----------
    wcs_src, wcs_tgt : WCSBase (NumPy-backed, e.g. Tan/Sin)
        See :func:`mapprojax.utils.compose_relative_rotation` -- must be
        NumPy-backed so the one-time rotation composition below runs in
        true float64 regardless of ``jax_enable_x64``.
    frame_rotation : array, shape (3, 3), optional
        See :func:`mapprojax.utils.compose_relative_rotation`.
    matrix_precision : {"f32", "f64"}
        Precision used for the composed rotation matrix *and* the
        per-pixel ``pix_to_native``/``native_to_pix`` legs:

        - ``"f32"`` (default): plain float32 throughout the per-pixel
          path, independent of ``jax_enable_x64``. Accurate to ~1e-4 px
          for overlapping fields (separations up to a few degrees at
          arcsecond plate scales); error grows with field-center
          separation and with finer plate scales -- see
          ``fo64_report_and_plan.md``.
        - ``"f64"``: current (pre-Phase-1) behavior -- everything in
          float64; requires ``jax_enable_x64`` to actually get float64
          arrays (otherwise JAX silently truncates to float32, same as
          everywhere else in this library).

    Note
    ----
    The plan (``fo64_report_and_plan.md``, Phase 1) also proposed a
    ``"f32x2"`` mode: split the composed matrix into a float32 hi/lo pair
    and apply it as ``M_hi @ v + M_lo @ v``, to recover accuracy at large
    separations/fine plate scales without full float64. Implemented and
    measured, this does *not* work as described: the ``hi``/``lo``
    products are summed with a single plain float32 add, and ``lo``'s own
    contribution (by construction, always below ``hi``'s float32 ULP at
    that magnitude) is rounded away by that add regardless of matrix
    precision -- confirmed identical output to plain ``"f32"`` (to the
    last bit) across every separation/plate-scale combination tested, and
    unchanged even when the two products are combined with a
    Kahan-compensated sum instead of a plain one (compensated summation
    prevents accumulated rounding *during* a sum of many terms; it cannot
    manufacture precision beyond what the final scalar's own dtype can
    represent). Recovering the ``lo`` correction genuinely requires
    carrying a float32 hi/lo pair as the *value* through the rest of the
    pipeline (``native_to_pix``'s own arithmetic included), not just the
    matrix -- i.e. real double-single (df64) arithmetic, which is this
    plan's separate, larger Phase 3, not a Phase-1-sized add-on. For
    separations/plate-scales where plain ``"f32"`` isn't accurate enough,
    use ``"f64"`` (with ``jax_enable_x64``) until Phase 3 exists.

    ``M_rel`` is precomputed once from ``wcs_src``/``wcs_tgt`` and is not
    part of the traced computation graph: differentiating ``pix_to_pix``
    with respect to either WCS's ``crval`` will not see the effect of that
    change (only ``crpix``, ``cd``, and SIP coefficients remain
    differentiable, since only those flow into the per-pixel legs). Build
    a fresh ``Reprojector`` if ``crval`` changes.
    """

    def __init__(self, wcs_src, wcs_tgt, frame_rotation=None, matrix_precision="f32"):
        if matrix_precision not in SUPPORTED_MATRIX_PRECISIONS:
            raise ValueError(f"matrix_precision must be one of {SUPPORTED_MATRIX_PRECISIONS}, got {matrix_precision!r}")

        self.matrix_precision = matrix_precision
        self._dtype = jnp.float64 if matrix_precision == "f64" else jnp.float32
        self._src_leg = _leg(wcs_src, self._dtype)
        self._tgt_leg = _leg(wcs_tgt, self._dtype)

        m_rel = compose_relative_rotation(wcs_src, wcs_tgt, frame_rotation=frame_rotation)
        self.m_rel = jnp.asarray(m_rel, dtype=self._dtype)

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
        x = jnp.asarray(x, dtype=self._dtype)
        y = jnp.asarray(y, dtype=self._dtype)

        xn, yn, zn = self._src_leg.pix_to_native(x, y)
        return apply_rotation(self.m_rel, xn, yn, zn)

    def pix_to_pix(self, x, y):
        """Map pixel coordinates in ``wcs_src`` to the corresponding pixel
        coordinates in ``wcs_tgt``.
        """
        xt, yt, zt = self.pix_to_native_tgt(x, y)
        return self._tgt_leg.native_to_pix(xt, yt, zt)

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
