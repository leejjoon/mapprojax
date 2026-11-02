import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node, register_pytree_node_class

from .base import WCSBase, WCSArrayBase, _check_cd
from .sip import Sip

# Register Sip as a PyTree node so it can be used in JAX functions
def _sip_flatten(sip):
    # children: the arrays (or None). inverse_steps sets how many Newton
    # steps are unrolled, so it must be static: aux data, part of the jit
    # cache key (a different step count compiles a different program).
    children = (sip.a, sip.b, sip.ap, sip.bp)
    aux_data = sip.inverse_steps
    return (children, aux_data)

def _sip_unflatten(aux_data, children):
    return Sip(a=children[0], b=children[1], ap=children[2], bp=children[3],
               inverse_steps=aux_data)

register_pytree_node(Sip, _sip_flatten, _sip_unflatten)


def _require_x64(what):
    """Raise when x64 is off, before any cast silently truncates.

    ``jnp.asarray(x, dtype=jnp.float64)`` does not fail without x64; it
    returns float32 (JAX only emits a generic truncation warning). A class
    whose contract is "rotation in float64" cannot honour that contract
    there, and the measured error is large enough to matter: 93 mas on an
    ordinary field, 107 mas on the L2 header.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError(
            f"{what} requires float64. Call "
            "jax.config.update('jax_enable_x64', True) before constructing "
            "it. Without it every jnp.float64 array is float32, and the "
            "celestial rotation loses about 100 mas."
        )


def _effective_f64(value):
    """True when ``value`` really is float64, not just nominally cast.

    With x64 off every jnp array is float32 regardless of the requested
    dtype, and a float64 leaf built while x64 was on is truncated by any
    operation once it is off, so both the dtype and the flag matter. Both
    are static, so this is safe under ``jit``.
    """
    return jax.config.jax_enable_x64 and jnp.asarray(value).dtype == jnp.float64


class _JaxRotationPrecision:
    """The celestial rotation of a JAX WCS must run in float64.

    Checked on every component of the reference coordinate (``crval``, or
    both ``crvals`` arrays), at the one place the rotation is applied
    (``_celestial_to_native`` / ``_native_to_celestial``), so ``proj``,
    ``unproj`` and the ``*_xyz`` forms all raise while construction,
    ``pix_to_native``/``native_to_pix`` and the offset APIs keep working.
    """

    def _reference_coords(self):
        raise NotImplementedError

    def _check_rotation_precision(self):
        if not all(_effective_f64(c) for c in self._reference_coords()):
            raise TypeError(
                "proj/unproj on a JAX-backed WCS need a float64 reference "
                "coordinate: the celestial rotation loses about 100 mas in "
                "float32. Call jax.config.update('jax_enable_x64', True) "
                "before building the WCS, or use unproj_offset/proj_offset, "
                "which stay in the offset regime."
            )


@register_pytree_node_class
class WCSJax(_JaxRotationPrecision, WCSBase):
    xp = jnp

    def tree_flatten(self):
        children = (self.crpix, self.cd, self.crval, self.sip)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # children corresponds to (crpix, cd, crval, sip). Reconstruct via
        # __new__: calling __init__ would re-cast dtypes (breaking f32 leaves
        # and `eval_shape`'s ShapeDtypeStruct), and `r_matrix` is now a derived
        # property, so there is nothing to rebuild.
        obj = cls.__new__(cls)
        obj.crpix, obj.cd, obj.crval, obj.sip = children
        return obj

    def _reference_coords(self):
        return (self.crval,)


@register_pytree_node_class
class WCSJaxArray(_JaxRotationPrecision, WCSArrayBase):
    xp = jnp

    def tree_flatten(self):
        # WCSArrayBase stores crvals as tuple (ra_arr, dec_arr)
        children = (self.crpix, self.cd, self.crvals, self.sip)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # See WCSJax.tree_unflatten: children is (crpix, cd, crvals, sip).
        obj = cls.__new__(cls)
        obj.crpix, obj.cd, obj.crvals, obj.sip = children
        return obj

    def _reference_coords(self):
        return tuple(self.crvals)


class _MixedRotation:
    """fp64 rotation around an fp32 native/pixel pipeline.

    Only the two rotation hooks change; validity handling, masking and the
    precision check are inherited, so every public method (including
    ``unproj_xyz`` on the array variant) goes through the same guard.
    """

    def _celestial_to_native(self, x, y, z):
        xn, yn, zn = super()._celestial_to_native(x, y, z)  # f64 rotation
        return xn.astype(jnp.float32), yn.astype(jnp.float32), zn.astype(jnp.float32)

    def _native_to_celestial(self, xn, yn, zn):
        return super()._native_to_celestial(
            xn.astype(jnp.float64), yn.astype(jnp.float64), zn.astype(jnp.float64))


class MixedPrecisionWCSJaxMixin(_MixedRotation):
    def __init__(self, crpix, cd, crval, sip=None):
        _require_x64("MixedPrecisionWCSJaxMixin (TanJaxMixed/SinJaxMixed)")
        self.crpix = self.xp.array(crpix, dtype=jnp.float32)
        self.cd = self.xp.array(cd, dtype=jnp.float32)
        self.crval = self.xp.array(crval, dtype=jnp.float64)
        self.sip = sip
        _check_cd(self.cd)


class MixedPrecisionWCSJaxArrayMixin(_MixedRotation):
    def __init__(self, crpix, cd, crvals, sip=None):
        _require_x64("MixedPrecisionWCSJaxArrayMixin (TanJaxArrayMixed/SinJaxArrayMixed)")
        self.crpix = self.xp.array(crpix, dtype=jnp.float32)
        self.cd = self.xp.array(cd, dtype=jnp.float32)
        self.sip = sip
        _check_cd(self.cd)

        ra_arr = self.xp.asarray(crvals[0], dtype=jnp.float64)
        dec_arr = self.xp.asarray(crvals[1], dtype=jnp.float64)
        self.crvals = (ra_arr, dec_arr)
