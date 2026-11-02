import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node, register_pytree_node_class

from .base import WCSBase, WCSArrayBase
from .utils import rotation_matrix, apply_rotation, apply_rotation_transpose
from .sip import Sip

# Register Sip as a PyTree node so it can be used in JAX functions
def _sip_flatten(sip):
    # children: the arrays (or None). inverse_steps is static, so it is aux
    # data -- a leaf would make jit recompile on every step-count change.
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
            "jax.config.update('jax_enable_x64', True) before importing "
            "mapprojax. Without it every jnp.float64 array is silently "
            "float32, and the celestial rotation loses about 100 mas."
        )


def _effective_f64(value):
    """True when ``value`` really is float64, not just nominally cast.

    With x64 off every jnp array is float32 regardless of the requested
    dtype, so the stored dtype alone is not enough. This is evaluated on the
    array's static dtype, which is available under ``jit``.
    """
    return jnp.asarray(value).dtype == jnp.float64


@register_pytree_node_class
class WCSJax(WCSBase):
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

    def _check_rotation_precision(self, crval):
        """Raise when the rotation cannot actually run in float64.

        The test is on the effective precision of the stored ``crval``, not
        on ``jax_enable_x64``: a WCS built with x64 on keeps f64 leaves
        after x64 is switched off, but every later operation truncates. The
        dtype is static, so this is safe under ``jit``.
        """
        if not _effective_f64(crval):
            raise TypeError(
                "proj/unproj on a JAX-backed WCS need a float64 reference "
                "coordinate: the celestial rotation loses about 100 mas in "
                "float32. Call jax.config.update('jax_enable_x64', True) "
                "before building the WCS, or use unproj_offset/proj_offset, "
                "which stay in the offset regime."
            )

    def proj_xyz(self, x, y, z):
        self._check_rotation_precision(self.crval)
        return super().proj_xyz(x, y, z)

    def unproj_xyz(self, x, y):
        self._check_rotation_precision(self.crval)
        return super().unproj_xyz(x, y)

@register_pytree_node_class
class WCSJaxArray(WCSArrayBase):
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

    def _check_rotation_precision(self, crval):
        """Raise when the rotation cannot actually run in float64.

        The test is on the effective precision of the stored ``crval``, not
        on ``jax_enable_x64``: a WCS built with x64 on keeps f64 leaves
        after x64 is switched off, but every later operation truncates. The
        dtype is static, so this is safe under ``jit``.
        """
        if not _effective_f64(crval):
            raise TypeError(
                "proj/unproj on a JAX-backed WCS need a float64 reference "
                "coordinate: the celestial rotation loses about 100 mas in "
                "float32. Call jax.config.update('jax_enable_x64', True) "
                "before building the WCS, or use unproj_offset/proj_offset, "
                "which stay in the offset regime."
            )

    def proj_xyz(self, x, y, z):
        self._check_rotation_precision(self.crvals[0])
        return super().proj_xyz(x, y, z)

    def unproj_xyz(self, x, y):
        self._check_rotation_precision(self.crvals[0])
        return super().unproj_xyz(x, y)

class MixedPrecisionWCSJaxMixin:
    def _check_rotation_precision(self, crval):
        """Raise when the rotation cannot actually run in float64.

        The test is on the effective precision of the stored ``crval``, not
        on ``jax_enable_x64``: a WCS built with x64 on keeps f64 leaves
        after x64 is switched off, but every later operation truncates. The
        dtype is static, so this is safe under ``jit``.
        """
        if not _effective_f64(crval):
            raise TypeError(
                "proj/unproj on a JAX-backed WCS need a float64 reference "
                "coordinate: the celestial rotation loses about 100 mas in "
                "float32. Call jax.config.update('jax_enable_x64', True) "
                "before building the WCS, or use unproj_offset/proj_offset, "
                "which stay in the offset regime."
            )

    def __init__(self, crpix, cd, crval, sip=None):
        _require_x64("MixedPrecisionWCSJaxMixin (TanJaxMixed/SinJaxMixed)")
        self.crpix = self.xp.array(crpix, dtype=jnp.float32)
        self.cd = self.xp.array(cd, dtype=jnp.float32)
        self.crval = self.xp.array(crval, dtype=jnp.float64)
        self.sip = sip

    def proj_xyz(self, x, y, z):
        self._check_rotation_precision(self.crval)
        # 1. Celestial -> Native (Rotation in f64)
        xn, yn, zn = apply_rotation(self.r_matrix, x, y, z)
        
        # 2. Cast to f32 for native space and pixel mapping
        xn = xn.astype(jnp.float32)
        yn = yn.astype(jnp.float32)
        zn = zn.astype(jnp.float32)
        
        return self.native_to_pix(xn, yn, zn)

    def unproj_xyz(self, x, y):
        self._check_rotation_precision(self.crval)
        # 1. Pixel -> Native (f32)
        xn, yn, zn = self.pix_to_native(x, y)
        
        # 2. Cast to f64 for celestial rotation
        xn = xn.astype(jnp.float64)
        yn = yn.astype(jnp.float64)
        zn = zn.astype(jnp.float64)
        
        # 3. Native -> Celestial (Inverse Rotation in f64)
        return apply_rotation_transpose(self.r_matrix, xn, yn, zn)

class MixedPrecisionWCSJaxArrayMixin:
    def _check_rotation_precision(self, crval):
        """Raise when the rotation cannot actually run in float64.

        The test is on the effective precision of the stored ``crval``, not
        on ``jax_enable_x64``: a WCS built with x64 on keeps f64 leaves
        after x64 is switched off, but every later operation truncates. The
        dtype is static, so this is safe under ``jit``.
        """
        if not _effective_f64(crval):
            raise TypeError(
                "proj/unproj on a JAX-backed WCS need a float64 reference "
                "coordinate: the celestial rotation loses about 100 mas in "
                "float32. Call jax.config.update('jax_enable_x64', True) "
                "before building the WCS, or use unproj_offset/proj_offset, "
                "which stay in the offset regime."
            )

    def __init__(self, crpix, cd, crvals, sip=None):
        _require_x64("MixedPrecisionWCSJaxArrayMixin (TanJaxArrayMixed/SinJaxArrayMixed)")
        self.crpix = self.xp.array(crpix, dtype=jnp.float32)
        self.cd = self.xp.array(cd, dtype=jnp.float32)
        self.sip = sip
        
        ra_arr, dec_arr = self.xp.asarray(crvals[0], dtype=jnp.float64), self.xp.asarray(crvals[1], dtype=jnp.float64)
        self.crvals = (ra_arr, dec_arr)

    def proj_xyz(self, x, y, z):
        self._check_rotation_precision(self.crvals[0])
        # 1. Celestial -> Native (Rotation in f64)
        xn, yn, zn = apply_rotation(self.r_matrices, x, y, z)
        
        # 2. Cast to f32 for native space and pixel mapping
        xn = xn.astype(jnp.float32)
        yn = yn.astype(jnp.float32)
        zn = zn.astype(jnp.float32)
        
        return self.native_to_pix(xn, yn, zn)

    def unproj_xyz(self, x, y):
        # 1. Pixel -> Native (f32)
        xn, yn, zn = self.pix_to_native(x, y)
        
        # 2. Cast to f64 for celestial rotation
        xn = xn.astype(jnp.float64)
        yn = yn.astype(jnp.float64)
        zn = zn.astype(jnp.float64)
        
        # 3. Native -> Celestial (Inverse Rotation in f64)
        return apply_rotation_transpose(self.r_matrices, xn, yn, zn)