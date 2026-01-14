import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node, register_pytree_node_class

from .base import WCSBase, WCSArrayBase
from .utils import rotation_matrix, apply_rotation, apply_rotation_transpose
from .sip import Sip

# Register Sip as a PyTree node so it can be used in JAX functions
def _sip_flatten(sip):
    # children: the arrays (or None)
    children = (sip.a, sip.b, sip.ap, sip.bp)
    aux_data = None
    return (children, aux_data)

def _sip_unflatten(aux_data, children):
    return Sip(a=children[0], b=children[1], ap=children[2], bp=children[3])

register_pytree_node(Sip, _sip_flatten, _sip_unflatten)


@register_pytree_node_class
class WCSJax(WCSBase):
    xp = jnp
    
    def tree_flatten(self):
        children = (self.crpix, self.cd, self.crval, self.sip)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # children corresponds to (crpix, cd, crval, sip)
        return cls(*children)

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
        return cls(*children)

class MixedPrecisionWCSJaxMixin:
    def __init__(self, crpix, cd, crval, sip=None):
        self.crpix = self.xp.array(crpix, dtype=jnp.float32)
        self.cd = self.xp.array(cd, dtype=jnp.float32)
        self.crval = self.xp.array(crval, dtype=jnp.float64)
        self.sip = sip
        
        # Precompute rotation matrix in f64
        self.r_matrix = rotation_matrix(
            self.xp.radians(self.crval[0]), 
            self.xp.radians(self.crval[1]), 
            xp=self.xp
        )

    def proj_xyz(self, x, y, z):
        # 1. Celestial -> Native (Rotation in f64)
        xn, yn, zn = apply_rotation(self.r_matrix, x, y, z)
        
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
        return apply_rotation_transpose(self.r_matrix, xn, yn, zn)

class MixedPrecisionWCSJaxArrayMixin:
    def __init__(self, crpix, cd, crvals, sip=None):
        self.crpix = self.xp.array(crpix, dtype=jnp.float32)
        self.cd = self.xp.array(cd, dtype=jnp.float32)
        self.sip = sip
        
        ra_arr, dec_arr = self.xp.asarray(crvals[0], dtype=jnp.float64), self.xp.asarray(crvals[1], dtype=jnp.float64)
        self.crvals = (ra_arr, dec_arr)
        
        # Precompute rotation matrices in f64
        self.r_matrices = rotation_matrix(
            self.xp.radians(ra_arr), 
            self.xp.radians(dec_arr), 
            xp=self.xp
        )

    def proj_xyz(self, x, y, z):
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