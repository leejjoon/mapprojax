import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .base import WCSBase, WCSArrayBase

@register_pytree_node_class
class WCSJax(WCSBase):
    xp = jnp
    
    def tree_flatten(self):
        children = (self.crpix, self.cd, self.crval)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class WCSJaxArray(WCSArrayBase):
    xp = jnp

    def tree_flatten(self):
        # WCSArrayBase stores crvals as tuple (ra_arr, dec_arr)
        children = (self.crpix, self.cd, self.crvals)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
