"""JAX precision-policy utilities and the PyTree registration for ``Sip``."""

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

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
