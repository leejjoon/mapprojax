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
    there. Rotation errors depend on the field and inputs; this guard
    enforces the requested dtype rather than an angular-error bound.
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

    With x64 off, JAX converts requested float64 arrays to float32; this
    does not mean that arrays of other dtypes become float32. Check both
    the x64 flag and the dtype returned by ``jnp.asarray(value)``. Both
    checks are static, so this is safe under ``jit``.
    """
    return jax.config.jax_enable_x64 and jnp.asarray(value).dtype == jnp.float64
