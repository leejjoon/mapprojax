from jax.tree_util import register_pytree_node_class
from .projections import TanMixin, SinMixin
from .jax_base import WCSJax, WCSJaxArray, MixedPrecisionWCSJaxMixin, MixedPrecisionWCSJaxArrayMixin

# We must register each leaf class as a PyTree node because
# JAX requires explicit registration for subclasses.

@register_pytree_node_class
class TanJax(TanMixin, WCSJax):
    pass

@register_pytree_node_class
class TanJaxMixed(TanMixin, MixedPrecisionWCSJaxMixin, WCSJax):
    pass

@register_pytree_node_class
class TanJaxArray(TanMixin, WCSJaxArray):
    pass

@register_pytree_node_class
class TanJaxArrayMixed(TanMixin, MixedPrecisionWCSJaxArrayMixin, WCSJaxArray):
    pass

@register_pytree_node_class
class SinJax(SinMixin, WCSJax):
    pass

@register_pytree_node_class
class SinJaxMixed(SinMixin, MixedPrecisionWCSJaxMixin, WCSJax):
    pass

@register_pytree_node_class
class SinJaxArray(SinMixin, WCSJaxArray):
    pass

@register_pytree_node_class
class SinJaxArrayMixed(SinMixin, MixedPrecisionWCSJaxArrayMixin, WCSJaxArray):
    pass
