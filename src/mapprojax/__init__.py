import jax
import warnings

if not jax.config.jax_enable_x64:
    warnings.warn(
        "JAX 64-bit precision is not enabled. This may result in significant precision loss "
        "for WCS transformations. Consider running `jax.config.update('jax_enable_x64', True)` "
        "before importing mapprojax.",
        UserWarning
    )

from .base import WCSBase, WCSArrayBase
from .projections import Tan, TanArray, Sin, SinArray
from .jax_projections import (
    TanJax, TanJaxArray, TanJaxMixed, TanJaxArrayMixed,
    SinJax, SinJaxArray, SinJaxMixed, SinJaxArrayMixed
)

__all__ = [
    'Tan', 'TanArray', 'Sin', 'SinArray', 
    'TanJax', 'TanJaxArray', 'TanJaxMixed', 'TanJaxArrayMixed',
    'SinJax', 'SinJaxArray', 'SinJaxMixed', 'SinJaxArrayMixed',
    'WCSBase', 'WCSArrayBase'
]
