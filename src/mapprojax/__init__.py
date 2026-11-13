from .base import (
    absolute_from_offset, offset_from_absolute,
)
from .wcs import Tan, Sin
from .reproject import Reprojector
from .utils import compose_relative_rotation

__all__ = [
    'Tan', 'Sin',
    'absolute_from_offset', 'offset_from_absolute',
    'Reprojector', 'compose_relative_rotation',
    'from_astropy',
]


def __getattr__(name):
    # PEP 562: `from_astropy` lives in `mapprojax.astropy`, whose module body
    # is astropy-free, but importing it here keeps the documented single
    # import surface. Deferring the import means `import mapprojax` never pays
    # for astropy -- and never fails without it.
    if name == "from_astropy":
        from .astropy import from_astropy

        globals()["from_astropy"] = from_astropy
        return from_astropy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # A plain `sorted(__all__)` would hide submodules and dunders from
    # `dir(mapprojax)`, which makes introspection dishonest.
    return sorted(set(globals()) | set(__all__))
