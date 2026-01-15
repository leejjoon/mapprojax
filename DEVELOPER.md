# Developer Guide & Architecture Overview

`mapprojax` is a Python library designed to provide fast, differentiable WCS (World Coordinate System) projections using [JAX](https://github.com/google/jax), while maintaining full compatibility with [NumPy](https://numpy.org/).

This document outlines the internal architecture and provides a guide for developers contributing to or extending the package.

## 1. Architecture Overview

### Design Philosophy: Backend Agnosticism
The core philosophy of `mapprojax` is to implement the mathematical logic of projections once and run it on multiple backends (NumPy for standard CPU usage, JAX for GPU acceleration and automatic differentiation).

This is achieved by:
1.  **Abstracting the Array Library**: Classes use `self.xp` (or pass `xp` arguments) which defaults to `numpy` but can be swapped for `jax.numpy`.
2.  **Mixins for Logic**: The mathematical equations for projections (e.g., Gnomonic/TAN, Orthographic/SIN) are implemented in independent "Mixin" classes that do not depend on the specific WCS machinery.
3.  **Composition**: Concrete classes differ only in their base class (`WCSBase` vs `WCSJax`) and the Mixin they use.

### Class Hierarchy

The system is built on a modular hierarchy:

*   **`Sip` (`src/mapprojax/sip.py`)**:
    *   Handles SIP (Simple Imaging Polynomial) distortion.
    *   Backend-agnostic; methods accept an `xp` argument.
    *   Registered as a JAX PyTree node to support differentiation through distortion coefficients.

*   **Base Classes (`src/mapprojax/base.py`)**:
    *   **`WCSBase`**: The standard NumPy implementation. Handles the WCS pipeline:
        `Pixel -> SIP (Distortion) -> Linear (CD Matrix) -> Projection (Plane to Native) -> Rotation (Native to Celestial)`.
    *   **`WCSArrayBase`**: Optimized version for handling arrays of WCS parameters (e.g., different CRVALs for every pixel or batch).

*   **JAX Base Classes (`src/mapprojax/jax_base.py`)**:
    *   **`WCSJax`**: Inherits from `WCSBase` but sets `xp = jax.numpy`.
    *   **PyTree Registration**: These classes are decorated with `@register_pytree_node_class`, allowing instances to be passed seamlessly into JIT-compiled JAX functions.

*   **Projection Mixins (`src/mapprojax/projections.py`)**:
    *   **`TanMixin`, `SinMixin`**: Implement `_native_to_plane` and `_plane_to_native`. These are pure math functions using `self.xp`.

### Concrete Implementations

*   **NumPy**: `Tan(TanMixin, WCSBase)`
*   **JAX**: `TanJax(TanMixin, WCSJax)`

## 2. Directory Structure

```text
src/mapprojax/
├── base.py             # NumPy base classes (WCSBase)
├── sip.py              # SIP distortion implementation
├── utils.py            # Math utilities (rotation matrices, spherical conversions)
├── projections.py      # Projection Logic Mixins + NumPy concrete classes
├── jax_base.py         # JAX base classes + PyTree registration
└── jax_projections.py  # JAX concrete classes
```

## 3. Usage Guide

### Installation

```bash
pip install .
# or
pip install -e .  # for development
```

### Basic Usage (NumPy)

Use the standard classes for CPU-based calculations, similar to `astropy.wcs`.

```python
import numpy as np
from mapprojax.projections import Tan

# Define WCS parameters
crpix = [50, 50]
cd = [[-0.00028, 0], [0, 0.00028]] # ~1 arcsec/pix
crval = [180.0, 45.0]

wcs = Tan(crpix, cd, crval)

# Pixel to Sky
x, y = np.array([10.0, 90.0]), np.array([10.0, 90.0])
ra, dec = wcs.unproj(x, y) # Returns radians
print(np.degrees(ra), np.degrees(dec))

# Sky to Pixel
x_back, y_back = wcs.proj(ra, dec)
```

### Basic Usage (JAX)

Use the `Jax` suffixed classes. These can be JIT-compiled or differentiated.

```python
import jax
import jax.numpy as jnp
from mapprojax.jax_projections import TanJax

# Enable 64-bit precision for astronomy
jax.config.update("jax_enable_x64", True)

wcs = TanJax(crpix, cd, crval)

@jax.jit
def process_coords(x, y):
    ra, dec = wcs.unproj(x, y)
    return ra, dec

x_gpu = jnp.array([10.0, 90.0])
y_gpu = jnp.array([10.0, 90.0])

ra, dec = process_coords(x_gpu, y_gpu)
```

### Using SIP Distortion

SIP coefficients are passed via the `Sip` object.

```python
from mapprojax.sip import Sip

# Define distortion coefficients (A_2_0, B_0_2, etc.)
a = np.zeros((3, 3))
a[2, 0] = 1.5e-6

sip = Sip(a=a) # a, b for forward; ap, bp for reverse

wcs = Tan(crpix, cd, crval, sip=sip)

# Calculations automatically include distortion
ra, dec = wcs.unproj(x, y)
```

## 4. Extending the Package

### Adding a New Projection
To add a new projection (e.g., Stereographic `STG`):

1.  **Create the Logic Mixin** in `src/mapprojax/projections.py`:
    ```python
    class StgMixin:
        def _native_to_plane(self, x, y, z):
            # Implement formula using self.xp
            return 2 * y / (1 + x), 2 * z / (1 + x)

        def _plane_to_native(self, X, Y):
            # Implement inverse
            ...
    ```

2.  **Create the NumPy Class** in `src/mapprojax/projections.py`:
    ```python
    class Stg(StgMixin, WCSBase):
        pass
    ```

3.  **Create the JAX Class** in `src/mapprojax/jax_projections.py`:
    ```python
    @register_pytree_node_class
    class StgJax(StgMixin, WCSJax):
        pass
    ```

### Running Tests
Run the test suite using `pytest`:

```bash
pytest tests/
```
