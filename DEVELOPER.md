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

## 4. Precision Strategy: Why fp64 (and Where fp32 Is Safe)

The pipeline mixes fp64 and fp32 deliberately. The rule is:

> **fp64 is required for anything carrying an *absolute* position on the celestial sphere** — `crval`, the rotation matrix built from it, the celestial↔native rotation (`apply_rotation` / `apply_rotation_transpose` in `utils.py`), and RA/Dec values themselves. **Everything on the other side of that rotation is safe in fp32** — the TAN/SIN projection formulas, the CD matrix, SIP polynomials, and `crpix` offsets — because those operate on coordinates *relative* to the projection center.

The `MixedPrecisionWCSJaxMixin` / `MixedPrecisionWCSJaxArrayMixin` classes in `jax_base.py` implement exactly this split.

### The dynamic-range argument

Floating point has *relative* precision: fp32 gives ~7 significant digits (ε ≈ 1.2e-7) regardless of magnitude. What matters is the ratio between the magnitude of stored values and the absolute accuracy required.

*   **Absolute sky coordinates** span the full sphere: unit-vector components are O(1), RA spans 0–2π. Sub-pixel astrometry needs absolute angular accuracy of ~1e-8–1e-9 rad (0.001–0.01″), i.e. a relative precision of ~1e-8 on an O(1) quantity — *below* fp32's ε. At best, fp32 quantizes a unit-vector direction to ~1.2e-7 rad ≈ 0.025″, and RA near 2π to ~0.15″. At a 3.6″/px plate scale that is a ~0.01 px error; at a modern 0.1″/px scale the same absolute error becomes ~0.4 px — unusable.
*   **Native-frame coordinates** are offsets from the projection center: plane coordinates are field-of-view-sized (≤1e-2 rad), pixel offsets from `crpix` are ≤1e4. Because float precision is relative, small numbers carry proportionally small *absolute* error: a 1e-4 rad native offset in fp32 has ~1e-11 rad of error; a 1e4-pixel offset has ~1e-3 px. Both are far below sub-pixel requirements.

### The mechanism: catastrophic cancellation in the rotation

The celestial→native rotation is where "large absolute" becomes "small relative". In `apply_rotation`:

```python
yn = m10*x + m11*y + m12*z
```

Near the field center, the O(1) products nearly cancel to yield a tiny result (`yn`, `zn` ≈ field offsets, e.g. 1e-4 rad). In fp32 each product carries an *absolute* rounding error of ~1e-7, and those errors do not cancel with the values — so the tiny outputs inherit ~1e-7 rad of absolute error, i.e. ~0.1% of the entire field of view. Essentially all of the fp32 damage is injected in this one step (and its transpose on the inverse path).

### Numerical evidence

TAN round-trip (`unproj` → `proj`) on a 200×200 grid at 3.6″/px, varying which stage runs in fp32:

| Configuration | Max round-trip error |
|---|---|
| all fp64 | 2.7e-11 px |
| **rotation fp64, everything else fp32 (the Mixed classes)** | **7.6e-6 px** |
| rotation fp32, everything else fp64 | 1.076e-2 px |
| all fp32 | 1.077e-2 px |
| all fp64 but RA/Dec quantized to fp32 | 1.7e-3 px |

Casting *only* the rotation to fp32 reproduces essentially 100% of the all-fp32 error, while keeping *only* the rotation in fp64 recovers nearly all the accuracy.

### How the mixed classes exploit this

`MixedPrecisionWCSJaxMixin` (`jax_base.py`):

*   Stores `crval` in fp64 and precomputes `r_matrix` in fp64; `crpix`/`cd` are fp32.
*   `proj_xyz`: rotates in fp64, **then** casts the native vector down to fp32 for the plane/pixel math — safe, because at that point the components are relative quantities.
*   `unproj_xyz`: runs pixel→native in fp32, casts **up** to fp64 *before* the inverse rotation. The upcast adds no information to the inputs, but it ensures the O(1) celestial outputs (which need ~1e-9 absolute accuracy) are not re-quantized by an fp32 rotation.

One subtlety makes the fp32 native path work: the vector component along the boresight (e.g. `x = sqrt(1 - r²)` in `SinMixin._plane_to_native`) sits near 1.0 and does pick up ~6e-8 of fp32 error — but that error is *radial* (it perturbs the vector's length, not its direction) and is normalized away in `xyz_to_radec`, which divides by the vector norm. Directional accuracy is carried by the small transverse components, which fp32 stores well.

The payoff is performance: fp32 halves memory bandwidth and is substantially faster on GPU/TPU (where fp64 throughput is often 1/32–1/64 of fp32), while the fp64 portion is just a 3×3 matrix–vector multiply per pixel — so the mixed classes deliver fp64-grade astrometry at close to fp32 cost.

### Beyond the mixed classes: removing fp64 from the per-pixel path entirely

The mixed classes still keep the celestial↔native rotation itself in fp64 — cheap (one 3×3 matmul per pixel), but it means `jax_enable_x64` must be on, and every per-pixel op after the upcast still touches an fp64 value. `fo64_report_and_plan.md` asked whether fp64 can be removed *entirely* from the per-pixel path, and worked through two of its four phases; this section is the decision tree for what to reach for. Full research/derivation/validation is in that document; measured numbers are in `docs/benchmarks/precision_results.md`.

**Decision tree:**

| Need | Use | Why |
|---|---|---|
| Reproject a pixel from one WCS to the corresponding pixel in another (the `reproject`-style use case) | **`mapprojax.reproject.Reprojector`** (Phase 1), `matrix_precision="f32"` (default) | Composes both WCSes' rotations into one matrix, once, in true float64 on the host — independent of `jax_enable_x64` — then runs the entire per-pixel path (`pix_to_native → apply that matrix → native_to_pix`) in float32. Fastest mode measured on both CPU and GPU (§ below); accurate to ~1e-3–1e-2 px for overlapping fields at typical plate scales, degrading with field-center separation and finer plate scales — see its docstring before trusting it for a wide-separation or sub-arcsecond-plate-scale pair. |
| True absolute (RA, Dec) output from a single WCS, without requiring `jax_enable_x64` | **`WCSBase.unproj_offset`/`unproj_absolute`** (and `proj_offset`/`proj_absolute` for the inverse) (Phase 2) | Computes a small, field-of-view-sized sky offset from `crval` via spherical trig (never forming an O(1) RA/Dec value in the bulk math, so fp32 is safe), then adds it to `crval` in a cheap float64 epilogue. Requires a NumPy-backed WCS (`Tan`/`Sin`) for that epilogue — a JAX WCS's `crval` may already be truncated to float32 at construction time if `jax_enable_x64` was off then, and this can't retroactively fix that. |
| Differentiate through the rotation with respect to `crval` itself, or need the full per-pixel `unproj_xyz`/`proj_xyz` (not a fused src→tgt map) on a datacenter GPU or CPU where fp64 is cheap (≤2x fp32, not 32–64x) | **`TanJaxMixed`/`SinJaxMixed`** (or the plain fp64 classes) | `Reprojector`'s rotation is precomputed outside JAX's trace, so it is *not* differentiable with respect to either WCS's `crval` — only `crpix`/`cd`/SIP coefficients remain differentiable through it. The mixed classes' rotation stays inside the trace. |
| Anything above, but need to squeeze out fp64's residual quantization error at large separations or fine plate scales | *(nothing shipped)* | Phase 1's plan proposed a `matrix_precision="f32x2"` (hi/lo split matrix) for exactly this. Implemented and measured: it does not work — the `lo` correction is by construction smaller than the `hi` result's own float32 ULP, so a plain (or even Kahan-compensated) float32 add rounds it away regardless of matrix precision. Recovering it needs the value itself carried as a float32 hi/lo pair through the *entire* pipeline (true double-single/df64 arithmetic), which is Phase 3 — not implemented (only worth building if a concrete need appears; see the plan's own caveat about XLA reassociation breaking compensated-arithmetic assumptions, which is why JAX dropped its own `doubledouble` experiment). Until then, use the plain fp64 classes for this case. |
| Never | Tensor-core fp64 emulation (bf16x9, Ozaki/cuBLAS math modes) | Ruled out in the plan's research: wrong accuracy target (bf16x9) or GEMM-only and slower than native at this operation's problem size (a memory-bound per-pixel 3×3 rotation, not a GEMM). |

## 5. Extending the Package

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
