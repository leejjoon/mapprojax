# Developer Guide

`mapprojax` is a JAX coordinate kernel for FITS TAN and SIN projections with SIP distortion. It owns projection-domain validity, the pixel/native/celestial transforms, precision policy, and coordinate-only reprojection. It does not own general FITS WCS parsing or image interpolation; use Astropy for broad WCS support and `reprojax` for image resampling.

## Architecture

- `src/mapprojax/wcs.py` defines the public `Tan` and `Sin` classes. Each is a JAX PyTree with an explicit `precision` (`"f32"`, `"mixed"`, or `"f64"`). There are no separate NumPy, Array, or Mixed class families.
- `src/mapprojax/projections.py` contains the backend-independent TAN/SIN formula mixins used by the precision-explicit WCS classes.
- `src/mapprojax/base.py` contains the shared validity-aware pixel/native pipeline and host offset-composition helpers.
- `src/mapprojax/jax_base.py` contains JAX precision-policy utilities and the PyTree registration for `Sip`.
- `src/mapprojax/sip.py` implements SIP forward and inverse distortion. Its coefficient shapes and `inverse_steps` are static PyTree structure/configuration for batching and compilation.
- `src/mapprojax/reproject.py` composes two conforming projection pipelines into a coordinate map. It does not resample data.

The public root exports are `Tan`, `Sin`, `Reprojector`, `absolute_from_offset`, `offset_from_absolute`, and `compose_relative_rotation`. Do not add aliases for retired APIs.

## Precision contract

| Mode | Native and pixel math | Celestial rotation | Absolute `proj` / `unproj` |
| --- | --- | --- | --- |
| `f32` | float32 | unavailable | No; use the offset methods and host helper composition. |
| `mixed` | float32 | float64 | Yes; JAX x64 must be enabled. |
| `f64` | float64 | float64 | Yes; JAX x64 must be enabled. |

`crval` is retained at the supplied reference dtype. Do not assume that casting a truncated reference back to float64 restores its information. Host composition of the reference rotation uses concrete float64 references; it must reject tracers and non-float64 references. Precision trade-offs are workload-dependent; measured envelopes are maintained in `docs/benchmarks/accuracy_results.md` and `docs/benchmarks/precision_results.md`.

All pixel coordinates are 0-based. `crval` is in degrees; projection methods take and return radians in `(longitude, latitude)` order. TAN excludes the horizon and back hemisphere; SIN includes the front-hemisphere boundary and its inverse includes the unit-disc boundary. Public projection outputs are NaN when invalid; `Reprojector.pixel_map` instead returns finite zero stand-ins and a separate boolean validity mask. Internal validity flags keep invalid values from poisoning downstream arithmetic or gradients.

## Public API patterns

Construct a projection with an explicit precision mode:

```python
import jax
from mapprojax import Tan

jax.config.update("jax_enable_x64", True)
wcs = Tan(crpix, cd, crval, precision="mixed")
lon, lat = wcs.unproj(x, y)
```

Batch compatible WCS PyTrees with `jax.tree_util.tree_map` and `jax.vmap`, rather than adding per-centre `*Array` classes. A batch must share projection type, precision, PyTree structure, SIP coefficient shapes, and `inverse_steps`. For many WCS objects over the same pixel grid, use `jax.vmap(lambda w: w.unproj(x, y), in_axes=0, out_axes=1)` to return `(pixels, centres)`.

For true absolute coordinates from the offset path, compose the producer-typed offsets on the host:

```python
from mapprojax import absolute_from_offset

offset = wcs.unproj_offset(x, y)
absolute = absolute_from_offset(offset, crval=wcs.crval, offset_scale=wcs.offset_scale())
```

This helper is not JIT- or autodiff-compatible. `crval` and `offset_scale` are keyword-only. Preserve the offset component and scale dtypes returned by the producer.

Use `Reprojector(src, tgt)` for source-to-target pixel coordinates. Its `matrix_precision="f32"` mode removes fp64 from the per-pixel path; the host-composed rotation is not differentiable with respect to the two WCS `crval` values. When gradients with respect to `crval` are required, build the relative matrix from the live references inside the traced function and pass it to `Reprojector.from_matrix`. After a transformation that leaves `crval` traced or float32, compose from retained host references with `compose_relative_rotation(src, tgt, crval_src=..., crval_tgt=...)` and pass the result to `from_matrix`.

`pixel_map(x, y, *, bounds_shape=None)` returns mapped coordinates and a validity mask. Coordinates are finite and zero where invalid; a supplied `(ny, nx)` bounds tuple additionally applies the half-open target bounds `[-0.5, nx-0.5) × [-0.5, ny-0.5)`. The tuple must be static and contain two positive built-in Python integers. Use `jnp.where` to mask before nonlinear losses, and choose regular stand-ins before downstream computations that are unsafe at invalid coordinates. `pix_to_pix` is unchanged and keeps its NaN invalid-output contract. See the [user manual](docs/user_manual.md#differentiability-contract) for the complete differentiation table, including the distinction between eager composition and `from_matrix`.

Serialization uses the tested JSON-safe `to_dict()` / `from_dict()` contract. It preserves projection type, precision, reference dtype, and SIP state and rejects mismatched types or legacy dictionaries. ASDF support and the undocumented byte-serialization claims have been retired.

## Validation and changes

Use the focused tests while iterating, followed by the project test suite on CPU and the supported ROCm setup before release. Independent Astropy agreement is the projection-correctness check; do not replace it with only a mapprojax round trip. The regression suite also covers domain edges, masked gradients, PyTree transforms, SIP inverse behavior, CD validation, and JSON restoration.

When changing numerical behavior, update the accuracy harness/baselines and regenerate the corresponding results using the existing scripts in `benchmarks/`. Do not change projection formulas without independent reference coverage.
