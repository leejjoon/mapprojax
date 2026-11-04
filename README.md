# mapprojax

`mapprojax` is a differentiable JAX coordinate kernel for FITS TAN and SIN WCS projections, including SIP distortion. It provides a precision-explicit PyTree API, projection-domain validity, offset coordinate methods, and source-to-target pixel maps. It does not parse arbitrary FITS WCS or interpolate image data: use Astropy for general WCS support and `reprojax` for image resampling.

## Install

```bash
pip install mapprojax
```

## Quick start

```python
import jax
import jax.numpy as jnp
from mapprojax import Tan

jax.config.update("jax_enable_x64", True)
wcs = Tan(
    crpix=[512.0, 512.0],
    cd=[[-0.001, 0.0], [0.0, 0.001]],
    crval=[180.0, 30.0],  # degrees
    precision="mixed",
)
lon, lat = wcs.unproj(jnp.array([500.0, 520.0]), jnp.array([500.0, 530.0]))
x, y = wcs.proj(lon, lat)
```

Pixels and `crpix` are 0-based. `crval` is in degrees; `proj` and `unproj` use radians in `(longitude, latitude)` order. TAN is invalid at and behind the horizon; SIN is valid on the front hemisphere and inside its inverse unit disc. Invalid samples produce NaNs.

## Precision modes

| Mode | Native/pixel path | Absolute celestial rotation | Requirement |
| --- | --- | --- | --- |
| `"f32"` | float32 | not available | Use `proj_offset` / `unproj_offset` for offset-space work. |
| `"mixed"` | float32 | float64 | JAX x64 enabled. |
| `"f64"` | float64 | float64 | JAX x64 enabled. |

Choose `mixed` for a lower-precision native path with absolute celestial coordinates, or `f64` for a fully float64 path. Accuracy depends on field geometry and plate scale; consult the measured limits in `docs/benchmarks/accuracy_results.md` and `docs/benchmarks/precision_results.md`.

Batch compatible WCS objects by stacking their PyTrees and using `jax.vmap`; all entries must share projection, precision, SIP shapes, and `inverse_steps`. `Reprojector(src, tgt)` computes a coordinate map only, not image resampling.

## Serialization and migration

Use the JSON-safe `to_dict()` and matching projection class's `from_dict()` methods. The contract preserves type, precision, reference dtype, and SIP state. ASDF integration and the old undocumented byte methods have been removed.

The old NumPy, `*Array`, `*Jax`, and `*Mixed` public class families have no compatibility aliases. Import `Tan` / `Sin` from `mapprojax` and specify a precision mode. Replace Array classes with stacked PyTrees and `jax.vmap`; replace Mixed classes with `precision="mixed"`. See [the user manual](docs/user_manual.md) for offset composition, SIP use, batching, serialization, and detailed migration notes.
