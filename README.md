# mapprojax

`mapprojax` is a differentiable JAX coordinate kernel for FITS TAN and SIN WCS projections, including SIP distortion. It provides a precision-explicit PyTree API, projection-domain validity, offset coordinate methods, and source-to-target pixel maps. It does not parse arbitrary FITS WCS or interpolate image data: use Astropy for general WCS support and `reprojax` for image resampling.

## Install

The PyPI command becomes available once 0.3.0 is published; publication may still be pending:

```bash
pip install mapprojax
```

For a checkout, use `pip install '.[test]'`. Supported Python/JAX/NumPy minima are declared in [pyproject.toml](pyproject.toml), based on the tested minimum environment. The `rocm` extra is for Linux ROCm installations with Python 3.12 or newer and version-matched JAX/jaxlib/ROCm plugins; it is not a generic GPU extra. See the [manual](docs/user_manual.md).

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
jax.block_until_ready((x, y))
assert jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(y))
```

Pixels and `crpix` are 0-based. `crval` is in degrees; `proj` and `unproj` use radians in `(longitude, latitude)` order. TAN is invalid at and behind the horizon; SIN is valid on the front hemisphere and inside its inverse unit disc. Invalid samples produce NaNs.

## Precision modes

| Mode | Native/pixel path | Absolute celestial rotation | Requirement |
| --- | --- | --- | --- |
| `"f32"` | float32 | not available | Use `proj_offset` / `unproj_offset` for offset-space work. |
| `"mixed"` | float32 | float64 | JAX x64 enabled. |
| `"f64"` | float64 | float64 | JAX x64 enabled. |

Choose `mixed` for a lower-precision native path with absolute celestial coordinates, or `f64` for a fully float64 path. Mixed forward regression **caps** span 0.17–17 mas across the [reference fixtures](tests/data/accuracy_baselines.json), largest on the coarse 240 arcsec/px fields; these are not measured errors or an fp64-equivalence promise.

| Coordinate operation | Workload-specific precision evidence |
|---|---|
| WCS f32 | Offset-only native path; no absolute `proj`/`unproj`. [Independent accuracy and host composition](docs/benchmarks/accuracy_results.md). |
| WCS mixed / f64 | Absolute paths with x64. [Per-case Astropy agreement](docs/benchmarks/accuracy_results.md), separate from regression caps. |
| Reprojector f32 | Sampled maxima 5.28e-4–8.44e-3 px across the 4096² plate-scale/separation cases, on Strix Halo CPU and ROCm, against the JAX f64 absolute path with typed coordinate inputs. [Geometry, sampling and raw results](docs/benchmarks/precision_results_stage5_2026-11-07.md). |
| Reprojector f64 | Same cases: maximum agreement differences ≤4.75e-9 px on CPU and ≤3.88e-9 px on ROCm against that f64 path; not an independent astrometric bound. [Results](docs/benchmarks/precision_results_stage5_2026-11-07.md). |

As a separate worked example, the real SPHEREx pair has f32 agreement maxima 2.49e-4 / 2.03e-4 px against Astropy on 1009/1090 forward / 320/1090 reverse overlapping samples, **not every image pixel** ([frozen report](docs/benchmarks/accuracy_results.md)). The synthetic envelope is not a universal guarantee; errors depend on geometry, plate scale and coordinate dtype. The [August precision report](docs/benchmarks/precision_results.md) is historical.

Batch compatible WCS objects by stacking their PyTrees and using `jax.vmap`; all entries must share projection, precision, SIP shapes, and `inverse_steps`. `Reprojector(src, tgt)` computes a coordinate map only, not image resampling. `Reprojector.pixel_map(x, y, *, bounds_shape=None)` additionally returns validity with finite zero stand-ins for invalid coordinates and optional half-open image bounds. See the [user manual](docs/user_manual.md#reprojection) for grid/vmap examples and the [differentiability contract](docs/user_manual.md#differentiability-contract).

## Serialization and migration

Use the JSON-safe `to_dict()` and matching projection class's `from_dict()` methods. The contract preserves type, precision, reference dtype, and SIP state. ASDF integration and the old undocumented byte methods have been removed.

The old NumPy, `*Array`, `*Jax`, and `*Mixed` public class families have no compatibility aliases. Import `Tan` / `Sin` from `mapprojax` and specify a precision mode. Replace Array classes with stacked PyTrees and `jax.vmap`; replace Mixed classes with `precision="mixed"`. See [the user manual](docs/user_manual.md) for offset composition, SIP use, batching, serialization, differentiability limits, and detailed migration notes.

## Related tools

Scope/backend comparison, checked against the linked package versions on 2026-11-07; no comparative speed or accuracy benchmark is implied:

| Tool | Scope and precision design |
|---|---|
| mapprojax 0.3.0 | Coordinate-only TAN/SIN + SIP JAX kernel; explicit WCS precision and host-composed relative f32/f64 mapping. Image resampling belongs in reprojax. |
| [JAX-GalSim 2026.7.0 `GSFitsWCS`](https://github.com/GalSim-developers/JAX-GalSim/blob/v2026.7.0/jax_galsim/fitswcs.py) | WCS inside a JAX galaxy-simulation toolkit, with more projection families; its constructor uses JAX array conversion rather than mapprojax's explicit three-mode API. |
| [dfreproject 1.3.1](https://pypi.org/project/dfreproject/1.3.1/) | TAN/SIP image reprojection and interpolation using PyTorch on CPU/GPU; tensor dtype/device policy, not this coordinate-only JAX API. See its [methodology](https://dfreproject.readthedocs.io/en/latest/methodology.html). |
| [Astropy-affiliated reproject 0.21.0](https://pypi.org/project/reproject/0.21.0/) | General astronomical image reprojection with interpolation, adaptive and pixel-overlap methods in the NumPy/Astropy/SciPy ecosystem; different scope from a split-precision JAX coordinate kernel. |

## License

MIT; parts of the projection code derive from [cds-mapproj-rust](https://github.com/cds-astro/cds-mapproj-rust), used under its MIT option. Both copyright notices and permission texts are in [LICENSE](LICENSE).
