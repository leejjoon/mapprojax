# User Manual

`mapprojax` is a JAX coordinate kernel for TAN and SIN FITS WCS projections, including SIP distortion. It is not an image-resampling package: use `reprojax` for resampling and coadds, and Astropy for general FITS header parsing and WCS support beyond these projections.

## Conventions and installation

Install with `pip install mapprojax`, or with `pip install .` from a source checkout. `from_astropy` needs astropy: use `pip install 'mapprojax[astropy]'`. The kernel itself does not import astropy. Supported Python/JAX/NumPy minima are declared in [pyproject.toml](../pyproject.toml). The Linux `rocm` extra requires Python 3.12 or newer and matching JAX/jaxlib/ROCm plugin versions; use `uv sync --extra test --extra rocm` with the supported ROCm runtime, not a generic GPU installation. Record interpreter, driver/runtime and actual device separately from dependency constraints.

A WCS uses 0-based pixel coordinates, stores `crpix` in pixels and `crval` in degrees, and takes/returns sky coordinates in radians in `(longitude, latitude)` order. `proj(lon, lat)` maps sky to pixels; `unproj(x, y)` maps pixels to sky.

Use `mapprojax.wcs_utils.load_wcs_params_from_header` to extract the supported parameters from an Astropy FITS header. It converts FITS `CRPIX` to the package's 0-based pixel convention.

## Constructing a WCS

The root `Tan` and `Sin` exports are the precision-explicit JAX PyTree classes in `mapprojax.wcs`. Choose a precision mode at construction:

```python
import jax
import jax.numpy as jnp
from mapprojax import Tan, Sin

jax.config.update("jax_enable_x64", True)
crpix = [100.0, 100.0]
cd = [[-0.001, 0.0], [0.0, 0.001]]
crval = [45.0, 30.0]  # degrees

tan = Tan(crpix, cd, crval, precision="mixed")
sin = Sin(crpix, cd, crval, precision="f64")
x, y = tan.proj(jnp.radians(jnp.asarray([45.1, 45.2])),
                jnp.radians(jnp.asarray([30.1, 30.2])))
lon, lat = tan.unproj(x, y)
assert jnp.allclose(jnp.degrees(lon), jnp.array([45.1, 45.2]), atol=1e-5)
assert jnp.allclose(jnp.degrees(lat), jnp.array([30.1, 30.2]), atol=1e-5)
```

| `precision` | Native/pixel pipeline | Celestial rotation | Use |
| --- | --- | --- | --- |
| `"f32"` | float32 | unavailable | Use `unproj_offset` / `proj_offset` where absolute sky coordinates are not needed in the traced computation. |
| `"mixed"` | float32 | float64 | Differentiable absolute coordinates with a lower-precision native pipeline. Requires JAX x64. |
| `"f64"` | float64 | float64 | Full-float64 coordinate path. Requires JAX x64. |

The f32 mode does not provide `proj`/`unproj`: absolute celestial rotation requires float64. For a host-side absolute coordinate result from an offset-mode WCS, use the helpers below. Mixed-mode accuracy is workload-dependent; consult the measured per-case limits in `docs/benchmarks/accuracy_results.md` and `docs/benchmarks/precision_results.md` rather than treating it as an fp64-equivalent bound.

TAN is valid on the forward hemisphere and has no finite inverse-plane radius limit; the exact horizon is invalid. SIN is valid on the front hemisphere and its inverse is valid only inside the unit disc, including its boundary. Invalid samples are returned as NaN, independently per sample.

## SIP distortion

Pass a `Sip` instance to either projection. Coefficients may be supplied as arrays; the SIP object and its coefficient shapes are part of the WCS PyTree structure.

```python
import jax
import jax.numpy as jnp
import numpy as np
from mapprojax import Tan
from mapprojax.sip import Sip

jax.config.update("jax_enable_x64", True)
crpix = [2.0, 2.0]
cd = [[-0.001, 0.0], [0.0, 0.001]]
crval = [45.0, 30.0]
a = np.zeros((3, 3))
a[2, 0] = 1.5e-6
sip = Sip(a=a)
wcs = Tan(crpix, cd, crval, sip=sip, precision="mixed")
x, y = jnp.array([1.0, 3.0]), jnp.array([2.0, 3.0])
lon, lat = wcs.unproj(x, y)
x_back, y_back = wcs.proj(lon, lat)
assert jnp.allclose(x_back, x, atol=1e-3)
assert jnp.allclose(y_back, y, atol=1e-3)
```

SIP inverse refinement is controlled by the `Sip.inverse_steps` value. Use matching SIP shapes and `inverse_steps` when batching WCS objects.

## Loading from astropy

`mapprojax.from_astropy` builds a `Tan` or `Sin` directly from an `astropy.wcs.WCS`. Prefer it over `load_wcs_params_from_header`: astropy normalises CROTA, units and alternate-version cards that the raw header loader refuses.

```python
import jax
import jax.numpy as jnp
from astropy.io import fits
from astropy.wcs import WCS
from mapprojax import from_astropy

jax.config.update("jax_enable_x64", True)

header = fits.Header()
header["NAXIS"], header["NAXIS1"], header["NAXIS2"] = 2, 1024, 1024
header["CTYPE1"], header["CTYPE2"] = "RA---TAN-SIP", "DEC--TAN-SIP"
header["CRPIX1"], header["CRPIX2"] = 512.5, 512.5
header["CRVAL1"], header["CRVAL2"] = 150.0, 30.0
header["CD1_1"], header["CD2_2"] = -1.0e-4, 1.0e-4
header["A_ORDER"], header["B_ORDER"] = 2, 2
header["A_0_2"] = 2.0e-5
header["B_2_0"] = 2.0e-5

wcs = WCS(header)
projection = from_astropy(wcs, precision="f64")
lon, lat = projection.unproj(jnp.array([0.0, 1023.0]), jnp.array([0.0, 1023.0]))
x, y = projection.proj(lon, lat)
print(type(projection).__name__, float(x[0]), float(y[0]))
```

`precision` is required and behaves exactly as it does on `Tan`/`Sin`: it never enables JAX x64. In `"f32"` the absolute `proj`/`unproj` are unavailable (they raise `TypeError`), so use `unproj_offset`/`proj_offset` or a `Reprojector`. `inverse_steps` forwards the Newton iteration count for the SIP inverse.

For a cube, pass `wcs.celestial` yourself. Automatic axis extraction would hide a caller's cube bug and would silently change which axes the SIP and pixel-scale matrices cover, so a `naxis != 2` WCS is refused.

The returned kernel is in the input WCS's own celestial frame. RADESYS, EQUINOX, MJD and observation keywords are **ignored, not applied and not kept**: the conversion preserves the numerical pixel-to-coordinate mapping but not the frame identity or the observation metadata. Keep the original astropy WCS when composing different frames. In the same spirit, the result describes the materialised `WCS`, not the original FITS file: cards astropy already discarded can be neither recovered nor refused.

### Refusals

Every refusal is a `ValueError` whose message reads `from_astropy: <property> is unsupported (<detail>)`, except a missing astropy (`ImportError`) and a non-`WCS` argument (`TypeError`).

| `<property>` | Refused when |
| --- | --- |
| `type` | the argument is not an `astropy.wcs.WCS` (`TypeError`) |
| `axis order` | `naxis != 2` (message names `wcs.celestial`), or the longitude/latitude axes are swapped |
| `non-celestial axes` | the two axes are not a celestial pair |
| `CPDIS`, `CQDIS`, `DP`, `DQ` | a lookup-table distortion is present (also reported as `cpdis1`/`cpdis2`) |
| `det2im1`, `det2im2` | a detector-to-image correction is present |
| `PS` | SIP `PS` parameters are present |
| `TPV`, `TPD`, `-TAB` | the CTYPE names a convention this kernel does not reproduce, including one hidden behind an apparent `TAN` |
| `-SIP without SIP` | the CTYPE names SIP but the WCS carries no coefficients |
| `projection` | the projection is not TAN or SIN |
| `PV` | a `PV` parameter other than SIN's no-op `PV2_1`/`PV2_2` is present |
| `LONPOLE` | `LONPOLE` is not the default 180-degree orientation (`+90` with no card resolves to `0.0`) |
| `cunit1`, `cunit2` | the axis unit is not angular |
| `sip.crpix` | the SIP reference pixel disagrees with `CRPIX1`/`CRPIX2` |
| `sip triangle` | a nonzero SIP coefficient lies outside `i + j <= order` |
| `non-finite <name>` | `crpix`, `crval`, the scale matrix or a SIP array is not finite |
| `init`, `header` | astropy itself rejects the WCS while initialising or serialising it |

### Where it differs from `load_wcs_params_from_header`

The two entry points disagree in both directions, deliberately. Astropy semantics are authoritative for `from_astropy`.

| Case | `from_astropy` | `load_wcs_params_from_header` |
| --- | --- | --- |
| CROTA rotation | accepted (normalised) | refused |
| `rad` / `arcsec` units | converted to degrees | refused |
| mixed `CD` with `PC`/`CDELT` | follows astropy: `PC * CDELT` wins | reads the `CD` cards |
| `-SIP` CTYPE with no coefficients | refused | accepted, `sip=None` |
| SIP order 0 or 1 | reads astropy's matrix | reads the `i + j <= order` triangle |

Numerical agreement between the two is therefore claimed only for headers with a single linear representation and card-stable values, which is what the test suite asserts.

## Batching with `jax.vmap`

The former `*Array` classes are replaced by stacking compatible WCS PyTrees and applying `jax.vmap`. A batch must share projection type, precision, PyTree structure, SIP shapes, and `inverse_steps`.

```python
import jax
import jax.numpy as jnp
from mapprojax import Tan

jax.config.update("jax_enable_x64", True)
crpix = [2.0, 2.0]
cd = [[-0.001, 0.0], [0.0, 0.001]]
wcses = [
    Tan(crpix, cd, [45.0, 30.0], precision="mixed"),
    Tan(crpix, cd, [50.0, 31.0], precision="mixed"),
]
stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *wcses)
x = jnp.arange(3.0)
y = jnp.arange(3.0)
# Shared pixel grid: outputs have shape (pixels, centres).
lon, lat = jax.vmap(lambda w: w.unproj(x, y), in_axes=0, out_axes=1)(stacked)
assert lon.shape == lat.shape == (3, 2)
assert jnp.all(jnp.isfinite(lon)) and jnp.all(jnp.isfinite(lat))
```

One WCS also accepts ordinary broadcastable coordinate arrays. For paired WCSes and pixel samples, use `jax.vmap` with the WCS and pixel arrays on axis 0.

## Offset helpers

`unproj_offset(x, y)` returns `(Δlongitude · cos(latitude₀), Δlatitude)` in radians relative to `crval`; `proj_offset` is its inverse. These methods avoid materializing absolute sky coordinates in the native per-pixel calculation. For a host-side absolute result, pass the producer's offset scale (without recomputing it) to the exported helpers:

```python
import jax
import numpy as np
from mapprojax import Tan, absolute_from_offset, offset_from_absolute

jax.config.update("jax_enable_x64", False)
wcs = Tan([2.0, 2.0], [[-0.001, 0.0], [0.0, 0.001]],
          [45.0, 30.0], precision="f32")
x, y = np.array([1.0, 3.0], dtype=np.float32), np.array([2.0, 3.0], dtype=np.float32)
lon_offset, lat_offset = wcs.unproj_offset(x, y)
lon, lat = absolute_from_offset(
    (lon_offset, lat_offset),
    crval=wcs.crval,
    offset_scale=wcs.offset_scale(),
)
offsets = offset_from_absolute(
    (lon, lat), crval=wcs.crval, offset_scale=wcs.offset_scale()
)
np.testing.assert_allclose(offsets, (lon_offset, lat_offset), atol=1e-10)
```

`crval` and `offset_scale` are keyword-only, so a coordinate pair cannot be passed in their place by position. The helpers are host-only and not JIT- or autodiff-compatible. Their inputs must retain the dtypes returned by the offset-producing WCS: a float32 producer's offsets must come with its float32 `offset_scale()`, and a Python float scale means float64. Longitude is wrapped to `[0, 2π)`. `offset_from_absolute((lon, lat), crval=..., offset_scale=...)` performs the inverse conversion and returns offsets in the scale's dtype; absolute float64 coordinates are accepted with a float32 scale. If the scale passed is exactly zero, the longitude is undefined and returned as NaN. A reference at a pole does not by itself give zero: the producer's cosine of 90° rounds to about 6e-17 in float64 and -4e-8 in float32, and the helpers divide by that same rounded value.

## Reprojection

`Reprojector(src, tgt)` composes two supported WCS pipelines and maps source pixels to target pixels. Use `pix_to_pix(x, y)` for map coordinates; it returns NaN where a projection leg is invalid. Use `pixel_map(x, y, *, bounds_shape=None)` when a separate validity mask and finite coordinates are needed. `matrix_precision="f32"` is the default and keeps the per-pixel path, including the relative rotation, in float32; use `"f64"` (requires x64) when that measured precision trade-off is unsuitable. `Reprojector` maps coordinates only; it does not interpolate image data.

```python
import jax
import jax.numpy as jnp
from mapprojax import Reprojector, Tan, Sin

jax.config.update("jax_enable_x64", False)
crpix = [2.0, 2.0]
cd = [[-0.001, 0.0], [0.0, 0.001]]
src = Tan(crpix, cd, [45.0, 30.0], precision="f32")
tgt = Sin(crpix, cd, [45.1, 30.05], precision="f32")
rp = Reprojector(src, tgt)  # default matrix_precision="f32"
x_tgt, y_tgt = rp.pix_to_pix(jnp.arange(3.0), jnp.arange(3.0))
assert jnp.all(jnp.isfinite(x_tgt)) and jnp.all(jnp.isfinite(y_tgt))
```

`pixel_map` has the same source-to-target direction and broadcasting as `pix_to_pix`, and returns `(x_mapped, y_mapped, valid)`. Coordinates use the Reprojector's kernel dtype and are finite everywhere; invalid entries have zero stand-ins, with `valid=False`. `pix_to_pix` is unchanged and continues to return NaN for invalid samples.

When `bounds_shape=(ny, nx)` is supplied, validity additionally requires the mapped target pixel to lie in `[-0.5, nx-0.5) × [-0.5, ny-0.5)`: lower edges are included and upper edges excluded. The tuple must contain exactly two positive built-in Python integers (not booleans or NumPy integers). It is static configuration, so keep it fixed at trace time under `jit`. Bounds classify the computed kernel-dtype coordinates without clipping, epsilon adjustments, or promotion; float32 may classify a value within an ulp of an edge differently from float64.

For resampling, map each output pixel into the input image's pixel grid by constructing `Reprojector(output_wcs, input_wcs)`. `jnp.indices` returns `(yy, xx)` in row/column order, so pass `xx` as x and `yy` as y. The bounds shape is the input image shape:

```python
import jax
import jax.numpy as jnp
from mapprojax import Reprojector, Tan

jax.config.update("jax_enable_x64", False)
input_image = jnp.ones((4, 4), dtype=jnp.float32)
output_ny, output_nx = 2, 3
input_wcs = Tan([1.0, 1.0], [[-0.001, 0.0], [0.0, 0.001]],
                [45.0, 30.0], precision="f32")
output_wcs = Tan([1.0, 1.0], [[-0.001, 0.0], [0.0, 0.001]],
                 [45.0, 30.0], precision="f32")
input_ny, input_nx = input_image.shape
rp = Reprojector(output_wcs, input_wcs)
yy, xx = jnp.indices((output_ny, output_nx), dtype=jnp.float32)
x_in, y_in, valid = rp.pixel_map(
    xx, yy, bounds_shape=(input_ny, input_nx)
)
assert x_in.shape == y_in.shape == valid.shape == (2, 3)
assert jnp.all(valid)
assert jnp.allclose(x_in, xx, atol=1e-4)
assert jnp.allclose(y_in, yy, atol=1e-4)
```

Tiles, broadcastable coordinate arrays, and sparse samples can be passed directly; no grid helper is required. Compatible Reprojectors can be stacked as PyTrees and mapped over with an explicit `jax.vmap` axis:

```python
import jax
import jax.numpy as jnp
from mapprojax import Reprojector, Tan

jax.config.update("jax_enable_x64", False)
input_ny, input_nx = 4, 4
src = Tan([1.0, 1.0], [[-0.001, 0.0], [0.0, 0.001]],
          [45.0, 30.0], precision="f32")
tgt = Tan([1.25, 1.0], [[-0.001, 0.0], [0.0, 0.001]],
          [45.0, 30.0], precision="f32")
reprojectors = [Reprojector(src, src), Reprojector(src, tgt)]
yy, xx = jnp.indices((2, 3), dtype=jnp.float32)
stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *reprojectors)
x_in, y_in, valid = jax.vmap(
    lambda r: r.pixel_map(xx, yy, bounds_shape=(input_ny, input_nx)),
    in_axes=0,
)(stacked)
assert x_in.shape == y_in.shape == valid.shape == (2, 2, 3)
assert jnp.all(valid)
assert jnp.allclose(x_in[1] - x_in[0], 0.25, atol=1e-4)
```

Apply validity before nonlinear loss operations. If downstream evaluation can be unsafe at invalid coordinates, substitute a regular in-domain value before evaluating it as well:

```python
import jax
import jax.numpy as jnp
from mapprojax import Reprojector, Tan

jax.config.update("jax_enable_x64", False)
wcs = Tan([1.0, 1.0], [[-0.001, 0.0], [0.0, 0.001]],
          [45.0, 30.0], precision="f32")
rp = Reprojector(wcs, wcs)
x_in, y_in, valid = rp.pixel_map(
    jnp.array([1.0, 20.0]), jnp.array([1.0, 20.0]), bounds_shape=(4, 4)
)
safe_x, safe_y = 1.0, 1.0  # log is regular and differentiable here.
observed = jnp.array([0.5, 0.0])

def model(x, y):
    return jnp.log(x) + jnp.log(y)

x_safe = jnp.where(valid, x_in, safe_x)
y_safe = jnp.where(valid, y_in, safe_y)
prediction = model(x_safe, y_safe)
residual = jnp.where(valid, prediction - observed, 0.0)
loss = jnp.sum(residual ** 2)
assert jnp.array_equal(valid, jnp.array([True, False]))
assert jnp.isfinite(loss) and jnp.allclose(loss, 0.25, atol=1e-4)
```

`safe_x` and `safe_y` must be values where the downstream model is regular. Masking only after a nonlinear operation can still allow invalid values or singular derivatives to contaminate gradients.

There are two constructors:

- `Reprojector(src, tgt, frame_rotation=None, matrix_precision="f32")` builds the native(src) → native(tgt) matrix on the host in float64 from each WCS's `crval`, which must be a concrete float64 `(2,)` array. `frame_rotation` is an optional finite `(3, 3)` host matrix between the two celestial frames (for example ICRS → Galactic). This is the right constructor outside `jit`.
- `Reprojector.from_matrix(src, tgt, m_rel, *, matrix_precision="f32")` takes the relative matrix as a JAX value. It never goes through the host, so it works inside `jit` and lets gradients flow through `m_rel` and anything it was computed from, such as live `crval` values being fitted.

The host matrix is the same as `compose_relative_rotation(src, tgt, frame_rotation)`, which you can call yourself.

### Keeping the original references

A WCS that has passed through a JAX transformation may no longer carry its original `crval`: with x64 off, `jax.jit` returns its float64 leaves as float32, and inside a trace `crval` is a tracer. `Reprojector(src, tgt)` then refuses to build the host matrix rather than silently losing precision. Keep the original host references and compose with them:

```python
import jax
import jax.numpy as jnp
import numpy as np
from mapprojax import Reprojector, Tan, Sin, compose_relative_rotation

jax.config.update("jax_enable_x64", False)
crpix = [2.0, 2.0]
cd = [[-0.001, 0.0], [0.0, 0.001]]
host_src = np.asarray([45.0, 30.0])
host_tgt = np.asarray([45.1, 30.05])
src = Tan(crpix, cd, host_src, precision="f32")
tgt = Sin(crpix, cd, host_tgt, precision="f32")

src_t, tgt_t = jax.jit(lambda s, t: (s, t))(src, tgt)  # crval may now be float32
m = compose_relative_rotation(src_t, tgt_t, crval_src=host_src, crval_tgt=host_tgt)
rp = Reprojector.from_matrix(src_t, tgt_t, m, matrix_precision="f32")
assert src_t.crval.dtype == tgt_t.crval.dtype == jnp.float32
x, y = rp.pix_to_pix(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
assert jnp.all(jnp.isfinite(x)) and jnp.all(jnp.isfinite(y))
```

`crval_src` and `crval_tgt` are keyword-only and validated like a WCS `crval`: concrete, finite, shape `(2,)` and already float64. Nothing can prove that an override describes the intended frame, so keeping the right originals is the caller's responsibility. A changed reference or frame needs a new composition. Host composition is outside autodiff; to differentiate with respect to `crval`, build the matrix inside the traced function from the live references and pass it to `from_matrix`.

## Differentiability contract

Differentiability applies on smooth, valid interior regions. It does not imply that every derivative is nonzero.

| API surface | Differentiability |
| --- | --- |
| Native/pixel WCS methods (`pix_to_native`, `native_to_pix`) | Differentiable with respect to coordinates and numerical WCS leaves on valid smooth regions. They do not use `crval`, so the `crval` derivative is zero by construction. |
| Offset methods (`unproj_offset`, `proj_offset`) | Differentiable with respect to coordinates and numerical WCS leaves on valid smooth regions. The reference longitude is unused (zero sensitivity by construction); the longitude offset scale can depend on reference latitude. `f32` supports these methods. |
| Absolute WCS methods (`proj`, `unproj`) | Differentiable with respect to coordinates and numerical WCS leaves, including `crval`, in `mixed` and `f64` on valid smooth regions. These methods are unavailable in `f32`; `mixed` and `f64` require JAX x64. |
| Already-built eager `Reprojector` | Its pixel maps differentiate through input coordinates and projection-leg parameters on valid regions. The host-composed relative rotation is fixed at construction, so there is no derivative from that matrix back to either leg's `crval` (zero by construction). |
| `Reprojector.from_matrix` | Differentiable through the supplied `m_rel` and any live values used to construct it inside the traced function. The matrix is not automatically tied to the source or target `crval`; make that dependency explicit. |
| Host composition and offset helpers (`compose_relative_rotation`, `absolute_from_offset`, `offset_from_absolute`) | Host-only; not JIT- or autodiff-compatible. |
| Validity and bounds masks | The boolean `valid` mask is not differentiable. Coordinates have gradients on valid smooth regions; no gradient is promised across mask transitions or for invalid samples. `pixel_map` gives invalid samples finite zero stand-ins, while `pix_to_pix` returns NaN. |
| Static configuration (precision, projection type, array shapes, SIP coefficient shapes, `Sip.inverse_steps`, `bounds_shape`) | Not differentiable; these are Python/static choices that determine PyTree structure, validation, or compiled program shape. |
| SIP inverse seed (`AP`/`BP`) | With A/B present, AP/BP seed Newton refinement: sensitivity is convergence-dependent, not identically zero for a finite step count, and vanishes as the inverse converges. If AP/BP alone defines the inverse (no A/B), its sensitivity is genuine and remains differentiable. |

Projection singularities, invalid-domain regions, and validity/bounds transitions are non-differentiable regions. Keep optimization samples away from them, and use `pixel_map`'s validity result to exclude invalid samples before nonlinear losses.

## JSON serialization

Serialize the precision-aware numerical state with `to_dict()` and standard JSON. Restore it with the matching projection class's `from_dict()`:

```python
import json
import jax
import jax.numpy as jnp
import numpy as np
from mapprojax import Tan
from mapprojax.sip import Sip

jax.config.update("jax_enable_x64", True)
a = np.zeros((3, 3))
a[2, 0] = 1.5e-6
wcs = Tan([2.0, 2.0], [[-0.001, 0.0], [0.0, 0.001]],
          np.array([45.0, 30.0], dtype=np.float64),
          sip=Sip(a=a, inverse_steps=3), precision="mixed")
payload = json.dumps(wcs.to_dict())
restored = Tan.from_dict(json.loads(payload))
assert restored.precision == wcs.precision
assert restored.crval.dtype == wcs.crval.dtype
assert restored.sip.inverse_steps == wcs.sip.inverse_steps
np.testing.assert_array_equal(restored.sip.a, wcs.sip.a)
assert jnp.allclose(jnp.array(restored.unproj(1.0, 2.0)),
                    jnp.array(wcs.unproj(1.0, 2.0)), atol=1e-12)
```

The serialized contract includes the projection type, precision, reference dtype, and SIP coefficients and inverse-step count. Loading through a different projection class or loading the former untyped dictionary format is rejected. ASDF conversion and the old undocumented `to_bytes`/`from_bytes` methods are not supported.

## Migration from the retired API

Version 0.3.0 is an API break with no compatibility aliases. Numerical projection formulas are unchanged.

| Before 0.3.0 | From 0.3.0 |
| --- | --- |
| `from mapprojax.projections import Tan, Sin` (NumPy) | `from mapprojax import Tan, Sin` with `precision="f64"`; outputs are JAX arrays |
| `from mapprojax import TanJax, SinJax` | `Tan`/`Sin` with `precision="f64"` (requires x64) |
| `TanJaxMixed`, `SinJaxMixed` | `Tan`/`Sin` with `precision="mixed"` (requires x64) |
| float32 offset-only JAX use (x64 off) | `Tan`/`Sin` with `precision="f32"`; `proj`/`unproj` are unavailable, use `proj_offset`/`unproj_offset` |
| `TanJaxArray`, `TanJaxArrayMixed`, `SinJaxArray*` with `crvals=(ra_array, dec_array)` | stack per-centre `Tan`/`Sin` PyTrees and use `jax.vmap`; see "Batching with `jax.vmap`" for `in_axes`/`out_axes` |
| `wcs.unproj_absolute(x, y)` | `absolute_from_offset(wcs.unproj_offset(x, y), crval=wcs.crval, offset_scale=wcs.offset_scale())` |
| `wcs.proj_absolute(lon, lat)` | `wcs.proj_offset(*offset_from_absolute((lon, lat), crval=wcs.crval, offset_scale=wcs.offset_scale()))` |
| `mapprojax.jax_projections`, `mapprojax.base.WCSBase`, `WCSArrayBase` | removed; use `mapprojax.wcs.Tan`/`Sin` |
| `Reprojector(src, tgt)` reading any `crval` | unchanged call; `crval` must be a concrete float64 pair, otherwise use `compose_relative_rotation(..., crval_src=..., crval_tgt=...)` with `Reprojector.from_matrix` |
| ASDF serialization (`mapprojax[asdf]`, the ASDF extension) | `to_dict()`/`from_dict()` with JSON |

The constructor stores `crval` as given: a NumPy float64 array or list of Python floats stays float64 on the host; an integer list is converted to float64; a float32 array stays float32 and cannot be used for host composition or the mixed/f64 celestial rotation.

The all-NumPy high-accuracy route that worked with x64 off is retired on purpose. Absolute output with x64 off remains available through the f32 offset path plus `absolute_from_offset`, which composes the reference in float64 on the host.

### Compatibility

mapprojax 0.3.x is tested with reprojax 0.2.0, which declares `mapprojax>=0.3.0,<0.4`. reprojax 0.1.0 imports the retired classes and does not work with mapprojax 0.3.0; the verified Stage 4 revision pair and closure evidence are recorded in `benchmarks/results/stage4-closure.md`.
