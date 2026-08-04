# Precision-Mode Benchmark Results

**Date:** August 4, 2026
**JAX:** 0.11.0
**CPU:** 64 cores (`JAX_PLATFORMS=cpu`)
**GPU:** 4x CUDA device (default backend)

This benchmark (`benchmarks/benchmark_precision.py`) compares every precision
strategy this package currently offers for a src-WCS-pixel -> tgt-WCS-pixel
reprojection: `NumPy (f64)`, `JAX Std (f64)`, `JAX Mixed` (the pre-existing
`MixedPrecisionWCSJaxMixin`), and `Reprojector` (`fo64_report_and_plan.md`
Phase 1) at both `matrix_precision="f32"` and `"f64"`. Offset-form absolute
output (Phase 2: `unproj_absolute`/`proj_absolute`) is accuracy-checked
separately below, since it answers a different question (true RA/Dec output)
than the pixel-to-pixel modes.

WCS pair: TAN at CRVAL `(45.0, 30.0)` -> SIN at CRVAL `(45.5, 30.3)` (a
genuine, if modest, pointing difference -- a zero-separation pair would make
every mode's rotation matrix the identity and hide any real difference
between them).

## Timing (ms), GPU (`backend=gpu`)

| Size | NumPy (f64) | JAX Std (f64) | JAX Mixed | Reprojector f32 | Reprojector f64 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 512x512 | 42.69 | 1.00 | 1.09 | **0.94** | 1.08 |
| 1024x1024 | 220.23 | 1.84 | 1.70 | **0.82** | 1.09 |
| 2048x2048 | 1165.14 | 3.42 | 2.96 | **0.98** | 1.81 |
| 4096x4096 | 4681.38 | 9.73 | 7.92 | **1.75** | 3.52 |
| 8192x8192 | 18595.02 | 35.55 | 28.39 | **3.43** | 10.57 |

At 8192x8192, Reprojector (f32) is **~10.4x** faster than JAX Std and **~8.3x**
faster than JAX Mixed, and the gap widens with resolution. Reprojector (f64)
sits between JAX Std and JAX Mixed, as expected -- the speedup at f32 is from
running the *entire* per-pixel path in float32 (not just casting the
non-rotation math, which is all JAX Mixed does).

## Timing (ms), CPU (`JAX_PLATFORMS=cpu`, 64 cores)

| Size | NumPy (f64) | JAX Std (f64) | JAX Mixed | Reprojector f32 | Reprojector f64 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 512x512 | 57.06 | 16.55 | 10.43 | **2.26** | 3.50 |
| 1024x1024 | 214.52 | 60.51 | 32.58 | **3.53** | 12.56 |
| 2048x2048 | 1211.46 | 201.19 | 100.64 | **22.63** | 51.60 |
| 4096x4096 | 4864.49 | 570.31 | 309.23 | **91.39** | 173.49 |
| 8192x8192 | 18807.44 | 1463.75 | 831.91 | **289.45** | 526.37 |

The win is smaller on CPU (~5.1x vs JAX Std, ~2.9x vs JAX Mixed at
8192x8192) than on GPU, consistent with fp64 being only ~2x slower than fp32
on CPU rather than the 32-64x gap typical on consumer GPUs -- Reprojector's
float32-throughout path still wins by simplifying the traced graph (see
reprojax's own finding that `reproject_adaptive`'s GPU warm time dropped
0.93s -> 0.24s from the same change, driven by a simpler jaxpr for
`jax.jacfwd` to trace and XLA to compile, not just raw fp32-vs-fp64 FLOP
rate).

## Accuracy (max pixel error over 5000 random points, vs NumPy f64 ground truth)

Same on both backends (accuracy doesn't depend on which device ran it):

| Mode | Max error (px) |
| :--- | ---: |
| JAX Mixed (fp64 rotation + fp32 rest) | 1.9e-3 |
| **Reprojector (f32, no `jax_enable_x64` needed)** | **2.6e-3** |
| Reprojector (f64) | 2.0e-11 |
| Offset-form (`unproj_absolute`/`proj_absolute`, NumPy f64) | 2.9e-11 |
| Offset-form's `unproj_offset`, fp32, `jax_enable_x64=False` | 6.9-7.4e-3 |

Reprojector (f32) and JAX Mixed land in the same order of magnitude (~2e-3 px
for this WCS pair/plate scale) despite Reprojector doing strictly *more* of
the pipeline in float32 (JAX Mixed keeps the rotation itself in float64;
Reprojector doesn't need to, because it never routes through an absolute
celestial vector at all -- see its docstring). The f64 modes (Reprojector
f64, offset-form) land at the expected ~1e-11 px floor: ordinary float64
round-off, not a precision *strategy* difference.

## Conclusion

1. **Reprojector (f32) is the fastest mode at every resolution tested, on
   both backends**, and the only one that needs no `jax_enable_x64` at all.
   For the reprojection use case (map a pixel in one WCS to a pixel in
   another), it should be the default recommendation over JAX Mixed.
2. **JAX Mixed is not obsolete**: for callers needing full per-pixel
   `unproj_xyz`/`proj_xyz` (not just a fused src->tgt map), or working with a
   single WCS's own absolute output relative to a live `crval` for
   fine-grained differentiation through that step (unlike Reprojector's
   rotation, which is precomputed outside the trace -- see its docstring),
   JAX Mixed remains the right tool.
3. **The accuracy tradeoff is real but bounded** (~2e-3 px for this pair, at
   a coarse plate scale) and, per Phase 1's own separation sweep, grows with
   field-center separation and finer plate scales -- see
   `fo64_report_and_plan.md` and `Reprojector`'s docstring for the documented
   envelope before assuming f32 is accurate enough for a given instrument.
4. **Offset-form's fp32 path (~7e-3 px here) is in the same regime as
   Reprojector (f32)**, for a different use case (single-WCS absolute sky
   output rather than a src/tgt pixel map) -- see DEVELOPER.md Sec 4's
   decision tree for which one to reach for.

*Note: absolute numbers depend on CPU/GPU model and load; the ratios between
modes are the reproducible part.*
