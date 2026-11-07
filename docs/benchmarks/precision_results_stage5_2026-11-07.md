# Stage 5 precision measurements — 2026-11-07

Corrected 0.3.0 harness, AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S (Strix Halo), Linux x86_64, Python 3.12.13, JAX/jaxlib 0.11.1, NumPy 2.5.1, ROCm plugins 0.11.1. The GPU client reports `rocm 70253211`; raw metadata includes devices and platform information. These are local working-tree measurements, not a CI-built release-candidate validation or release closure. The JSON records the base revision and exact harness SHA-256.

## Reproduce and inspect

From the checkout and its reference environment:

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python benchmarks/benchmark_precision.py --sizes 512 1024 2048 4096 --warmups 3 --repeats 10 --json /tmp/precision-cpu.json --markdown /tmp/precision-cpu.md
JAX_PLATFORMS=rocm XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python benchmarks/benchmark_precision.py --sizes 512 1024 2048 4096 --warmups 3 --repeats 10 --json /tmp/precision-rocm.json --markdown /tmp/precision-rocm.md
```

Raw results: [CPU](../../benchmarks/results/stage5-precision-2026-11-07-cpu.json), [ROCm](../../benchmarks/results/stage5-precision-2026-11-07-rocm.json). Each includes every warm timing sample, median/min/max/std, input/output dtypes, finite counts, geometry, command, dependency versions and script hash. Timed JIT paths are labelled explicitly; eager JAX f64 is not NumPy. Each call waits for the complete returned tree. Setup, transfer, warmup and compilation are excluded. These are throughput observations, not an attribution of speed differences to arithmetic alone.

**Timing coverage limitation:** the initial CPU production run including 8192² was killed with exit 137 before emitting a report on this 32 GB shared-memory host. The cause was not independently established; memory pressure is plausible, not proven. Successful recorded timings cover 512²–4096² only. No 8192² timing is claimed on either backend; CLI production defaults still include it. This limitation does not shrink the 5000-point accuracy geometry below.

## Synthetic accuracy, not independent ground truth

Reference: the jitted JAX f64 TAN `unproj` → SIN `proj` path with float64 coordinates. Seed 0, 5000 uniformly sampled points in `[0,8192)` on each axis, CRPIX `(100,100)`, 3.6 arcsec/px, source `(45°,30°)`, target `(45.5°,30.3°)`. Every comparison has 5000/5000 finite overlapping samples. Lower-precision paths use float32 coordinate inputs; f64 paths use float64. Errors below are maximum Euclidean **pixel** differences against that f64 path, not astrometric error bounds.

| Operation | CPU max px | ROCm max px |
|---|---:|---:|
| WCS mixed, jitted absolute path | 0.00167718 | 0.00167718 |
| Reprojector f32, jitted | 0.00251680 | 0.00251680 |
| Reprojector f64, jitted | 1.58e-11 | 1.92e-11 |
| f32 offset path + host absolute composition, x64 off | 0.00897877 | 0.0103966 |

The host offset example uses the producer's typed `offset_scale()`, never a recomputed cosine. It answers an absolute-coordinate composition question and is not a pure on-device reprojection timing.

## Plate-scale and pointing-separation envelope

TAN → SIN, CRPIX `(2048,2048)`, 5000 seed-0 samples in a **4096²** span, source `(180°,30°)`, target offset in declination. Reference is the same JAX f64 absolute path; f32/f64 reprojection coordinate inputs match their kernel dtype. All cases have 5000/5000 finite overlap. Sampled maximum f32 pixel differences on both CPU and ROCm range from **5.28e-4 to 8.44e-3 px** over the listed cases; this is not a universal guarantee. F64 maxima across these cases are ≤4.75e-9 px (CPU) and ≤3.88e-9 px (ROCm), agreement with the f64 path, not independent accuracy.

| Case | Plate scale arcsec/px | Centre separation arcsec | f32 max px, CPU / ROCm |
|---|---:|---:|---:|
| Cross-field | 3.6 | 1800 | 7.37e-4 / 7.37e-4 |
| Cross-field | 0.1 | 1800 | 4.82e-3 / 4.82e-3 |
| Cross-field | 0.05 | 1800 | 8.43e-3 / 8.43e-3 |
| Same-pointing grid | 0.1, 0.05, 0.02 | 1, 10, 60, 300 | See all 12 rows per backend in raw JSON |

## Independent reference and caps are separate

The frozen [Astropy accuracy report](accuracy_results.md) measures one real SPHEREx pair: f32 maxima 2.49e-4 / 2.03e-4 px over 1009/1090 forward and 320/1090 reverse overlapping samples. It does not test every image pixel or establish the synthetic envelope. Mixed WCS forward caps of 0.17–17 mas in [accuracy_baselines.json](../../tests/data/accuracy_baselines.json) are regression **caps**, not measured maxima; coarse 240 arcsec/px fields have the largest caps. Mixed supports absolute `proj`/`unproj`; f32 does not. The [August report](precision_results.md) remains historical, with its original hardware and pre-0.3.0 API.
