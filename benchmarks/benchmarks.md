# Mapprojax Benchmarks

This directory contains scripts for evaluating the performance of different coordinate transformation strategies in `mapprojax`.

## Benchmark Scripts

### 1. `benchmark.py`
**Tests:** Broad-scale coordinate transformation efficiency.
- **Comparison:** NumPy (float64) vs. JAX Standard (float64) vs. JAX Mixed Precision (float32/float64).
- **Workload:** Broadcasts 100,000 celestial points against 100 separate WCS definitions (10 million total transformations).
- **Goal:** Measures raw throughput for high-volume coordinate mapping.

### 2. `benchmark_optimization.py`
**Tests:** Path-specific geometric optimizations.
- **Comparison:** 
  1. **Full (RA/Dec):** Standard path through spherical coordinates.
  2. **XYZ (Vector):** Direct path using celestial unit vectors (bypassing RA/Dec trigonometry).
  3. **Optimized (Rot):** Path using manually fused rotation matrices to map between native spheres directly.
- **Goal:** Demonstrates the speedup achievable by avoiding trigonometric conversions.

### 3. `benchmark_precision.py`
**Tests:** Performance scaling across image resolutions and precision modes, plus an accuracy comparison.
- **Comparison:** NumPy (f64) vs. JAX Std (f64) vs. JAX Mixed vs. `Reprojector` (`f32`/`f64`, `fo64_report_and_plan.md` Phase 1) across resolutions (512x512 to 8192x8192), plus a separate accuracy check that includes offset-form absolute output (Phase 2: `unproj_absolute`/`proj_absolute`).
- **Goal:** Evaluates how each precision strategy scales with data volume, and quantifies the speed/accuracy tradeoff of each -- not speed alone. Run once with the default backend and once with `JAX_PLATFORMS=cpu` to compare GPU vs CPU (JAX picks its backend at process start and can't switch mid-process). Recorded results: `docs/benchmarks/precision_results.md`.

## How to Run
Run any script directly using `python3`:
```bash
python3 benchmarks/benchmark_optimization.py
python3 benchmarks/benchmark_precision.py                  # default backend (GPU if present)
JAX_PLATFORMS=cpu python3 benchmarks/benchmark_precision.py # forced CPU
```

*Note: Benchmarks automatically detect and use available JAX backends (CPU/GPU/TPU).*

## Results

### AMD Strix Halo (2026-11-01)

- **Machine:** AMD Ryzen AI MAX+ PRO 395 (16 cores / 32 threads) with integrated Radeon 8060S GPU, 30 GB RAM visible to the OS (memory shared between CPU and GPU)
- **Software:** Linux 7.0, ROCm 7.2.3, Python 3.12, jax/jaxlib 0.11.1 with `jax-rocm7-plugin` (`uv sync --extra rocm`)
- **GPU runs** use `XLA_PYTHON_CLIENT_PREALLOCATE=false` (see note below). **CPU runs** use `JAX_PLATFORMS=cpu`.

All times in milliseconds.

#### `benchmark.py` (100 WCS × 100,000 points)

| Method | PROJ (CPU) | UNPROJ (CPU) | PROJ (GPU) | UNPROJ (GPU) |
|---|---|---|---|---|
| NumPy (f64) | 189 | 480 | — | — |
| JAX Std (f64) | 74.6 | 80.6 | 5.45 | 8.44 |
| JAX Mixed | 58.6 | 62.0 | 1.13 | 6.02 |

Each method is timed once, so these numbers vary noticeably between runs.

#### `benchmark_optimization.py`

| Size | Full (CPU) | XYZ (CPU) | Rot (CPU) | Full (GPU) | XYZ (GPU) | Rot (GPU) |
|---|---|---|---|---|---|---|
| 512×512 | 5.26 | 0.91 | 0.54 | 1.17 | 0.36 | 0.29 |
| 1024×1024 | 14.4 | 1.83 | 2.15 | 2.18 | 0.64 | 0.54 |
| 2048×2048 | 57.2 | 15.9 | 16.5 | 7.24 | 2.01 | 1.71 |
| 4096×4096 | 210 | 57.4 | 66.9 | 27.6 | 6.60 | 5.94 |

GPU numbers here were run with the default memory preallocation; that setting does not affect JAX-only timings.

#### `benchmark_precision.py`

| Size | NumPy (f64) | JAX Std (CPU) | JAX Mix (CPU) | JAX Std (GPU) | JAX Mix (GPU) |
|---|---|---|---|---|---|
| 512×512 | 18.9 | 5.60 | 3.54 | 0.88 | 0.77 |
| 1024×1024 | 61.8 | 18.7 | 8.84 | 1.94 | 1.69 |
| 2048×2048 | 307 | 72.9 | 32.3 | 7.13 | 6.23 |
| 4096×4096 | 1254 | 237 | 138 | 28.1 | 23.8 |
| 8192×8192 | 11425 | 755 | 422 | 111 | 93.4 |

#### Observations

- At 8192×8192, JAX on the GPU is ~100× faster than NumPy and ~7× faster than JAX on the CPU.
- On the GPU, mixed precision is only ~15–20% faster than full f64 (vs. ~1.8× on the CPU). The workload appears memory-bandwidth bound there, so avoiding f64 buys less on this chip.
- **Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` on this machine.** By default JAX reserves 75% of GPU memory at startup; on an APU that is system RAM. With preallocation on, NumPy in the same process ran 3–5× slower (e.g. 6396 ms vs. 1254 ms at 4096×4096) while the GPU timings were unchanged.
