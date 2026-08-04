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
