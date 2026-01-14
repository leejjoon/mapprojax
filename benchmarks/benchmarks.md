# Mapprojax Benchmarks

This directory contains scripts for evaluating the performance and memory efficiency of different coordinate transformation and reprojection strategies in `mapprojax`.

## Benchmark Scripts

### 1. `benchmark.py`
**Tests:** Broad-scale coordinate transformation efficiency.
- **Comparison:** NumPy (float64) vs. JAX Standard (float64) vs. JAX Mixed Precision (float32/float64).
- **Workload:** Broadcasts 100,000 celestial points against 100 separate WCS definitions (10 million total transformations).
- **Goal:** Measures raw throughput for high-volume coordinate mapping.

### 2. `benchmark_blocking.py`
**Tests:** Memory scalability and tiling overhead for image reprojection.
- **Comparison:** Full grid vectorization (`block_size=None`) vs. sequential chunked processing (`block_size=16k, 64k, 256k, 1M`).
- **Workload:** Executes the full `adaptive_reproject` algorithm (Jacobians, SVD, convolution) on images from 1k to 4k.
- **Goal:** Identifies the optimal trade-off between peak performance and memory safety for large images.

### 3. `benchmark_optimization.py`
**Tests:** Path-specific geometric optimizations.
- **Comparison:** 
  1. **Full (RA/Dec):** Standard path through spherical coordinates.
  2. **XYZ (Vector):** Direct path using celestial unit vectors (bypassing RA/Dec trigonometry).
  3. **Optimized (Rot):** Path using manually fused rotation matrices to map between native spheres directly.
- **Goal:** Demonstrates the speedup achievable by avoiding trigonometric conversions.

### 4. `benchmark_precision.py`
**Tests:** Performance scaling across image resolutions and precision modes.
- **Comparison:** NumPy vs. JAX Std vs. JAX Mixed across resolutions (512x512 to 8192x8192).
- **Goal:** Evaluates how JAX-accelerated geometric mapping scales with data volume compared to standard NumPy implementations.

## How to Run
Run any script directly using `python3`:
```bash
python3 benchmarks/benchmark_optimization.py
```

*Note: Benchmarks automatically detect and use available JAX backends (CPU/GPU/TPU).*
