# Mapprojax Benchmarks

This directory contains scripts for evaluating the performance of different coordinate transformation strategies in `mapprojax`.

## Benchmark Scripts

### 1. `benchmark.py`
**Tests:** JAX-first batching: a three-centre `vmap` over f64 `Tan` WCS PyTrees. It is not the old NumPy/JAX/mixed 100-WCS comparison described in the historical tables below.

### 2. `benchmark_optimization.py`
**Tests:** Path-specific geometric optimizations.
- **Comparison:** 
  1. **Full (RA/Dec):** Standard path through spherical coordinates.
  2. **XYZ (Vector):** Direct path using celestial unit vectors (bypassing RA/Dec trigonometry).
  3. **Optimized (Rot):** Path using manually fused rotation matrices to map between native spheres directly.
- **Goal:** Measures the paths without attributing every timing difference to a single optimization; report the backend and JIT policy.

### 3. `benchmark_precision.py`
**Tests:** Performance scaling across image resolutions and precision modes, plus an accuracy comparison.
- **Comparison:** `Tan`/`Sin` with `precision="f64"` vs. `precision="mixed"` vs. `Reprojector` (`f32`/`f64`) across resolutions (512x512 to 8192x8192), plus a separate accuracy check that includes offset-form absolute output (`absolute_from_offset`/`offset_from_absolute`).
- **Measurement:** eager JAX f64 and explicitly jitted f64/mixed/Reprojector paths, warm medians with complete output-tree synchronization. Reference is JAX f64, not NumPy ground truth. CLI supports `--sizes`, `--warmups`, `--repeats`, `--json`, `--markdown`. Preserve producer `offset_scale()` in host composition. [Corrected Stage 5 evidence](../docs/benchmarks/precision_results_stage5_2026-11-07.md); August results remain historical.

### 4. `benchmark_accuracy.py`
**Tests:** Accuracy against astropy, no timing.
- **Comparison:** mapprojax (`jax`, `offset_host`, `jax_mixed`) vs. `astropy.wcs` on every case in `tests/accuracy_reference.py` (header fixtures in `tests/data/`, synthetic near-pole and RA-wrap fields), plus the real SPHEREx pair through `Reprojector` with an astropy-derived ICRS↔Galactic rotation and a transposed-rotation negative control.
- **Goal:** Generates the published table `docs/benchmarks/accuracy_results.md` and exits non-zero on any budget/cap regression. Stage 5 never uses `--write-baselines` or `--allow-cap-increase`; committed caps and the reference report remain frozen.

### Other validation scripts

`benchmark_stage3.py` measures warm/lowering/compile/cold subprocess workloads; `compare_stage3.py` checks matched result pairs and budgets. `benchmark_sip.py` measures SIP paths. `compare_reprojax_pixel_map.py` checks downstream capability equivalence and `sweep_reprojector_gradients.py` records scale-aware finite-difference sweeps. Stage 3/4 canonical evidence is indexed by the append-only closure files in `benchmarks/results/`; do not overwrite old result files.

## How to Run
Run any script directly using `python3`:
```bash
python3 benchmarks/benchmark_optimization.py
python3 benchmarks/benchmark_precision.py                  # default backend (GPU if present)
JAX_PLATFORMS=cpu python3 benchmarks/benchmark_precision.py # forced CPU
```

*Note: Benchmarks automatically detect and use available JAX backends (CPU/GPU/TPU).*

## Results

### Current 0.3.0 precision evidence (2026-11-07)

The [new report](../docs/benchmarks/precision_results_stage5_2026-11-07.md) links corrected CPU/ROCm raw timings (512²–4096², three warmups, ten measured repeats) and the realistic 4096² plate-scale/separation envelope. Sampled f32 Reprojector maxima are 5.28e-4–8.44e-3 px against the JAX f64 absolute path, not an independent accuracy guarantee. The 8192² CPU attempt exited 137 before reporting; no 8192² timing is claimed on either backend. All timing paths' actual JIT status and transfer exclusions are recorded. No causal hardware/speedup attribution is made from these measurements alone.

### Historical AMD Strix Halo (2026-11-01, pre-Stage 3)

The following tables and observations are retained historical text. Script definitions and labels have since changed; these are not current 0.3.0 comparisons or validated explanations of current performance.

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
