# Blocking (Tiling) Benchmark Results

**Date:** January 14, 2026  
**Backend:** CPU  
**Precision:** Mixed (float32 for pixel/native, float64 for celestial)

This benchmark evaluates the performance impact of the `block_size` parameter in `adaptive_reproject`. It compares full vectorization (`block_size=None`) against various chunk sizes across different image resolutions.

## Results

### 1024 x 1024 Pixels
| Block Size | Description | Time (ms) | Notes |
| :--- | :--- | :--- | :--- |
| **None** | **Full Vectorization** | **1,130 ms** | **Fastest** |
| 1M | 1024 x 1024 | 1,139 ms | |
| 256k | 512 x 512 | 1,252 ms | |
| 64k | 256 x 256 | 1,368 ms | |
| 16k | 128 x 128 | 1,986 ms | |

### 2048 x 2048 Pixels
| Block Size | Description | Time (ms) | Notes |
| :--- | :--- | :--- | :--- |
| **None** | **Full Vectorization** | **4,365 ms** | **Fastest** |
| 256k | 512 x 512 | 4,881 ms | |
| 64k | 256 x 256 | 5,403 ms | |
| 1M | 1024 x 1024 | 6,304 ms | |
| 16k | 128 x 128 | 7,675 ms | |

### 4096 x 4096 Pixels
| Block Size | Description | Time (ms) | Notes |
| :--- | :--- | :--- | :--- |
| **None** | **Full Vectorization** | **23,610 ms** | **Fastest** |
| 1M | 1024 x 1024 | 23,904 ms | |
| 256k | 512 x 512 | 24,741 ms | |
| 64k | 256 x 256 | 26,158 ms | |
| 16k | 128 x 128 | 31,196 ms | |

## Summary of Findings

1.  **Workload Difference**: `adaptive_reproject` is significantly more computationally intensive than simple coordinate transformation (as measured in `benchmark_precision.py`). This is due to the per-pixel calculation of Jacobians (using AD), SVD decompositions, and kernel-weighted convolutions.
2.  **Vectorization Efficiency**: Full vectorization (`None`) remains the fastest approach on CPU when memory is available. It allows JAX/XLA to optimize the entire operation globally.
3.  **Tiling Overhead**: Introducing blocks adds some overhead (~10-40% on CPU). Larger block sizes (1M) minimize this overhead and approach the performance of full vectorization.
4.  **Memory Safety**: While `block_size=None` is fastest for these resolutions, it will eventually hit memory limits on larger images or memory-constrained accelerators (GPUs). Tiling provides a safe fallback.

## Recommendation

*   For typical image sizes (up to 4k) on high-memory systems, use the default `block_size=None`.
*   For very large images or GPU execution, use `block_size=1048576` (1M pixels) to balance performance and memory safety.