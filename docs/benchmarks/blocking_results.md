# Blocking (Tiling) Benchmark Results

**Date:** January 14, 2026  
**Backend:** CPU  
**Precision:** float64

This benchmark evaluates the performance impact of the `block_size` parameter in `adaptive_reproject`. It compares full vectorization (`block_size=None`) against various chunk sizes.

## Results

### 4096 x 4096 Pixels

| Block Size | Description | Time (ms) | Notes |
| :--- | :--- | :--- | :--- |
| **None** | **Full Vectorization** | **16,776 ms** | **Fastest on CPU** |
| 1M | 1024 x 1024 | 23,208 ms | ~1.38x slower |
| 256k | 512 x 512 | 23,703 ms | ~1.41x slower |
| 64k | 256 x 256 | 24,940 ms | ~1.48x slower |
| 16k | 128 x 128 | 30,009 ms | ~1.78x slower |

### 8192 x 8192 Pixels

| Block Size | Description | Time (ms) | Notes |
| :--- | :--- | :--- | :--- |
| None | Full Vectorization | **FAILED** | **OOM Error** |
| **1M** | **1024 x 1024** | **93,683 ms** | **Fastest Blocked** |
| 256k | 512 x 512 | 94,072 ms | |
| 64k | 256 x 256 | 101,136 ms | |
| 16k | 128 x 128 | 119,441 ms | Slowest |

## Key Findings

1.  **Memory Limit**: Full vectorization (`None`) works best for smaller images (4k) where memory is sufficient. However, it **crashes (OOM)** on 8k images, proving the necessity of tiling for large-scale data.
2.  **Overhead**: Using `jax.lax.scan` introduces some overhead compared to pure `vmap`. On CPU, this overhead makes blocked processing ~30-40% slower than full vectorization when memory isn't an issue.
3.  **Optimal Block Size**: Larger block sizes (1M pixels) generally perform better than small blocks (16k) because they amortize the loop overhead and allow better vectorization utilization within each chunk.
4.  **Scalability**: The blocked implementation successfully processed the 8k image (~67 megapixels) which failed with the naive approach.

## Recommendation

*   Use `block_size=None` (default) for images $\le$ 4k x 4k.
*   Use `block_size=1048576` (1M, or roughly 1024x1024) for larger images to avoid OOM while maintaining good performance.
