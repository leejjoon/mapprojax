"""
Benchmark: Memory Scalability and Tiling Overhead

Tests:
    Evaluates the trade-off between full vectorization (`block_size=None`) and sequential 
    chunked processing for the `adaptive_reproject` algorithm. 
    Crucial for identifying optimal strategies to avoid OOM on large images or limited memory hardware.

Workload:
    Executes the full adaptive reprojection pipeline (Jacobians, SVD, Convolution) 
    on images of varying sizes (1k, 2k, 4k).
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
from mapprojax import TanJaxMixed, SinJaxMixed
from mapprojax.adaptive import adaptive_reproject

# Enable x64 globally for JAX
jax.config.update("jax_enable_x64", True)

def run_benchmark():
    backend = jax.devices()[0].platform.upper()
    print(f"JAX Backend: {backend}")
    print("Using Mixed Precision (f32 for pixel/native, f64 for celestial)")

    # Image sizes to test
    image_sizes = [1024, 2048, 4096]
    
    # Block sizes to test
    # None = Full Vectorization
    # Powers of 2 for tiling
    block_sizes = [None, 16384, 65536, 262144, 1048576]
    block_labels = ["None (Full)", "16k (128^2)", "64k (256^2)", "256k (512^2)", "1M (1024^2)"]

    # WCS Params
    crpix = [1000.0, 1000.0]
    cd = [[-0.0001, 0.0], [0.0, 0.0001]] 
    crval = [45.0, 30.0]

    # Setup WCS (Using Mixed Precision objects)
    wcs_src = SinJaxMixed(crpix, cd, crval)
    wcs_tgt = TanJaxMixed(crpix, cd, crval)

    # Transform function (Optimized XYZ path)
    def transform(x, y):
        u, v, w = wcs_tgt.unproj_xyz(x, y)
        return wcs_src.proj_xyz(u, v, w)

    print(f"{'Image Size':<12} | {'Block Size':<15} | {'Time (ms)':<15} | {'Notes'}")
    print("-" * 65)

    for N in image_sizes:
        shape = (N, N)
        # Create dummy input image in f32
        data_in = jnp.zeros(shape, dtype=jnp.float32)
        
        for b_size, b_label in zip(block_sizes, block_labels):
            
            # Skip full vectorization for 8k on memory constrained systems if needed,
            # but let's try to run it. If it OOMs, we catch it.
            
            try:
                # Warmup
                # We trace a new function for each block_size since it's static
                # Use a smaller dummy output shape for warmup to save time? 
                # No, adaptive_reproject JIT compiles based on shapes.
                # We must use real shapes.
                
                # To avoid excessive waiting, we run once for warmup+measure if N is huge?
                # Standard practice: Run once to compile, then measure.
                
                # Note: 'adaptive_reproject' is already JIT-ted. 
                # Calling it triggers compilation for these specific static args.
                
                start_compile = time.time()
                out = adaptive_reproject(
                    data_in, transform, shape, 
                    block_size=b_size,
                    max_window_radius=4 # Small radius for speed in benchmark logic
                )
                out.block_until_ready()
                compile_time = time.time() - start_compile
                
                # Measure Execution
                start_run = time.time()
                out = adaptive_reproject(
                    data_in, transform, shape, 
                    block_size=b_size,
                    max_window_radius=4
                )
                out.block_until_ready()
                run_time = (time.time() - start_run) * 1000
                
                print(f"{N}x{N:<7} | {b_label:<15} | {run_time:<15.2f} | Compile: {compile_time:.2f}s")
                
            except Exception as e:
                print(f"{N}x{N:<7} | {b_label:<15} | {'FAILED':<15} | {str(e)[:40]}...")
                # If we OOM, likely subsequent large runs will also fail or be unstable
                # But we continue.

if __name__ == "__main__":
    run_benchmark()
