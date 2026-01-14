"""
Benchmark: Performance Scaling with Resolution

Tests:
    Evaluates how JAX-accelerated geometric mapping scales with data volume compared 
    to standard NumPy implementations across image resolutions from 512x512 to 8192x8192.
    
Comparison:
    NumPy (f64) vs. JAX Standard (f64) vs. JAX Mixed Precision (Opt).
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
from mapprojax import Tan, Sin, TanJax, SinJax, TanJaxMixed, SinJaxMixed

# Enable x64 globally for JAX
jax.config.update("jax_enable_x64", True)

def run_benchmark():
    sizes = [512, 1024, 2048, 4096, 8192]
    
    # WCS Params
    crpix = [100.0, 100.0]
    cd = [[-0.001, 0.0], [0.0, 0.001]] 
    crval = [45.0, 30.0]
    
    # Header
    print(f"{'Size':<12} | {'NumPy (f64)':<15} | {'JAX Std (f64)':<15} | {'JAX Mix (Opt)':<15}")
    print("-" * 65)
    
    # Pre-compile JIT functions
    # Note: We include the tuple unpacking inside the lambda to measure the full pipeline
    proj2_std_jit = jax.jit(lambda ws, wt, x, y: (wt.proj(*ws.unproj(x, y))))
    proj2_mix_jit = jax.jit(lambda ws, wt, x, y: (wt.proj(*ws.unproj(x, y))))

    for N in sizes:
        # --- Generate Data ---
        # NumPy: Use f64 to represent standard high-precision workflow
        y_np, x_np = np.indices((N, N), dtype=np.float64)
        
        # JAX: Use f32 input for Mixed (it handles casting), f32/f64 for Std
        y_jax, x_jax = jnp.array(y_np, dtype=jnp.float32), jnp.array(x_np, dtype=jnp.float32)
        
        # --- NumPy Benchmark ---
        wcs_src_np = Tan(crpix, cd, crval)
        wcs_tgt_np = Sin(crpix, cd, crval)
        
        start = time.time()
        # Explicitly unpacking unproj results
        ra_np, dec_np = wcs_src_np.unproj(x_np, y_np)
        _ = wcs_tgt_np.proj(ra_np, dec_np)
        t_np = (time.time() - start) * 1000
        
        # --- JAX Standard Benchmark ---
        wcs_src_std = TanJax(crpix, cd, crval)
        wcs_tgt_std = SinJax(crpix, cd, crval)
        
        # Warmup
        out_std = proj2_std_jit(wcs_src_std, wcs_tgt_std, x_jax, y_jax)
        out_std[0].block_until_ready()
        
        # Measure
        start = time.time()
        out_std = proj2_std_jit(wcs_src_std, wcs_tgt_std, x_jax, y_jax)
        out_std[0].block_until_ready()
        t_std = (time.time() - start) * 1000

        # --- JAX Mixed Benchmark ---
        wcs_src_mix = TanJaxMixed(crpix, cd, crval)
        wcs_tgt_mix = SinJaxMixed(crpix, cd, crval)
        
        # Warmup
        out_mix = proj2_mix_jit(wcs_src_mix, wcs_tgt_mix, x_jax, y_jax)
        out_mix[0].block_until_ready()
        
        # Measure
        start = time.time()
        out_mix = proj2_mix_jit(wcs_src_mix, wcs_tgt_mix, x_jax, y_jax)
        out_mix[0].block_until_ready()
        t_mix = (time.time() - start) * 1000
        
        print(f"{N}x{N:<7} | {t_np:<15.2f} | {t_std:<15.2f} | {t_mix:<15.2f}")

if __name__ == "__main__":
    run_benchmark()