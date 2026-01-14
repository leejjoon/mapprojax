import time
import numpy as np
import jax
import jax.numpy as jnp
from mapprojax import TanArray, TanJaxArray, TanJaxArrayMixed

# Enable x64 for JAX
jax.config.update("jax_enable_x64", True)

def benchmark():
    # Setup: Create a WCSArray with 100 separate WCS definitions
    n_wcs = 100
    crpix = [500, 500]
    cd = [[-0.0002777, 0.0], [0.0, 0.0002777]]
    
    # Random centers for the WCSs
    ra_centers = np.random.uniform(0, 360, n_wcs).reshape(-1, 1)
    dec_centers = np.random.uniform(-90, 90, n_wcs).reshape(-1, 1)
    
    # Inputs: 100,000 points per WCS (Broadcasting: 100 WCS x 100,000 points = 10,000,000 calculations)
    n_points = 100_000
    ra_input_np = np.random.uniform(0, 360, n_points).reshape(1, -1)
    dec_input_np = np.random.uniform(-90, 90, n_points).reshape(1, -1)
    
    print(f"{'Method':<20} | {'PROJ (ms)':<15} | {'UNPROJ (ms)':<15}")
    print("-" * 55)

    # --- NumPy ---
    wcs_np = TanArray(crpix, cd, (ra_centers, dec_centers))
    
    start = time.time()
    res_x, res_y = wcs_np.proj(ra_input_np, dec_input_np)
    t_proj_np = (time.time() - start) * 1000
    
    start = time.time()
    wcs_np.unproj(res_x, res_y)
    t_unproj_np = (time.time() - start) * 1000
    
    print(f"{'NumPy (f64)':<20} | {t_proj_np:<15.2f} | {t_unproj_np:<15.2f}")

    # JAX inputs
    ra_input_jax = jnp.array(ra_input_np)
    dec_input_jax = jnp.array(dec_input_np)

    # --- JAX Standard (f64) ---
    wcs_jax = TanJaxArray(crpix, cd, (ra_centers, dec_centers))
    
    # JIT wrap
    proj_jit = jax.jit(wcs_jax.proj)
    unproj_jit = jax.jit(wcs_jax.unproj)
    
    # Warmup
    out_x, out_y = proj_jit(ra_input_jax, dec_input_jax)
    out_x.block_until_ready()
    out_ra, out_dec = unproj_jit(out_x, out_y)
    out_ra.block_until_ready()
    
    start = time.time()
    res_x, res_y = proj_jit(ra_input_jax, dec_input_jax)
    res_x.block_until_ready()
    t_proj_jax = (time.time() - start) * 1000
    
    start = time.time()
    ra_out, dec_out = unproj_jit(res_x, res_y)
    ra_out.block_until_ready()
    t_unproj_jax = (time.time() - start) * 1000
    
    print(f"{'JAX Std (f64)':<20} | {t_proj_jax:<15.2f} | {t_unproj_jax:<15.2f}")

    # --- JAX Mixed Precision ---
    wcs_mix = TanJaxArrayMixed(crpix, cd, (ra_centers, dec_centers))
    
    # JIT wrap
    proj_mix_jit = jax.jit(wcs_mix.proj)
    unproj_mix_jit = jax.jit(wcs_mix.unproj)
    
    # Warmup
    out_x, out_y = proj_mix_jit(ra_input_jax, dec_input_jax)
    out_x.block_until_ready()
    out_ra, out_dec = unproj_mix_jit(out_x, out_y)
    out_ra.block_until_ready()
    
    start = time.time()
    res_x, res_y = proj_mix_jit(ra_input_jax, dec_input_jax)
    res_x.block_until_ready()
    t_proj_mix = (time.time() - start) * 1000
    
    start = time.time()
    ra_out, dec_out = unproj_mix_jit(res_x, res_y)
    ra_out.block_until_ready()
    t_unproj_mix = (time.time() - start) * 1000
    
    print(f"{'JAX Mixed':<20} | {t_proj_mix:<15.2f} | {t_unproj_mix:<15.2f}")

if __name__ == "__main__":
    benchmark()