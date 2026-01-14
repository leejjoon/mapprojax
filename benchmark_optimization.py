import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
from mapprojax.jax_projections import TanJax, SinJax
from mapprojax.utils import apply_rotation

# Enable x64 globally for JAX
jax.config.update("jax_enable_x64", True)

def run_benchmark():
    backend = jax.devices()[0].platform.upper()
    print(f"JAX Backend: {backend}")

    sizes = [512, 1024, 2048, 4096]
    
    # WCS Params
    crpix_src = [100.0, 100.0]
    cd_src = [[-0.001, 0.0], [0.0, 0.001]] 
    crval_src = [45.0, 30.0]

    crpix_tgt = [120.0, 120.0]
    cd_tgt = [[-0.001, 0.0], [0.0, 0.001]]
    crval_tgt = [40.0, 35.0]
    
    # Header
    print(f"{'Size':<12} | {'Std (Full)':<15} | {'Std (XYZ)':<15} | {'Optimized (Rot)':<15}")
    print("-" * 65)

    # 1. Standard Full Path: Pixel -> Plane -> Sphere -> RA/Dec -> Sphere -> Plane -> Pixel
    def path_full(ws, wt, x, y):
        ra, dec = ws.unproj(x, y)
        return wt.proj(ra, dec)
    
    path_full_jit = jax.jit(path_full)

    # 2. XYZ Path: Pixel -> Plane -> Sphere -> Sphere -> Plane -> Pixel
    # (Uses unproj_xyz and proj_xyz to skip RA/Dec trig)
    def path_xyz(ws, wt, x, y):
        x, y, z = ws.unproj_xyz(x, y)
        return wt.proj_xyz(x, y, z)
    
    path_xyz_jit = jax.jit(path_xyz)

    # 3. Optimized Path: Pixel -> Plane -> Native -> Native -> Plane -> Pixel
    # (Pre-multiplies rotation matrices to skip Celestial Sphere entirely)
    def path_optimized(ws, wt, m_combined, x, y):
        v_out = ws.pix_to_native(x, y)
        v_in = apply_rotation(m_combined, *v_out)
        return wt.native_to_pix(*v_in)
    
    path_optimized_jit = jax.jit(path_optimized)

    for N in sizes:
        # Generate Data
        y_np, x_np = np.indices((N, N), dtype=np.float64)
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        
        # WCS Objects
        wcs_src = SinJax(crpix_src, cd_src, crval_src)
        wcs_tgt = TanJax(crpix_tgt, cd_tgt, crval_tgt)
        
        # Precompute Combined Rotation for Optimized Path
        # Note: In the example script, I mapped Output -> Input.
        # Here, let's map Source (unproj) -> Target (proj).
        # Src unproj: v_c = M_src.T * v_n_src
        # Tgt proj: v_n_tgt = M_tgt * v_c
        # Combined: v_n_tgt = M_tgt * (M_src.T * v_n_src) = (M_tgt * M_src.T) * v_n_src
        m_combined = jnp.matmul(wcs_tgt.r_matrix, wcs_src.r_matrix.T)

        # --- Benchmark: Standard Full ---
        # Warmup
        out = path_full_jit(wcs_src, wcs_tgt, x_jax, y_jax)
        out[0].block_until_ready()
        
        start = time.time()
        out = path_full_jit(wcs_src, wcs_tgt, x_jax, y_jax)
        out[0].block_until_ready()
        t_full = (time.time() - start) * 1000

        # --- Benchmark: XYZ Path ---
        # Warmup
        out = path_xyz_jit(wcs_src, wcs_tgt, x_jax, y_jax)
        out[0].block_until_ready()
        
        start = time.time()
        out = path_xyz_jit(wcs_src, wcs_tgt, x_jax, y_jax)
        out[0].block_until_ready()
        t_xyz = (time.time() - start) * 1000

        # --- Benchmark: Optimized Path ---
        # Warmup
        out = path_optimized_jit(wcs_src, wcs_tgt, m_combined, x_jax, y_jax)
        out[0].block_until_ready()
        
        start = time.time()
        out = path_optimized_jit(wcs_src, wcs_tgt, m_combined, x_jax, y_jax)
        out[0].block_until_ready()
        t_opt = (time.time() - start) * 1000
        
        print(f"{N}x{N:<7} | {t_full:<15.2f} | {t_xyz:<15.2f} | {t_opt:<15.2f}")

if __name__ == "__main__":
    run_benchmark()
