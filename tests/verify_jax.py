import time
import numpy as np
import jax
import jax.numpy as jnp
from mapprojax import Tan, Sin
from mapprojax.jax_projections import TanJax, SinJax

# Enable 64-bit precision for JAX to match NumPy
jax.config.update("jax_enable_x64", True)

def proj2_logic(wcs_src, wcs_tgt, x_src, y_src):
    ra, dec = wcs_src.unproj(x_src, y_src)
    x_tgt, y_tgt = wcs_tgt.proj(ra, dec)
    return x_tgt, y_tgt

# JIT compile the function
# By passing WCS objects as arguments, JAX will use their tree_flatten methods
@jax.jit
def proj2_jit(wcs_src, wcs_tgt, x_src, y_src):
    return proj2_logic(wcs_src, wcs_tgt, x_src, y_src)

def test_jax_consistency():
    print("Setting up WCS parameters...")
    crpix_src = [100.0, 100.0]
    cd_src = [[-0.001, 0.0], [0.0, 0.001]] 
    crval_src = [45.0, 30.0]

    crpix_tgt = [100.0, 100.0]
    cd_tgt = [[-0.001, 0.0], [0.0, 0.001]] 
    crval_tgt = [45.0, 30.0]

    # Instantiate NumPy classes
    wcs_src_np = Tan(crpix_src, cd_src, crval_src)
    wcs_tgt_np = Sin(crpix_tgt, cd_tgt, crval_tgt)

    # Instantiate JAX classes
    wcs_src_jax = TanJax(crpix_src, cd_src, crval_src)
    wcs_tgt_jax = SinJax(crpix_tgt, cd_tgt, crval_tgt)

    # Generate test data
    y_src, x_src = np.indices((200, 200), dtype=float)
    
    print("Running NumPy version...")
    start = time.time()
    x_out_np, y_out_np = proj2_logic(wcs_src_np, wcs_tgt_np, x_src, y_src)
    print(f"NumPy time: {time.time() - start:.4f}s")

    print("Running JAX version (first run includes compilation)...")
    # Convert inputs to JAX arrays
    x_src_jax = jnp.array(x_src)
    y_src_jax = jnp.array(y_src)
    
    start = time.time()
    x_out_jax, y_out_jax = proj2_jit(wcs_src_jax, wcs_tgt_jax, x_src_jax, y_src_jax)
    # Block until ready to measure actual time
    x_out_jax.block_until_ready()
    print(f"JAX time (1st run): {time.time() - start:.4f}s")

    print("Running JAX version (2nd run)...")
    start = time.time()
    x_out_jax, y_out_jax = proj2_jit(wcs_src_jax, wcs_tgt_jax, x_src_jax, y_src_jax)
    x_out_jax.block_until_ready()
    print(f"JAX time (2nd run): {time.time() - start:.4f}s")

    # Verification
    print("Verifying results...")
    np.testing.assert_allclose(x_out_jax, x_out_np, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_out_jax, y_out_np, rtol=1e-12, atol=1e-12)
    print("SUCCESS: JAX results match NumPy results!")

if __name__ == "__main__":
    test_jax_consistency()
