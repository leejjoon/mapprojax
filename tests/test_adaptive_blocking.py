import jax.numpy as jnp
from mapprojax.adaptive import adaptive_reproject
import pytest

def test_block_processing():
    """Test that block processing yields same result as full processing."""
    H, W = 40, 40
    # Gradient image
    y_grid, x_grid = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    image = (x_grid + y_grid).astype(float)
    
    def identity(x, y):
        return x, y
        
    # Full processing (block_size=None)
    out_full = adaptive_reproject(
        image, identity, (H, W), block_size=None
    )
    
    # Block processing (block_size=100 -> 16 blocks)
    out_blocked = adaptive_reproject(
        image, identity, (H, W), block_size=100
    )
    
    # Check equality
    assert jnp.allclose(out_full, out_blocked, atol=1e-6)

def test_block_padding():
    """Test block processing when size is not a multiple of block_size."""
    H, W = 21, 21 # Total 441 pixels
    image = jnp.ones((H, W))
    
    def identity(x, y):
        return x, y
        
    # Block size 100 -> 4 blocks full, 1 partial (padded)
    out_blocked = adaptive_reproject(
        image, identity, (H, W), block_size=100
    )
    
    assert out_blocked.shape == (H, W)
    assert jnp.allclose(out_blocked, 1.0)

if __name__ == "__main__":
    test_block_processing()
    test_block_padding()
    print("Block processing tests passed!")
