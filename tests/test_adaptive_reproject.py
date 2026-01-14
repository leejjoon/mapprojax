import jax.numpy as jnp
import numpy as np
import pytest
from mapprojax.adaptive import adaptive_reproject

def test_identity_reproject():
    """Test that identity transform preserves the image (roughly)."""
    # Create a simple Gaussian blob image
    H, W = 30, 30
    y, x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    xc, yc = 15.0, 15.0
    image = jnp.exp(-((x - xc)**2 + (y - yc)**2) / (3.0**2))
    
    # Identity transform
    def identity(x, y):
        return x, y
        
    out = adaptive_reproject(
        image, 
        identity, 
        (H, W), 
        kernel='gaussian', 
        kernel_width=0.5, # Sharp kernel for identity
        max_window_radius=10
    )
    
    # Check if peak is preserved
    assert jnp.abs(out.max() - image.max()) < 0.1
    # Check if location is preserved
    max_idx = jnp.unravel_index(jnp.argmax(out), out.shape)
    assert max_idx == (15, 15)

def test_downsample_reproject():
    """Test downsampling (averaging)."""
    # Constant image
    image = jnp.ones((40, 40))
    
    # Scale down by 2 (output 20x20 covers input 40x40)
    # T(x, y) = (2x, 2y)
    def scale_2x(x, y):
        return 2.0 * x, 2.0 * y
        
    out = adaptive_reproject(
        image,
        scale_2x,
        (20, 20),
        kernel='gaussian',
        kernel_width=1.0,
        max_window_radius=10
    )
    
    # Should be close to 1.0 (averaging constant region)
    # Ignore edges
    center_val = out[10, 10]
    assert jnp.abs(center_val - 1.0) < 0.01

def test_coordinate_shift():
    """Test shifting the image."""
    H, W = 20, 20
    image = jnp.zeros((H, W))
    image = image.at[10, 10].set(1.0) # Delta function
    
    # Shift input by +2.5 in x and y
    # output(x, y) samples input(x-2.5, y-2.5)
    def shift(x, y):
        return x - 2.5, y - 2.5
        
    out = adaptive_reproject(
        image,
        shift,
        (H, W),
        kernel='gaussian',
        kernel_width=1.0
    )
    
    # Peak should move to roughly (12.5, 12.5)
    # Since it's discrete, peak pixel should be at (12, 12) or (13, 13)
    # Actually, input (10, 10) corresponds to output satisfying x-2.5=10 => x=12.5
    # So peak around 12, 13
    
    # Just check that mass moved
    assert out[10, 10] < 0.1 # Moved away
    # Check region around 12,13
    assert jnp.sum(out[11:15, 11:15]) > 0.5

def test_nan_handling():
    """Test that NaNs in source image are ignored and don't poison output."""
    H, W = 10, 10
    image = jnp.ones((H, W))
    image = image.at[5, 5].set(jnp.nan) # One NaN in the middle
    
    def identity(x, y):
        return x, y
        
    out = adaptive_reproject(
        image,
        identity,
        (H, W),
        kernel='gaussian',
        kernel_width=1.0,
        max_window_radius=3
    )
    
    # The output at (5, 5) should be 1.0 (average of neighbors) instead of NaN
    assert not jnp.isnan(out[5, 5])
    assert jnp.abs(out[5, 5] - 1.0) < 1e-5
    # Rest of the image should also be fine
    assert jnp.all(~jnp.isnan(out))

if __name__ == "__main__":
    test_identity_reproject()
    test_downsample_reproject()
    test_coordinate_shift()
    test_nan_handling()
    print("All tests passed!")
