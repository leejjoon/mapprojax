import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=['transform_func', 'output_shape', 'kernel', 'max_window_radius', 'block_size', 'min_coverage'])
def adaptive_reproject(
    image,
    transform_func,
    output_shape,
    kernel='gaussian',
    kernel_width=1.0,
    sample_region_width=4.0,
    max_window_radius=5,
    block_size=None,
    min_coverage=0.5
):
    """
    Adaptive reprojection algorithm using JAX.
    
    Based on 'On Re-sampling of Solar Images' by C. E. DeForest (2004).
    
    Args:
        image: Input image (2D JAX array).
        transform_func: A JAX-transformable function that takes (x_out, y_out) 
                        and returns (u_in, v_in).
        output_shape: Tuple (height, width) of the target image.
        kernel: 'gaussian' (default) or 'hann'.
        kernel_width: Width of the kernel (sigma for Gaussian).
        sample_region_width: Extent of the sampling region in output pixels.
        max_window_radius: Maximum radius (in input pixels) for the convolution window.
        block_size: Optional integer. Number of pixels to process in a single vectorized batch.
                    Use this to reduce memory usage for large images. If None, processes all pixels at once.
        min_coverage: Minimum fraction of the kernel weight that must fall on valid (non-NaN) 
                      input pixels to produce a valid output pixel. 
                      0.0 means any overlap with valid data produces a result (can smear edges).
                      1.0 means the kernel must be fully within valid data.
                      Default 0.5.
    
    Returns:
        Reprojected image of shape output_shape.
    """
    H_out, W_out = output_shape
    H_in, W_in = image.shape
    
    # Pad input image to handle window boundary conditions without shifting/clipping
    # We use 0.0 padding.
    padded_image = jnp.pad(image, max_window_radius, mode='constant', constant_values=0.0)
    
    # Generate output grid
    y_grid, x_grid = jnp.meshgrid(jnp.arange(H_out, dtype=float), jnp.arange(W_out, dtype=float), indexing='ij')
    
    def get_kernel_weight(dx_prime, dy_prime):
        """Compute weight in filter space."""
        # Normalize by sample_region_width / 2 so that dx_prime=1.0 
        # corresponds to the edge of the sampling region.
        scale = sample_region_width / 2.0
        x = dx_prime / scale
        y = dy_prime / scale
        r2 = x**2 + y**2
        
        if kernel == 'gaussian':
            # Gaussian: W = exp( - r^2 / sigma^2 )
            # Here r is already scaled by sample_region_width/2.
            # kernel_width (sigma) is relative to the unit output pixel.
            return jnp.exp(- (dx_prime**2 + dy_prime**2) / (kernel_width**2)) * (r2 < 1.0)
        elif kernel == 'hann':
            # Hann: (cos(pi*x)+1)(cos(pi*y)+1) inside [-1,1]
            mask = (jnp.abs(x) < 1.0) & (jnp.abs(y) < 1.0)
            w = (jnp.cos(jnp.pi * x) + 1) * (jnp.cos(jnp.pi * y) + 1)
            return jnp.where(mask, w, 0.0)
        else:
            return jnp.where(r2 < 1.0, 1.0, 0.0)

    def process_pixel(x, y):
        # 1. Map center coordinates
        # transform_func returns (u, v)
        uv = transform_func(x, y)
        u0, v0 = uv[0], uv[1]
        
        # 2. Jacobian Calculation using JAX AD
        # We define a wrapper to differentiate with respect to (x, y)
        def t_func(xy_vec):
            # xy_vec is [x, y]
            res = transform_func(xy_vec[0], xy_vec[1])
            return jnp.stack([res[0], res[1]])
        
        # J = d(u,v)/d(x,y)
        J = jax.jacfwd(t_func)(jnp.array([x, y]))
        
        # 3. SVD and Clamping (Anti-aliasing)
        U, S, Vt = jnp.linalg.svd(J)
        s0, s1 = S[0], S[1]
        
        # Clamp singular values to min 1.0 (prevents upsampling artifacts/blurring)
        s0_c = jnp.maximum(1.0, s0)
        s1_c = jnp.maximum(1.0, s1)
        
        # 4. Inverse Effective Jacobian for filter mapping
        # J_inv = V * S_inv * U.T
        S_inv = jnp.diag(jnp.array([1.0/s0_c, 1.0/s1_c]))
        J_inv = Vt.T @ S_inv @ U.T
        
        # 5. Define Window
        # We extract a fixed-size window from the padded image.
        # Center in integer coordinates
        u0_int = jnp.rint(u0).astype(int)
        v0_int = jnp.rint(v0).astype(int)
        
        # Start indices in PADDED image
        # Padded image has (0,0) at (-R, -R) of original.
        # So index = original_index + R
        # slice_start = (v0_int + R) - R = v0_int
        start_v = v0_int
        start_u = u0_int
        
        win_size = 2 * max_window_radius + 1
        
        # Dynamic slice: clamps start index to keep window within array if needed,
        # but we padded enough so it should be fine mostly.
        # However, if u0 is way out of bounds, it clamps.
        window = jax.lax.dynamic_slice(padded_image, (start_v, start_u), (win_size, win_size))
        
        # 6. Coordinate Grid for Window (relative to u0, v0)
        win_v_idx, win_u_idx = jnp.meshgrid(jnp.arange(win_size), jnp.arange(win_size), indexing='ij')
        
        # Offset in input image space
        # global_v = v0_int - R + win_v_idx
        # dv = global_v - v0 = v0_int - v0 - R + win_v_idx
        dv = (win_v_idx.astype(float) - max_window_radius) + (v0_int - v0)
        du = (win_u_idx.astype(float) - max_window_radius) + (u0_int - u0)
        
        # 7. Map to Filter Space
        dx_prime = J_inv[0, 0] * du + J_inv[0, 1] * dv
        dy_prime = J_inv[1, 0] * du + J_inv[1, 1] * dv
        
        # 8. Compute Weights
        raw_weights = get_kernel_weight(dx_prime, dy_prime)
        
        # Mask valid pixels (check original image bounds and ignore NaNs)
        global_v = v0_int - max_window_radius + win_v_idx
        global_u = u0_int - max_window_radius + win_u_idx
        
        in_bounds = (global_v >= 0) & (global_v < H_in) & (global_u >= 0) & (global_u < W_in)
        not_nan = ~jnp.isnan(window)
        valid_mask = in_bounds & not_nan
        
        masked_weights = raw_weights * valid_mask
        
        total_possible_weight = jnp.sum(raw_weights)
        valid_weight_sum = jnp.sum(masked_weights)
        
        # Compute weighted sum ignoring NaNs
        weighted_sum = jnp.sum(jnp.where(valid_mask, window * masked_weights, 0.0))
        
        # Check coverage condition
        # If total_possible_weight is 0 (e.g. kernel completely out of bounds?), result is NaN
        coverage = jnp.where(total_possible_weight > 1e-8, valid_weight_sum / total_possible_weight, 0.0)
        
        return jnp.where(coverage >= min_coverage, weighted_sum / valid_weight_sum, jnp.nan)

    if block_size is None:
        # Standard full vectorization
        flat_out = jax.vmap(process_pixel)(x_grid.ravel(), y_grid.ravel())
        return flat_out.reshape(H_out, W_out)
    else:
        # Chunked processing
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        num_pixels = x_flat.size
        
        # Calculate padding needed for blocking
        remainder = num_pixels % block_size
        if remainder != 0:
            pad_size = block_size - remainder
            x_flat = jnp.pad(x_flat, (0, pad_size), constant_values=0)
            y_flat = jnp.pad(y_flat, (0, pad_size), constant_values=0)
        else:
            pad_size = 0
            
        # Reshape into blocks: (num_blocks, block_size)
        num_blocks = x_flat.size // block_size
        x_blocks = x_flat.reshape(num_blocks, block_size)
        y_blocks = y_flat.reshape(num_blocks, block_size)
        
        def process_block(carry, block_inputs):
            bx, by = block_inputs
            out_block = jax.vmap(process_pixel)(bx, by)
            return carry, out_block

        # lax.map over the blocks
        _, flat_blocks_out = jax.lax.scan(process_block, None, (x_blocks, y_blocks))
        
        flat_out = flat_blocks_out.ravel()
        
        if pad_size > 0:
            flat_out = flat_out[:-pad_size]
        
        return flat_out.reshape(H_out, W_out)