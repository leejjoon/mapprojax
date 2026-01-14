# Adaptive Reprojection Algorithm Details

This document details the algorithmic steps of the "adaptive" reprojection method implemented in the `reproject` package. This method is based on the paper "On Re-sampling of Solar Images" by C. E. DeForest (2004). The goal is to perform anti-aliased resampling of images under arbitrary coordinate transformations.

## 1. Introduction

The adaptive resampling algorithm smoothly transitions between interpolation (when upsampling) and spatial averaging (when downsampling). This is achieved by computing the local Jacobian of the coordinate transformation at each output pixel. The Jacobian tells us how a pixel in the output image maps to a region in the input image. If the mapping indicates downsampling (output pixel covers many input pixels), the algorithm averages over the corresponding elliptical region in the input image. If it indicates upsampling, it effectively performs interpolation.

## 2. Inputs and Outputs

### Inputs
*   **Source Image (`I`)**: A 2D array of pixel values.
*   **Coordinate Transformation (`T`)**: A function that maps output pixel coordinates $(x_{out}, y_{out})$ to input pixel coordinates $(u_{in}, v_{in})$.
    *   $(u, v) = T(x, y)$
*   **Output Shape**: The dimensions of the target image $(H_{out}, W_{out})$.
*   **Kernel**: The weighting function used for averaging. Common choices are Gaussian or Hann.
*   **Parameters**:
    *   `kernel_width`: Width of the kernel (sigma for Gaussian).
    *   `sample_region_width`: Extent of the sampling region (in output pixels) to consider.
    *   `center_jacobian`: Boolean, whether to compute Jacobian at pixel centers using finite differences (more accurate but slower) or averaging mid-point Jacobians.
    *   `despike_jacobian`: Boolean, whether to remove outliers in the Jacobian map.

### Output
*   **Target Image (`O`)**: The reprojected 2D array.
*   **Footprint** (Optional): A mask indicating valid data coverage.

## 3. Algorithm Steps

The algorithm iterates over every pixel in the output grid, determines the corresponding location and shape in the input grid, and computes the weighted average of input pixels.

### Step 3.1: Jacobian Calculation

For each pixel $(x, y)$ in the output image, we need the Jacobian matrix $J$ of the transformation $T$:

$$
J = \begin{pmatrix}
\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y}
\end{pmatrix}
$$

There are two ways to compute this:

1.  **Centered Jacobian (`center_jacobian=True`)**:
    Compute $T$ at $(x, y)$, $(x+1, y)$, and $(x, y+1)$.
    $$
    \frac{\partial u}{\partial x} \approx T_u(x+1, y) - T_u(x, y)
    $$
    *Note: The actual implementation might use central differences or forward differences depending on the grid setup.*

2.  **Mid-point Averaging (`center_jacobian=False`)**:
    Compute $T$ at grid vertices (e.g., $x-0.5, y-0.5$).
    Compute derivatives at mid-points (e.g., $x, y-0.5$ and $x-0.5, y$) using finite differences of vertices.
    Average these mid-point derivatives to get the Jacobian at the pixel center $(x, y)$.

### Step 3.2: Jacobian Despiking (Optional)

In some coordinate transformations (e.g., spherical projections with wrapping), the Jacobian can have singularities or discontinuities (spikes).
1.  Compute the magnitude squared of the Jacobian: $M^2 = \sum J_{ij}^2$.
2.  For each pixel, compare $M^2$ to the 25th percentile of its $3 \times 3$ neighborhood.
3.  If $M^2 > 10 \times \text{percentile}_{25}$, mark as spike.
4.  Replace spike Jacobians with the average of non-spike neighbors.

### Step 3.3: Singular Value Decomposition (SVD) and Anti-aliasing

This is the core of the adaptive method. We decompose $J$ to understand the local scaling and rotation.

1.  **Decompose**: $J = U \Sigma V^T$, where $\Sigma = \text{diag}(s_0, s_1)$ are singular values.
2.  **Clamp (Anti-aliasing)**: Ensure we never sample below the Nyquist rate of the input image.
    $$
    s_0' = \max(1.0, s_0)
    $$
    $$
    s_1' = \max(1.0, s_1)
    $$
    If $s < 1.0$, it means we are upsampling (magnifying). Clamping to 1.0 ensures we don't treat the input pixel as smaller than it is, effectively reverting to interpolation. If $s > 1.0$, we are downsampling (minifying), and the singular value represents the size of the averaging footprint.
3.  **Recompose**: Compute the effective Jacobian $J_{eff} = U \Sigma' V^T$.
4.  **Inverse**: Compute the inverse of the effective Jacobian, $J_{inv} = J_{eff}^{-1}$. This is used to map distance in input space back to the "filter space" (canonical unit circle/square).

### Step 3.4: Defining the Sampling Region

We need to determine which input pixels contribute to the current output pixel $(x, y)$. The output pixel is effectively a unit square (or circle) in output space. The Jacobian maps this to a parallelogram (or ellipse) in input space.

1.  **Gaussian Kernel**: The sampling region is an ellipse defined by the singular values.
    *   Calculate bounding box in input space centered at $(u_0, v_0) = T(x, y)$.
    *   Radius roughly proportional to `sample_region_width / min(1/s0, 1/s1)`.
    *   (In implementation: Since we inverted $J$, we use the singular values of $J_{inv}$, which are $1/s_i$).

2.  **Hann Kernel**: The region is defined by the transformed corners of the Hann window (typically $[-1, 1] \times [-1, 1]$ in output space).
    *   Map the 4 corners of the window using $J_{eff}$ to input space.
    *   Compute bounding box of these transformed corners.

### Step 3.5: Sampling and Weight Accumulation

Iterate over all input pixels $(u, v)$ within the computed bounding box.

1.  **Input Offset**: $\Delta u = u - u_0$, $\Delta v = v - v_0$.
2.  **Transform to Filter Space**: Map the offset back to the canonical filter domain using $J_{inv}$:
    $$
    \begin{pmatrix} \Delta x' \\ \Delta y' \end{pmatrix} = J_{inv} \begin{pmatrix} \Delta u \\ \Delta v \end{pmatrix}
    $$
3.  **Compute Weight $W$**:
    *   **Gaussian**: $W = \exp\left( - \frac{\Delta x'^2 + \Delta y'^2}{\sigma^2} \right)$
    *   **Hann**: $W = (\cos(\pi \Delta x') + 1)(\cos(\pi \Delta y') + 1)$ if $|\Delta x'|<1, |\Delta y'|<1$, else $0$.
4.  **Accumulate**:
    *   `weight_sum` += $W$
    *   `value_sum` += $W \times I(u, v)$
    *   Handle boundary conditions (e.g., if $(u, v)$ is outside input image, ignore or use fill value).

### Step 3.6: Normalization

The final value for output pixel $(x, y)$ is:
$$
O(x, y) = \frac{\text{value\_sum}}{\text{weight\_sum}}
$$

**Flux Conservation (Optional)**:
If flux conservation is required, multiply the result by the determinant of the original Jacobian $| \det(J) |$ (area scaling factor).

## 4. Implementation Details (Pseudo-code)

```python
for y in 0..H_out-1:
    for x in 0..W_out-1:
        # 1. Map center
        u_0, v_0 = T(x, y)

        # 2. Get Jacobian J at (x, y)
        J = calculate_jacobian(x, y)

        # 3. SVD and Clamp
        # J = U * S * V_t
        U, s, V_t = svd(J)
        s[0] = max(1.0, s[0])
        s[1] = max(1.0, s[1])

        # 4. Inverse Effective Jacobian for filter mapping
        # We need J_inv = J_eff^(-1) = (U * S_clamped * V_t)^(-1)
        # J_inv = V * S_clamped^(-1) * U_t
        # Note: In standard SVD returns, V_t is V transposed. So V_t.T is V.

        S_inv = diag(1.0/s[0], 1.0/s[1])
        J_inv = V_t.T @ S_inv @ U.T

        # 5. Determine Bounding Box in Input Image
        # (Simplified for Gaussian)
        # s[i] are scaling factors from output->input.
        # Large s means we are sampling a large region (downsampling).
        # We need to cover a region proportional to the largest scale.
        radius = sample_region_width / (2 * min(1.0/s[0], 1.0/s[1]))
        # Note: If J scales by 0.5 (downsampling), s=1 (clamped). radius ~ width/2.
        # If J scales by 2.0 (downsampling), s=2. radius ~ width.

        u_min = floor(u_0 - radius)
        u_max = ceil(u_0 + radius)
        v_min = floor(v_0 - radius)
        v_max = ceil(v_0 + radius)

        val_sum = 0
        w_sum = 0

        # 6. Convolution Loop
        for v in v_min..v_max:
            for u in u_min..u_max:
                if not in_bounds(u, v): continue

                # Offset in input space
                du = u - u_0
                dv = v - v_0

                # Map to filter space (roughly output pixel space)
                dx_prime = J_inv[0,0]*du + J_inv[0,1]*dv
                dy_prime = J_inv[1,0]*du + J_inv[1,1]*dv

                # Evaluate Filter
                w = kernel(dx_prime, dy_prime)

                if w > 0:
                    val_sum += w * I[v, u]
                    w_sum += w

        # 7. Normalize
        if w_sum > 0:
            O[y, x] = val_sum / w_sum
        else:
            O[y, x] = NAN
```

## 5. Notes on 2x2 SVD
The SVD of a 2x2 matrix $M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$ can be computed explicitly.
Let $E = (A+D)/2, F = (A-D)/2, G = (C+B)/2, H = (C-B)/2$.
$Q = \sqrt{E^2 + H^2}, R = \sqrt{F^2 + G^2}$.
Singular values: $s_0 = Q + R, s_1 = |Q - R|$.
Angles can be derived from $\arctan(G/F)$ and $\arctan(H/E)$.

## References
*   C. E. DeForest, "On Re-sampling of Solar Images", Solar Physics 219, 3–23 (2004).
