# SIP Implementation Plan

## Overview
Implement Simple Imaging Polynomial (SIP) distortion support for `mapprojax`.
The SIP convention treats distortion as a polynomial addition to the linear transformation.
See `docs/sip/shupeADASS.pdf` for the definition.

## Objectives
1.  Implement a `Sip` class to handle distortion coefficients.
2.  Integrate SIP into the `WCSBase` and `WCSArrayBase` transformation pipeline.
3.  Ensure compatibility with both NumPy and JAX backends.
4.  Verify correctness against `astropy.wcs`.

## Implementation Steps

### 1. `src/mapprojax/sip.py`
Create a `Sip` class that:
*   Stores `a`, `b` (forward distortion) and `ap`, `bp` (reverse distortion) polynomial coefficients.
*   Implements `pix_to_foc(u, v)`: Applies A/B polynomials to relative pixel coordinates `u, v` to get "focal plane" (linearized) coordinates.
    *   Formula: $u' = u + f(u, v)$, $v' = v + g(u, v)$
*   Implements `foc_to_pix(U, V)`: Applies AP/BP polynomials to linearized coordinates `U, V` to get distorted pixel coordinates.
    *   Formula: $u = U + F(U, V)$, $v = V + G(U, V)$
*   Uses a backend-agnostic `_evaluate_poly` method (accepting `xp` for numpy/jax) to compute sums like $\sum A_{pq} u^p v^q$.

### 2. Modify `src/mapprojax/base.py`
*   Update `WCSBase.__init__` and `WCSArrayBase.__init__` to accept an optional `sip` argument (instance of `Sip`).
*   **Forward Transform (Pix -> Sky)** (`pix_to_native`):
    *   **Step 1**: Calculate relative pixels: $u = x - CRPIX1$, $v = y - CRPIX2$.
    *   **Step 2 (New)**: If SIP exists, apply `sip.pix_to_foc(u, v)` to get $u', v'$. Else $u'=u, v'=v$.
    *   **Step 3**: Apply CD matrix to $u', v'$ to get intermediate world coords $X, Y$.
    *   **Step 4**: Convert $X, Y$ (degrees) to radians and apply `plane_to_native`.
*   **Inverse Transform (Sky -> Pix)** (`native_to_pix`):
    *   **Step 1**: `native_to_plane` to get $X, Y$ (radians).
    *   **Step 2**: Convert to degrees.
    *   **Step 3**: Apply Inverse CD matrix to get linearized coordinates $U, V$.
    *   **Step 4 (New)**: If SIP exists, apply `sip.foc_to_pix(U, V)` to get $u, v$. Else $u=U, v=V$.
    *   **Step 5**: Add CRPIX: $x = u + CRPIX1$, $y = v + CRPIX2$.

### 3. JAX Support (`src/mapprojax/jax_base.py`)
*   Ensure the `Sip` class is compatible with JAX (using `jax.numpy` if passed as `xp`).
*   The `Sip` object itself might need to be a PyTree if it holds arrays. Register it with `jax.tree_util.register_pytree_node_class`.
*   Update `WCSJax` constructors to handle the `sip` object.

## Test Plan

### Test Suite: `tests/test_sip.py`
*   **Dependency**: `astropy` (specifically `astropy.wcs`).
*   **Setup**:
    *   Define a WCS header with SIP coefficients (A_ORDER, A_p_q, B_p_q, etc.).
    *   Create an `astropy.wcs.WCS` object.
    *   Create a `mapprojax.WCS` object with the same parameters manually (extracting coeffs into a `Sip` object).
*   **Tests**:
    1.  **Forward (Pix -> Sky)**:
        *   Generate a grid of pixel coordinates (e.g., using `np.meshgrid`).
        *   Run `astropy_wcs.all_pix2world`.
        *   Run `mapprojax_wcs.proj`.
        *   Assert results match within tolerance (e.g., 1e-10 deg).
    2.  **Inverse (Sky -> Pix)**:
        *   Use the sky coordinates from the previous step.
        *   Run `astropy_wcs.all_world2pix`.
        *   Run `mapprojax_wcs.unproj`.
        *   Assert results match within tolerance.
    3.  **Consistency**:
        *   Check that passing `sip=None` behaves exactly like the old implementation.

## Detailed SIP Logic
Per Shupe et al. (2005):
*   $f(u, v) = \sum_{p,q} A_{p,q} u^p v^q$ where $p+q \le A\_ORDER$
*   Implementation detail: `numpy.polynomial` or explicit sums. Explicit broadcasting might be faster/easier for JAX.
*   Coefficients are usually stored in the header as `A_p_q`. We will store them in a 2D array `A[p, q]`.
