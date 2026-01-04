# Python Library Implementation Guide for WCS Projections

This document outlines the architecture and mathematical details required to implement a Python version of the `mapproj` library. It focuses on the `TAN` (Gnomonic) and `SIN` (Orthographic) projections as requested.

## Architecture Overview

The library transforms coordinates between **Pixel Coordinates** (image space) and **Sky Coordinates** (Right Ascension/Declination). The transformation pipeline consists of four main stages:

1.  **Pixel to Projection Plane**: Transforms pixel coordinates $(x, y)$ to intermediate projection coordinates $(X, Y)$ using linear transformation parameters (CRPIX, CD matrix).
2.  **Deprojection (Inverse Projection)**: Transforms projection coordinates $(X, Y)$ to a unit vector $(x, y, z)$ on the "native" sphere.
    *   **Crucial Convention**: Unlike the standard WCS paper which defines projections relative to the North Pole $(0, 0, 1)$, this implementation uses the **Vernal Point $(1, 0, 0)$** as the default center of the native projection.
3.  **Rotation**: Rotates the unit vector from the native frame to the celestial frame based on the reference values (CRVAL).
4.  **Cartesian to Spherical**: Converts the celestial unit vector to Longitude (RA) and Latitude (Dec).

### Data Structures

*   **Pixel Point**: `ImgXY` $(x_{pix}, y_{pix})$
*   **Projection Point**: `ProjXY` $(X_{proj}, Y_{proj})$
*   **Unit Vector**: `XYZ` $(x, y, z)$ where $x^2 + y^2 + z^2 = 1$
*   **Sky Coordinate**: `LonLat` $(\alpha, \delta)$ (in radians)

---

## 1. Pixel to Projection Plane (Linear Transformation)

This step applies the WCS linear transformation to convert image pixels to intermediate world coordinates.

**Input:**
*   Pixel coordinate: $p = (x_{img}, y_{img})$
*   Header Parameters:
    *   `CRPIX1`, `CRPIX2`: Reference pixel coordinates.
    *   `CD1_1`, `CD1_2`, `CD2_1`, `CD2_2`: Linear transformation matrix (including scale and rotation).

**Transformation:**
$$
\begin{pmatrix} X_{proj} \\ Y_{proj} \end{pmatrix} = 
\begin{pmatrix} CD_{1\_1} & CD_{1\_2} \\ CD_{2\_1} & CD_{2\_2} \end{pmatrix} 
\times 
\begin{pmatrix} x_{img} - \text{CRPIX1} \\ y_{img} - \text{CRPIX2} \end{pmatrix}
$$

**Inverse Transformation (Projection Plane to Pixel):**
$$
\begin{pmatrix} x_{img} \\ y_{img} \end{pmatrix} = 
\text{CD}^{-1} \times \begin{pmatrix} X_{proj} \\ Y_{proj} \end{pmatrix} + \begin{pmatrix} \text{CRPIX1} \\ \text{CRPIX2} \end{pmatrix}
$$
where $\text{CD}^{-1}$ is the inverse of the CD matrix.

---

## 2. Deprojection (Projection Plane to Native Sphere)

This step maps the 2D plane coordinates $(X, Y)$ to a 3D unit vector $(x, y, z)$ on the native sphere.
**Note**: The native sphere is oriented such that the projection center is at $(1, 0, 0)$.

### TAN - Gnomonic Projection

*   **Center**: $(1, 0, 0)$
*   **Plane**: Tangent at $x=1$.
*   **Axes**: $Y$ aligns with native $y$, $Z$ aligns with native $z$.

**Forward Projection (Sphere -> Plane):**
Given native unit vector $(x, y, z)$:
$$
X_{proj} = y / x \\
Y_{proj} = z / x
$$
*Condition*: Valid only if $x > 0$.

**Deprojection (Plane -> Sphere):**
Given $(X_{proj}, Y_{proj})$:
1.  Calculate scaling factor $r$:
    $$ r = \sqrt{1 + X_{proj}^2 + Y_{proj}^2} $$
2.  Compute vector components:
    $$
    x = 1 / r \\
    y = X_{proj} / r \\
    z = Y_{proj} / r
    $$

### SIN - Orthographic Projection

*   **Center**: $(1, 0, 0)$
*   **Plane**: The $yz$-plane ($x=0$).

**Forward Projection (Sphere -> Plane):**
Given native unit vector $(x, y, z)$:
$$
X_{proj} = y \\
Y_{proj} = z
$$
*Condition*: Valid only if $x \ge 0$ (front hemisphere).

**Deprojection (Plane -> Sphere):**
Given $(X_{proj}, Y_{proj})$:
1.  Check bounds:
    $$ R^2 = X_{proj}^2 + Y_{proj}^2 $$
    If $R^2 > 1$, the point is undefined.
2.  Compute vector components:
    $$
    x = \sqrt{1 - R^2} \\
    y = X_{proj} \\
    z = Y_{proj}
    $$

---

## 3. Rotation (Native Sphere to Celestial Sphere)

We must rotate the native vector so that the native center $(1, 0, 0)$ maps to the reference sky position $(\alpha_0, \delta_0)$ defined by `CRVAL1` and `CRVAL2`.

Let $(\lambda, \phi) = (\text{CRVAL1}, \text{CRVAL2})$ (converted to radians).

**Rotation Matrix $M$ (Celestial -> Native):**
The matrix that rotates a celestial vector into the native frame (where center is $(1,0,0)$) is:
$$
M = \begin{pmatrix}
\cos\phi \cos\lambda & \cos\phi \sin\lambda & \sin\phi \\
-\sin\lambda & \cos\lambda & 0 \\
-\sin\phi \cos\lambda & -\sin\phi \sin\lambda & \cos\phi
\end{pmatrix}
$$

**Native to Celestial (Deprojection Step):**
To go from Native $(x_n, y_n, z_n)$ to Celestial $(x_c, y_c, z_c)$, we use the transpose (inverse) of $M$:

$$
\begin{pmatrix} x_c \\ y_c \\ z_c \end{pmatrix} = 
\begin{pmatrix}
\cos\phi \cos\lambda & -\sin\lambda & -\sin\phi \cos\lambda \\
\cos\phi \sin\lambda & \cos\lambda & -\sin\phi \sin\lambda \\
\sin\phi & 0 & \cos\phi
\end{pmatrix}
\begin{pmatrix} x_n \\ y_n \\ z_n \end{pmatrix}
$$

**Celestial to Native (Projection Step):**
$$
\begin{pmatrix} x_n \\ y_n \\ z_n \end{pmatrix} = M \times \begin{pmatrix} x_c \\ y_c \\ z_c \end{pmatrix}
$$

---

## 4. Cartesian to Spherical Conversion

Convert the celestial unit vector $(x, y, z)$ to Right Ascension $(\alpha)$ and Declination $(\delta)$.

$$
\delta = \arcsin(z)
$$
$$
\alpha = \text{arctan2}(y, x)
$$

*   Ensure $\alpha$ is normalized to $[0, 2\pi)$.
*   $\delta$ will be in $[-\pi/2, \pi/2]$.

---

## Proposed Python Architecture

To implement this in Python, I recommend the following class structure:

### 1. `Projection` (Abstract Base Class)
*   **Methods**:
    *   `proj(x, y, z) -> (X, Y)`: Projects native unit vector to plane.
    *   `unproj(X, Y) -> (x, y, z)`: Deprojects plane coordinates to native unit vector.
    *   `bounds() -> (x_range, y_range)`: Returns valid ranges.

### 2. `Tan` and `Sin` (Subclasses of Projection)
*   Implement the specific mathematical formulas described in Section 2.

### 3. `WCS` (Main Class)
*   **Attributes**:
    *   `crpix`: Tuple (crpix1, crpix2)
    *   `cd_matrix`: 2x2 numpy array or tuple of 4 floats.
    *   `crval`: Tuple (lon, lat) in radians.
    *   `projection`: Instance of a `Projection` subclass (e.g., `Tan()`).
*   **Methods**:
    *   `pixel_to_sky(x_img, y_img)`:
        1.  Apply linear transform -> $(X, Y)$.
        2.  `self.projection.unproj(X, Y)` -> $(x_n, y_n, z_n)$.
        3.  Apply rotation matrix (derived from `crval`) -> $(x_c, y_c, z_c)$.
        4.  Convert to $(\alpha, \delta)$.
    *   `sky_to_pixel(alpha, delta)`:
        1.  Convert to $(x_c, y_c, z_c)$.
        2.  Apply rotation matrix (inverse) -> $(x_n, y_n, z_n)$.
        3.  `self.projection.proj(x_n, y_n, z_n)` -> $(X, Y)$.
        4.  Apply inverse linear transform -> $(x_{img}, y_{img})$.
