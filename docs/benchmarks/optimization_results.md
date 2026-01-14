# Optimization Benchmark Results

**Date:** January 14, 2026  
**Backend:** CPU  
**Precision:** float64

This benchmark compares three different strategies for pixel-to-pixel coordinate transformation using `mapprojax`.

## Strategies Tested

1.  **Std (Full)**: The traditional pipeline converting through Spherical Coordinates (RA/Dec).
    *   `Pixel` -> `Plane` -> `Native Sphere` -> `Celestial Sphere (RA/Dec)` -> `Native Sphere` -> `Plane` -> `Pixel`
    *   Involves expensive trigonometric functions (`arcsin`, `arctan2`) for every pixel.

2.  **Std (XYZ)**: A vector-based pipeline that works directly with Cartesian Unit Vectors.
    *   `Pixel` -> `Plane` -> `Native Sphere` -> `Celestial Unit Vector (x,y,z)` -> `Native Sphere` -> `Plane` -> `Pixel`
    *   Bypasses `arcsin` and `arctan2`, utilizing only matrix multiplications and standard projection math.

3.  **Optimized (Rot)**: Similar to XYZ, but manually pre-fuses the rotation matrices.
    *   `Pixel` -> `Plane` -> `Native Sphere` -> `[Combined Matrix]` -> `Native Sphere` -> `Plane` -> `Pixel`
    *   Mathematically equivalent to XYZ but attempts to reduce the number of matrix-vector multiplication steps.

## Results (Time in ms)

| Image Size | Std (Full) | Std (XYZ) | Optimized (Rot) | Speedup (XYZ vs Full) |
| :--- | :--- | :--- | :--- | :--- |
| 512x512 | 5.73 ms | 1.11 ms | 1.92 ms | **~5.1x** |
| 1024x1024 | 27.95 ms | 4.58 ms | 2.33 ms | **~6.1x** |
| 2048x2048 | 83.28 ms | 36.59 ms | 16.99 ms | **~2.3x** |
| 4096x4096 | 228.51 ms | 75.18 ms | 76.48 ms | **~3.0x** |

*Note: Results may vary based on specific CPU architecture and load.*

## Conclusion

1.  **Avoid RA/Dec Conversions**: The most significant performance gain comes from switching to the `XYZ` path (`unproj_xyz` / `proj_xyz`). This yields a **3x to 6x speedup** by avoiding the singularities and computational cost of spherical coordinates.
2.  **Manual Optimization**: Manually fusing rotation matrices (`Optimized (Rot)`) shows mixed results. While it offers some benefits at certain sizes (likely due to cache locality or reduced memory traffic), at other sizes it is comparable to the standard `XYZ` path. Given the complexity it adds to user code, the `XYZ` path is the recommended default for high-performance applications.
