import numpy as np

def radec_to_xyz(ra_rad, dec_rad, xp=np):
    """
    Convert RA/Dec (radians) to unit vector (x, y, z).
    """
    ra = ra_rad
    dec = dec_rad
    
    cos_dec = xp.cos(dec)
    x = cos_dec * xp.cos(ra)
    y = cos_dec * xp.sin(ra)
    z = xp.sin(dec)
    return x, y, z

def xyz_to_radec(x, y, z, xp=np):
    """
    Convert unit vector (x, y, z) to RA/Dec (radians).

    ``dec = atan2(z, hypot(x, y))`` rather than ``arcsin(z / |v|)``: near the
    poles the arcsin form loses precision (1.5e-4 to 2.0e-4 mas at dec +/-
    89.9) because it differentiates the whole sphere through one coordinate,
    while atan2 keeps the full precision of z against the small transverse
    component. The degenerate (0, 0, 0) input returns NaN instead of a valid
    sky position.
    """
    dist = xp.sqrt(x*x + y*y + z*z)
    dec = xp.arctan2(z, xp.hypot(x, y))
    ra = xp.arctan2(y, x)

    # Normalize RA to [0, 2*pi)
    ra = ra % (2 * np.pi)

    # A zero-length vector has no direction; without this it would come back
    # as a valid-looking (0, 0).
    invalid = dist == 0
    return xp.where(invalid, xp.nan, ra), xp.where(invalid, xp.nan, dec)

def rotation_matrix(lon_rad, lat_rad, xp=np):
    """
    Generate the rotation matrix M that rotates a celestial vector
    into the native frame (where center is (1,0,0)).
    
    Defined by CRVAL = (lon, lat) in radians.
    
    From implementation.md:
    M = [
      [ cosP cosL,  cosP sinL, sinP],
      [-sinL,       cosL,      0   ],
      [-sinP cosL, -sinP sinL, cosP]
    ]
    where L = lon (alpha), P = lat (delta).
    """
    p = lat_rad # phi
    
    cos_l = xp.cos(lon_rad)
    sin_l = xp.sin(lon_rad)
    cos_p = xp.cos(p)
    sin_p = xp.sin(p)
    
    # Row 0
    m00 = cos_p * cos_l
    m01 = cos_p * sin_l
    m02 = sin_p
    
    # Row 1
    m10 = -sin_l
    m11 = cos_l
    m12 = xp.zeros_like(lon_rad)
    
    # Row 2
    m20 = -sin_p * cos_l
    m21 = -sin_p * sin_l
    m22 = cos_p
    
    # Shape: (3, 3) or (..., 3, 3) if input is array
    # Constructing assuming scalar first, we can optimize for arrays later 
    # if we want this util to handle arrays.
    # For WCSArray, lon_deg/lat_deg will be arrays.
    
    # Use np.stack to build the matrix.
    # If inputs are scalars, result is (3,3).
    # If inputs are shape (N,), result is (3, 3, N) -> need to transpose to (N, 3, 3)?
    # Let's align with what we need.
    
    # If we construct it simply:
    mat = xp.array([
        [m00, m01, m02],
        [m10, m11, m12],
        [m20, m21, m22]
    ])
    
    # If inputs were arrays of shape S, mat has shape (3, 3, S).
    # We usually want (S, 3, 3) for matmul? Or we just handle it manually.
    # Let's return (3, 3, ...) and handle axes in usage.
    return mat

def apply_rotation(matrix, x, y, z):
    """
    Apply rotation matrix M to vector v=(x,y,z).
    v_new = M * v
    
    matrix: (3, 3) or (3, 3, N)
    x, y, z: scalars or arrays
    """
    # x, y, z -> v shape (3, ...)
    # result = sum(M[i, j] * v[j])
    
    xn = matrix[0,0]*x + matrix[0,1]*y + matrix[0,2]*z
    yn = matrix[1,0]*x + matrix[1,1]*y + matrix[1,2]*z
    zn = matrix[2,0]*x + matrix[2,1]*y + matrix[2,2]*z
    
    return xn, yn, zn

def apply_rotation_transpose(matrix, x, y, z):
    """
    Apply transpose of rotation matrix M to vector v.
    v_new = M.T * v
    """
    # Transpose implies swapping indices 0 and 1 of the matrix
    xn = matrix[0,0]*x + matrix[1,0]*y + matrix[2,0]*z
    yn = matrix[0,1]*x + matrix[1,1]*y + matrix[2,1]*z
    zn = matrix[0,2]*x + matrix[1,2]*y + matrix[2,2]*z

    return xn, yn, zn


def compose_relative_rotation(wcs_src, wcs_tgt, frame_rotation=None):
    """
    Compose the direct native(src) -> native(tgt) rotation
    ``M_rel = M_tgt @ frame_rotation @ M_src.T``, entirely in float64 on
    the host.

    This is the step that lets a fused src-pixel -> tgt-pixel reprojection
    (see :class:`mapprojax.reproject.Reprojector`) avoid ever forming an
    absolute celestial unit vector on the per-pixel path: with ``M_rel``
    precomputed once here, the bulk per-pixel work is
    ``pix_to_native -> apply M_rel -> native_to_pix``, which stays in the
    field-relative regime throughout and is safe in fp32 (see
    ``fo64_report_and_plan.md`` and DEVELOPER.md's Precision Strategy
    section for why the celestial<->native rotation is otherwise the one
    fp64-mandatory step).

    Parameters
    ----------
    wcs_src, wcs_tgt : WCSBase (NumPy-backed, e.g. Tan/Sin)
        Must be NumPy-backed (``wcs.xp is numpy``), not a JAX WCS: this
        composition must run in true float64 regardless of
        ``jax_enable_x64``, and a JAX WCS's own ``r_matrix`` may already
        be truncated to float32 if x64 was never enabled.
    frame_rotation : array, shape (3, 3), optional
        Rotation converting a celestial unit vector from ``wcs_src``'s
        reference frame into ``wcs_tgt``'s (e.g. ICRS -> Galactic).
        Defaults to the identity (both WCSes share a frame).

    Returns
    -------
    M_rel : numpy.ndarray, shape (3, 3), dtype float64
    """
    for name, wcs in (("wcs_src", wcs_src), ("wcs_tgt", wcs_tgt)):
        if wcs.xp is not np:
            raise TypeError(
                f"{name} must be a NumPy-backed WCS (e.g. Tan/Sin), not a JAX one: "
                "this composition must run in true float64 regardless of "
                "jax_enable_x64, and a JAX WCS's r_matrix may already be "
                "truncated to float32."
            )

    m_src = np.asarray(wcs_src.r_matrix, dtype=np.float64)
    m_tgt = np.asarray(wcs_tgt.r_matrix, dtype=np.float64)
    if frame_rotation is None:
        return m_tgt @ m_src.T
    frame_rotation = np.asarray(frame_rotation, dtype=np.float64)
    return m_tgt @ frame_rotation @ m_src.T
