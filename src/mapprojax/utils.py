import numpy as np

def radec_to_xyz(ra_rad, dec_rad):
    """
    Convert RA/Dec (radians) to unit vector (x, y, z).
    """
    ra = ra_rad
    dec = dec_rad
    
    cos_dec = np.cos(dec)
    x = cos_dec * np.cos(ra)
    y = cos_dec * np.sin(ra)
    z = np.sin(dec)
    return x, y, z

def xyz_to_radec(x, y, z):
    """
    Convert unit vector (x, y, z) to RA/Dec (radians).
    """
    dist = np.sqrt(x*x + y*y + z*z)
    # Avoid division by zero
    # If dist is effectively zero, just return 0,0 (or handle gracefully)
    # But usually these are unit vectors.
    
    dec = np.arcsin(z / dist)
    ra = np.arctan2(y, x)
    
    # Normalize RA to [0, 2*pi)
    ra = ra % (2 * np.pi)
    
    return ra, dec

def rotation_matrix(lon_rad, lat_rad):
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
    
    cos_l = np.cos(lon_rad)
    sin_l = np.sin(lon_rad)
    cos_p = np.cos(p)
    sin_p = np.sin(p)
    
    # Row 0
    m00 = cos_p * cos_l
    m01 = cos_p * sin_l
    m02 = sin_p
    
    # Row 1
    m10 = -sin_l
    m11 = cos_l
    m12 = np.zeros_like(lon_rad)
    
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
    mat = np.array([
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
