import jax
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
    poles the arcsin form is sensitive to rounding as its argument
    approaches +/-1. The atan2 form uses z and the transverse component
    directly, without an angular-accuracy guarantee. The degenerate
    (0, 0, 0) input returns NaN instead of a valid sky position.
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
    
    From docs/notes/implementation.md (historical derivation):
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
    
    # Scalar inputs produce shape (3, 3); input shape S produces (3, 3, *S).
    mat = xp.array([
        [m00, m01, m02],
        [m10, m11, m12],
        [m20, m21, m22]
    ])
    
    # Rotation helpers below index the two leading matrix axes directly.
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


def _check_sequence_components(value, name):
    """Reject a host sequence holding a traced or narrower-float component.

    ``np.asarray`` would promote ``[np.float32(x), 30]`` to float64 and so
    hide the truncation the dtype check exists to catch. Python floats and
    integers carry no stored precision and are left to the array checks.
    """
    if isinstance(value, (np.ndarray, jax.Array)) or not isinstance(value, (list, tuple)):
        return
    for item in value:
        if isinstance(item, jax.core.Tracer):
            raise TypeError(f"{name} is traced and cannot be used on the host")
        if isinstance(item, (bool, int, float)):
            continue
        dtype = getattr(item, "dtype", None)
        if dtype is not None and np.dtype(dtype).kind == "f" and np.dtype(dtype) != np.float64:
            raise TypeError(
                f"{name} has a {np.dtype(dtype)} component; host references must "
                "be float64 throughout, and assembling the sequence would hide "
                "the lost precision"
            )


def _host_reference_rotation(crval, name):
    """Build a host rotation from an actual float64 degree-valued reference.

    The reference must be a concrete, finite ``(2,)`` array whose dtype is
    already float64: a float32 or traced reference is rejected rather than
    upcast, since an upcast cannot restore precision lost earlier.
    """
    if isinstance(crval, jax.core.Tracer):
        raise TypeError(
            f"{name} is traced; build the relative matrix in JAX and pass it "
            "to Reprojector.from_matrix"
        )
    _check_sequence_components(crval, name)
    try:
        value = np.asarray(crval)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a concrete float64 array") from exc
    if value.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {value.shape}")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have true float64 dtype, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return rotation_matrix(np.radians(value[0]), np.radians(value[1]), xp=np)


def compose_relative_rotation(wcs_src, wcs_tgt, frame_rotation=None, *,
                              crval_src=None, crval_tgt=None):
    """
    Compose the direct native(src) -> native(tgt) rotation
    ``M_rel = M_tgt @ frame_rotation @ M_src.T``, entirely in float64 on
    the host.

    This is the step that lets a fused src-pixel -> tgt-pixel reprojection
    (see :class:`mapprojax.reproject.Reprojector`) avoid ever forming an
    absolute celestial unit vector on the per-pixel path: with ``M_rel``
    precomputed once here, the bulk per-pixel work is
    ``pix_to_native -> apply M_rel -> native_to_pix``, which stays in the
    field-relative regime throughout. This supports an fp32 per-pixel path,
    but does not guarantee an error bound for arbitrary fields or inputs;
    see docs/user_manual.md for the precision modes.

    Parameters
    ----------
    wcs_src, wcs_tgt : conforming WCS
        Any WCS exposing a concrete float64 ``crval`` pair in degrees. The
        rotation is rebuilt from those host references in true float64,
        independent of the WCS kernel precision and ``jax_enable_x64``.
    frame_rotation : array, shape (3, 3), optional
        Finite rotation converting a celestial unit vector from ``wcs_src``'s
        reference frame into ``wcs_tgt``'s (e.g. ICRS -> Galactic).
        Defaults to the identity (both WCSes share a frame).
    crval_src, crval_tgt : array, shape (2,), float64, optional, keyword-only
        Original host references in degrees, used instead of the
        corresponding ``wcs.crval``. Use them when a WCS has passed through a
        transformation (for example ``jit`` with x64 off) that left its own
        ``crval`` traced or float32. They are validated like ``wcs.crval``;
        nothing can prove that an override describes the intended frame, so
        retaining the original references is the caller's responsibility.

    Returns
    -------
    M_rel : numpy.ndarray, shape (3, 3), dtype float64

    Typical recovery after a transformation::

        m = compose_relative_rotation(src, tgt, crval_src=host_src,
                                      crval_tgt=host_tgt)
        rp = Reprojector.from_matrix(src, tgt, m, matrix_precision="f32")
    """
    m_src = _host_reference_rotation(
        wcs_src.crval if crval_src is None else crval_src,
        "wcs_src.crval" if crval_src is None else "crval_src",
    )
    m_tgt = _host_reference_rotation(
        wcs_tgt.crval if crval_tgt is None else crval_tgt,
        "wcs_tgt.crval" if crval_tgt is None else "crval_tgt",
    )
    if frame_rotation is None:
        return m_tgt @ m_src.T
    if isinstance(frame_rotation, jax.core.Tracer):
        raise TypeError("frame_rotation must be a concrete host matrix")
    try:
        frame_rotation = np.asarray(frame_rotation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("frame_rotation must be a concrete host matrix") from exc
    if frame_rotation.shape != (3, 3) or not np.all(np.isfinite(frame_rotation)):
        raise ValueError("frame_rotation must be a finite (3, 3) matrix")
    return m_tgt @ frame_rotation @ m_src.T
