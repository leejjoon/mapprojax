import re

import numpy as np
from .sip import Sip

#: Longitude/latitude CTYPE code pairs this loader supports. The latitude code
#: must be the matching one for the axis-1 longitude code.
_LAT_FOR_LON = {"RA--": "DEC-", "GLON": "GLAT", "ELON": "ELAT"}
_SUPPORTED_PROJECTIONS = ("TAN", "SIN")
_DEGREE_UNITS = ("deg", "degree", "degrees")


def _parse_ctype(header):
    """Validate ``CTYPE1``/``CTYPE2``; return the projection code.

    Accepts ``xxxx-PPP`` with an optional ``-SIP`` suffix, a supported
    longitude/latitude pair on the matching axes, and an equal ``PPP`` on
    both. Everything else -- CAR, swapped axes, mismatched projections --
    raises, because this loader cannot reproduce it.
    """
    ctype1 = str(header["CTYPE1"]).strip().upper()
    ctype2 = str(header["CTYPE2"]).strip().upper()

    def split(ctype, axis):
        base = ctype[:-4] if ctype.endswith("-SIP") else ctype
        if len(base) != 8 or base[4] != "-":
            raise ValueError(
                f"CTYPE{axis} = {ctype!r} is not an 8-character 'xxxx-PPP' code"
            )
        return base[:4], base[5:]

    lon_code, proj1 = split(ctype1, 1)
    lat_code, proj2 = split(ctype2, 2)

    if lon_code not in _LAT_FOR_LON:
        raise ValueError(
            f"CTYPE1 = {ctype1!r}: longitude code must be one of "
            f"{sorted(_LAT_FOR_LON)}, got {lon_code!r} (are the axes swapped?)"
        )
    if lat_code != _LAT_FOR_LON[lon_code]:
        raise ValueError(
            f"CTYPE2 = {ctype2!r}: expected latitude code "
            f"{_LAT_FOR_LON[lon_code]!r} to match CTYPE1 {lon_code!r}, got {lat_code!r}"
        )
    if proj1 != proj2:
        raise ValueError(
            f"CTYPE projections disagree: {ctype1!r} vs {ctype2!r}"
        )
    if proj1 not in _SUPPORTED_PROJECTIONS:
        raise ValueError(
            f"projection {proj1!r} is not supported; "
            f"this loader handles {list(_SUPPORTED_PROJECTIONS)}"
        )
    return proj1


def _check_cunit(header):
    """``CUNIT`` must be absent or degrees; anything else needs a conversion
    this loader does not apply."""
    for axis in (1, 2):
        unit = header.get(f"CUNIT{axis}")
        if unit is None:
            continue
        if str(unit).strip().lower() not in _DEGREE_UNITS:
            raise ValueError(
                f"CUNIT{axis} = {unit!r} is not degrees; the loader assumes degrees"
            )


def _check_lonpole(header, crval2):
    """The only orientation the rotation reproduces is the 180 degree one.

    WCSLIB's default depends on the reference declination: 180 at
    ``CRVAL2 = -90`` (accepted), 0 at ``CRVAL2 = +90`` (rejected unless the
    header says 180 explicitly), and 180 elsewhere. A rotation built for 180
    against a header whose effective value is 0 is wrong by an amount that
    grows with field size (141" on a 101 px field, 2828" at 1201^2).
    """
    if "LONPOLE" in header:
        lonpole = float(header["LONPOLE"])
        if abs(lonpole - 180.0) > 1e-12:
            raise ValueError(
                f"LONPOLE = {lonpole!r} is not supported; the celestial rotation "
                "is built for the default 180 degree orientation"
            )
        return
    if abs(float(crval2) - 90.0) < 1e-12:
        raise ValueError(
            "CRVAL2 = +90 with no explicit LONPOLE: WCSLIB's default there is 0, "
            "not 180. Add LONPOLE = 180 to the header if that is the orientation."
        )


def _check_pv(header, proj):
    """Only zero ``PV2_1``/``PV2_2`` on SIN is understood (it is a no-op).

    Cards belonging to an alternate WCS version (a trailing version letter,
    e.g. ``PV1_3W`` in the SPHEREx headers, which describes their
    ``CTYPE1W = 'WAVE-TAB'`` axis) are ignored: the loader only reads the
    primary version.
    """
    for key, value in header.items():
        key = str(key)
        if not key.startswith("PV"):
            continue
        if not re.fullmatch(r"PV\d+_\d+", key):
            continue  # alternate version, not the primary celestial WCS
        if proj == "SIN" and key in ("PV2_1", "PV2_2") and float(value) == 0.0:
            continue
        raise ValueError(f"{key} = {value!r} is not supported on {proj}")


def parse_sip_matrix(header, prefix, order_key):
    """
    Parse SIP coefficients from a FITS header.
    
    Args:
        header: dict-like object (e.g. astropy.io.fits.Header)
        prefix: str, e.g., 'A', 'B', 'AP', 'BP'
        order_key: str, e.g., 'A_ORDER'
        
    Returns:
        np.ndarray or None: The coefficient matrix.
    """
    if order_key not in header:
        return None
    
    order = header[order_key]
    matrix = np.zeros((order + 1, order + 1))
    
    for i in range(order + 1):
        for j in range(order + 1):
            key = f"{prefix}_{i}_{j}"
            if key in header:
                matrix[i, j] = header[key]
                
    return matrix

def load_wcs_params_from_header(header):
    """
    Extract WCS parameters (CRPIX, CD, CRVAL, SIP) from a FITS header.
    Converts 1-based FITS pixel coordinates (CRPIX) to 0-based Python coordinates.

    Only headers whose interpretation this package reproduces are accepted;
    keys it cannot honour raise ``ValueError`` rather than being silently
    ignored (see the rules in the module's helper functions).

    Args:
        header: dict-like object containing FITS keywords.
        
    Returns:
        tuple: (crpix, cd, crval, sip)
            crpix: list [x, y] (0-based)
            cd: list [[cd1_1, cd1_2], [cd2_1, cd2_2]]
            crval: list [lon, lat]
            sip: mapprojax.sip.Sip object or None

    Raises:
        ValueError: on a key this loader cannot interpret.
    """
    proj = _parse_ctype(header)
    _check_cunit(header)

    # 1. Parse SIP
    a = parse_sip_matrix(header, 'A', 'A_ORDER')
    b = parse_sip_matrix(header, 'B', 'B_ORDER')
    ap = parse_sip_matrix(header, 'AP', 'AP_ORDER')
    bp = parse_sip_matrix(header, 'BP', 'BP_ORDER')
    
    sip = None
    if any(x is not None for x in [a, b, ap, bp]):
        sip = Sip(a=a, b=b, ap=ap, bp=bp)
    
    # 2. Parse Linear WCS
    # CRPIX: FITS (1-based) -> Python (0-based)
    crpix = [header['CRPIX1'] - 1.0, header['CRPIX2'] - 1.0]
    
    # CRVAL
    crval = [header['CRVAL1'], header['CRVAL2']]
    _check_lonpole(header, crval[1])
    _check_pv(header, proj)

    # CD Matrix
    # Use CDi_j if any of the four keys is present, otherwise PCi_j * CDELTi.
    cd_keys = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    has_cd = any(key in header for key in cd_keys)
    has_pc = any(str(key).startswith("PC") and key != "PCOUNT" for key in header)
    if not has_cd and not has_pc and "CROTA2" in header:
        raise ValueError(
            "CROTA2 without CD/PC is not supported; the loader ignores it, "
            "which would misplace every pixel by the rotation angle"
        )

    if has_cd:
        cd = [
            [header.get('CD1_1', 0.0), header.get('CD1_2', 0.0)],
            [header.get('CD2_1', 0.0), header.get('CD2_2', 0.0)]
        ]
    else:
        # Fallback to PC + CDELT
        cdelt1 = header.get('CDELT1', 1.0)
        cdelt2 = header.get('CDELT2', 1.0)
        
        # PC default to Identity if not present
        pc1_1 = header.get('PC1_1', 1.0)
        pc1_2 = header.get('PC1_2', 0.0)
        pc2_1 = header.get('PC2_1', 0.0)
        pc2_2 = header.get('PC2_2', 1.0)
        
        cd = [
            [cdelt1 * pc1_1, cdelt1 * pc1_2],
            [cdelt2 * pc2_1, cdelt2 * pc2_2]
        ]
    
    return crpix, cd, crval, sip
