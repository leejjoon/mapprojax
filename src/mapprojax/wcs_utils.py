import numpy as np
from .sip import Sip

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
    
    Args:
        header: dict-like object containing FITS keywords.
        
    Returns:
        tuple: (crpix, cd, crval, sip)
            crpix: list [x, y] (0-based)
            cd: list [[cd1_1, cd1_2], [cd2_1, cd2_2]]
            crval: list [lon, lat]
            sip: mapprojax.sip.Sip object or None
    """
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
    
    # CD Matrix
    # Use CDi_j if present, otherwise PCi_j * CDELTi
    if 'CD1_1' in header:
        cd = [
            [header['CD1_1'], header.get('CD1_2', 0.0)],
            [header.get('CD2_1', 0.0), header['CD2_2']]
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
