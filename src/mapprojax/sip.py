import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Any

@dataclass
class Sip:
    """
    SIP (Simple Imaging Polynomial) distortion.
    
    Attributes:
        a (Any): Coefficients for the A polynomial (f(u, v)). Shape (M, N).
        b (Any): Coefficients for the B polynomial (g(u, v)). Shape (M, N).
        ap (Any): Coefficients for the AP polynomial (F(U, V)). Shape (M, N).
        bp (Any): Coefficients for the BP polynomial (G(U, V)). Shape (M, N).
    """
    a: Optional[Any] = None
    b: Optional[Any] = None
    ap: Optional[Any] = None
    bp: Optional[Any] = None

    def __post_init__(self):
        # We don't enforce numpy arrays here to allow JAX arrays or lists.
        # But for consistency, if they are lists, we might want to convert?
        # Let's assume the caller handles conversion or we convert on the fly in methods if needed.
        pass

    def _evaluate_poly(self, u, v, coeffs, xp):
        """
        Evaluate polynomial sum(coeffs[p, q] * u^p * v^q).
        """
        if coeffs is None:
            return xp.zeros_like(u)
        
        m, n = coeffs.shape
        
        # u, v can be arrays of any shape.
        # Construct powers.
        # u_powers: shape (..., m)
        # v_powers: shape (..., n)
        
        # xp.vander returns x^(N-1), ..., x^0. We want x^0, ..., x^(N-1).
        # But simplest is broadcasting: u[..., None] ** range
        
        # Ensure u, v are arrays of correct backend
        u = xp.asarray(u)
        v = xp.asarray(v)
        coeffs = xp.asarray(coeffs)
        
        # Range for powers
        # We need integer powers.
        p_range = xp.arange(m, dtype=xp.int32) if hasattr(xp, 'int32') else np.arange(m)
        q_range = xp.arange(n, dtype=xp.int32) if hasattr(xp, 'int32') else np.arange(n)
        
        u_pow = u[..., None] ** p_range
        v_pow = v[..., None] ** q_range
        
        # Equation: sum_{p,q} coeffs[p, q] * u^p * v^q
        # einsum: pq, ...p, ...q -> ...
        return xp.einsum('pq,...p,...q->...', coeffs, u_pow, v_pow)

    def pix_to_foc(self, u, v, xp=np) -> Tuple:
        """
        Apply forward distortion (pixels to focal plane/intermediate).
        u, v are relative pixel coordinates (pixel - CRPIX).
        
        u' = u + f(u, v)
        v' = v + g(u, v)
        
        where f is poly A, g is poly B.
        """
        f = self._evaluate_poly(u, v, self.a, xp)
        g = self._evaluate_poly(u, v, self.b, xp)
        return u + f, v + g

    def foc_to_pix(self, u, v, xp=np) -> Tuple:
        """
        Apply reverse distortion (focal plane/intermediate to pixels).
        u, v here are "linear" coordinates (U, V from CD inverse).
        
        u_pix = U + F(U, V)
        v_pix = V + G(U, V)
        
        where F is poly AP, G is poly BP.
        """
        F = self._evaluate_poly(u, v, self.ap, xp)
        G = self._evaluate_poly(u, v, self.bp, xp)
        return u + F, v + G

    def to_dict(self):
        def _to_list(arr):
            return arr.tolist() if arr is not None else None
        
        return {
            'a': _to_list(self.a),
            'b': _to_list(self.b),
            'ap': _to_list(self.ap),
            'bp': _to_list(self.bp)
        }
    
    @classmethod
    def from_dict(cls, data):
        def _to_array(lst):
            return np.array(lst) if lst is not None else None
        
        return cls(
            a=_to_array(data.get('a')),
            b=_to_array(data.get('b')),
            ap=_to_array(data.get('ap')),
            bp=_to_array(data.get('bp'))
        )
