import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any

#: Newton steps used for the SIP inverse. Two reach the fp64 floor from either
#: seed on the full SPHEREx detector (7.3e-9 then 3.1e-11 px from AP/BP,
#: 8.8e-3 then 4.5e-10 px from the identity), so 3 leaves one spare step.
DEFAULT_INVERSE_STEPS = 3


def _backend(xp, *values):
    """Return the array module to use.

    ``xp=np`` is the default so the NumPy callers need no ceremony, but a JAX
    tracer reaching a NumPy function is a hard error, so any JAX value wins.
    """
    if xp is not np:
        return xp
    for value in values:
        if isinstance(value, jax.Array):
            return jnp
    return np


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
    #: Newton steps for the inverse. Static, so it is aux data (not a leaf)
    #: and survives a flatten/unflatten unchanged.
    inverse_steps: int = DEFAULT_INVERSE_STEPS

    def __post_init__(self):
        # We don't enforce numpy arrays here to allow JAX arrays or lists.
        # But for consistency, if they are lists, we might want to convert?
        # Let's assume the caller handles conversion or we convert on the fly in methods if needed.
        pass

    def _evaluate_poly(self, u, v, coeffs, xp):
        """Evaluate ``sum(coeffs[p, q] * u**p * v**q)`` by Horner's rule.

        Nested division-free evaluation: for each row ``p`` the polynomial in
        ``v`` is evaluated by Horner, then those row results are combined by
        Horner in ``u``. That is O(order) multiplies per point instead of the
        O(order^2) power-table broadcast, which materialised
        ``u[..., None] ** p_range`` and dominated the whole transform on a
        real frame (498 ms vs 8.4 ms eager at 1201x1201, order 4).

        Coefficient indexing, broadcasting, an absent coefficient matrix and
        the input dtype are all preserved; the two forms agree to ~1e-13
        relative (floating-point association order differs, so not bitwise).
        """
        # Follow the array's backend rather than the `xp` default: under
        # `jax.jit` the coefficients are tracers, and `np.asarray(tracer)`
        # raises even though the caller only meant "use NumPy by default".
        xp = _backend(xp, coeffs, u, v)
        if coeffs is None:
            return xp.zeros_like(u)

        u = xp.asarray(u)
        v = xp.asarray(v)
        coeffs = xp.asarray(coeffs)
        # The coefficient *shape* is static (e.g. 4x4 for order 3); the
        # values may be tracers, so take the shape without converting.
        m, n = coeffs.shape

        def row(p):
            # Horner in v for one power of u.
            acc = coeffs[p, n - 1] * xp.ones_like(v)
            for q in range(n - 2, -1, -1):
                acc = acc * v + coeffs[p, q]
            return acc

        acc = row(m - 1)
        for p in range(m - 2, -1, -1):
            acc = acc * u + row(p)
        return acc

    def pix_to_foc(self, u, v, xp=np) -> Tuple:
        """
        Apply forward distortion (pixels to focal plane/intermediate).
        u, v are relative pixel coordinates (pixel - CRPIX).
        
        u' = u + f(u, v)
        v' = v + g(u, v)
        
        where f is poly A, g is poly B.
        """
        xp = _backend(xp, self.a, self.b, u, v)
        f = self._evaluate_poly(u, v, self.a, xp)
        g = self._evaluate_poly(u, v, self.b, xp)
        return u + f, v + g

    def _derivative(self, u, v, coeffs, axis, xp):
        """d/dv (axis=1) or d/du (axis=0) of ``_evaluate_poly``.

        Differentiates the coefficient matrix rather than the nested Horner
        expression: the powers of the untouched variable collapse to a
        weighted sum of rows, which keeps the O(order^2) work out of the
        per-Newton-step cost when the caller is JAX.
        """
        xp = _backend(xp, coeffs, u, v)
        if coeffs is None:
            return xp.zeros_like(u)
        coeffs = xp.asarray(coeffs)
        m, n = coeffs.shape
        if axis == 1:
            def row(p):
                acc = xp.zeros_like(v)
                for q in range(1, n):
                    acc = acc + q * coeffs[p, q] * v ** (q - 1)
                return acc
            powers = [u ** p for p in range(m)]
        else:
            def row(p):
                acc = xp.full_like(v, coeffs[p, n - 1])
                for q in range(n - 2, -1, -1):
                    acc = acc * v + coeffs[p, q]
                return acc
            powers = [p * u ** (p - 1) for p in range(1, m)]
        acc = xp.zeros_like(u)
        start = 1 if axis == 0 else 0
        for i, p in enumerate(range(start, m)):
            acc = acc + powers[i] * row(p)
        return acc

    def inverse_residual(self, u_pix, v_pix, u_foc, v_foc, xp=np):
        """Residual ``(u + f(u, v) - U, v + g(u, v) - V)`` in pixels.

        Public so a caller can test convergence of an approximate inverse
        without duplicating the polynomial evaluation.
        """
        xp = _backend(xp, self.a, self.b, u_pix, v_pix, u_foc, v_foc)
        u_pix = xp.asarray(u_pix)
        v_pix = xp.asarray(v_pix)
        r_u = u_pix + self._evaluate_poly(u_pix, v_pix, self.a, xp) - xp.asarray(u_foc)
        r_v = v_pix + self._evaluate_poly(u_pix, v_pix, self.b, xp) - xp.asarray(v_foc)
        return r_u, r_v

    def foc_to_pix(self, u, v, xp=np) -> Tuple:
        """Apply the inverse distortion (focal plane -> pixels).

        Solves ``u + f(u, v) = U``, ``v + g(u, v) = V`` by Newton's method in
        focal coordinates, with the Jacobian ``I + grad(f, g)``.

        The seed is AP/BP when present (they are SIP's own approximation of
        this inverse), otherwise the identity. AP/BP are only a seed: using
        them as the answer is 2.9e-2 px off on the full L2 detector.

        Points that do not converge, or where the Jacobian is singular or
        non-finite (the inverse has no usable derivative there), return NaN
        so a caller can mask them out. Steps are unrolled statically, so the
        whole solve is differentiable.
        """
        uu, vv, ok = self._foc_to_pix_valid(u, v, xp=xp)
        xp = _backend(xp, uu, vv)
        return xp.where(ok, uu, xp.nan), xp.where(ok, vv, xp.nan)

    def _jacobian_det(self, uu, vv, xp):
        j11 = 1 + self._derivative(uu, vv, self.a, 0, xp)
        j12 = self._derivative(uu, vv, self.a, 1, xp)
        j21 = self._derivative(uu, vv, self.b, 0, xp)
        j22 = 1 + self._derivative(uu, vv, self.b, 1, xp)
        return j11, j12, j21, j22, j11 * j22 - j12 * j21

    def _foc_to_pix_valid(self, u, v, xp=np) -> Tuple:
        """``(u_pix, v_pix, valid)`` with finite pixels everywhere.

        Used inside the WCS pipeline, which carries validity to its public
        outputs instead of inserting NaN here: a NaN returned mid-pipeline
        would reach ``+ crpix`` and later arithmetic and poison gradients.
        Invalid samples carry the focal coordinate itself as a stand-in.
        """
        xp = _backend(xp, self.a, self.b, self.ap, self.bp, u, v)
        u = xp.asarray(u)
        v = xp.asarray(v)
        if self.a is None and self.b is None:
            # No forward polynomial: AP/BP *is* the inverse.
            F = self._evaluate_poly(u, v, self.ap, xp)
            G = self._evaluate_poly(u, v, self.bp, xp)
            return u + F, v + G, xp.isfinite(u + F) & xp.isfinite(v + G)

        if self.ap is not None or self.bp is not None:
            uu = u + self._evaluate_poly(u, v, self.ap, xp)
            vv = v + self._evaluate_poly(u, v, self.bp, xp)
        else:
            uu, vv = u, v

        for _ in range(int(self.inverse_steps)):
            r_u, r_v = self.inverse_residual(uu, vv, u, v, xp)
            j11, j12, j21, j22, det = self._jacobian_det(uu, vv, xp)
            # Guard a singular or non-finite Jacobian without branching on a
            # tracer; the final check below rejects such points.
            bad = (det == 0) | ~xp.isfinite(det)
            safe_det = xp.where(bad, xp.ones_like(det), det)
            du = (j22 * r_u - j12 * r_v) / safe_det
            dv = (-j21 * r_u + j11 * r_v) / safe_det
            uu = uu - du
            vv = vv - dv

        r_u, r_v = self.inverse_residual(uu, vv, u, v, xp)
        *_, det = self._jacobian_det(uu, vv, xp)
        # dtype comes from a primal value, never np.asarray on a tracer.
        eps = jnp.finfo(uu.dtype).eps
        tol = 64 * eps * (1 + xp.abs(u) + xp.abs(v))
        # A fold of the forward map (det -> 0) has a root but no inverse
        # derivative; |det| ~ 1 for any real SIP, so 64 eps is far from it.
        valid = ((xp.abs(r_u) <= tol) & (xp.abs(r_v) <= tol)
                 & xp.isfinite(uu) & xp.isfinite(vv)
                 & xp.isfinite(det) & (xp.abs(det) > 64 * eps))
        return xp.where(valid, uu, u), xp.where(valid, vv, v), valid

    def to_dict(self):
        def _to_list(arr):
            return arr.tolist() if arr is not None else None
        
        return {
            'a': _to_list(self.a),
            'b': _to_list(self.b),
            'ap': _to_list(self.ap),
            'bp': _to_list(self.bp),
            'inverse_steps': self.inverse_steps,
        }
    
    @classmethod
    def from_dict(cls, data):
        def _to_array(lst):
            return np.array(lst) if lst is not None else None
        
        return cls(
            a=_to_array(data.get('a')),
            b=_to_array(data.get('b')),
            ap=_to_array(data.get('ap')),
            bp=_to_array(data.get('bp')),
            inverse_steps=data.get('inverse_steps', DEFAULT_INVERSE_STEPS),
        )
