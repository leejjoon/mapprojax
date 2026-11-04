"""Scale-aware derivative checks for ``Tan``/``Sin`` in every precision mode.

Every leaf is perturbed as ``base + scale * delta`` so each Jacobian block is
O(0.1-10) per unit ``delta`` (outputs: sky offsets in arcsec for the forward
path, pixels for the inverse path), and one tolerance pair fits all blocks.

``check_grads`` (forward and reverse) runs in every mode on the forward
(``unproj_offset``) and inverse (``proj_offset``) offset paths, and in mixed
and f64 on the absolute ``unproj``/``proj`` path:

* ``f64``: the existing step and tolerances, eps 1e-3, atol 1e-6, rtol 1e-5.
* ``mixed`` and ``f32``: the step is scaled up and the tolerance has an
  absolute floor, because a central difference divides the float32 rounding
  of the *output* by ``2 * eps``. The forward offsets reach ~5300 arcsec, so
  one float32 ulp is ~6e-4 arcsec and the measured finite-difference noise at
  eps 1 is 3e-3 to 1e-2 arcsec per leaf, while the same difference in f64
  agrees with AD to 1e-7 (truncation is negligible at these steps). Measured
  minimum passing ``atol = rtol`` over all cases (CPU, Nov 4 2026): mixed
  1e-3 at eps 0.3; f32 1e-3 at eps 1, except the AP/BP-only forward path at
  3e-3 (Tan and Sin; the same rounding floor, hit by check_grads' random
  tangent). The gates below keep at least a 3x margin.

The mixed-versus-f64 (and f32-versus-f64) Jacobian difference is *reported*,
not gated: the measured values are recorded below, and the test only rejects a
catastrophic divergence. Forward and reverse mode are also checked for
consistency (``<ct, J t> == <J^T ct, t>``) within each mode.

SIP cases: ``ab_seeded`` has A/B plus a crude AP/BP seed refined by Newton
steps, so the AP/BP derivative is zero to rounding (the converged inverse
does not depend on its seed); ``ap_bp_only`` has no A/B, so AP/BP *is* the
inverse and its derivative is non-trivial.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from mapprojax.wcs import Tan, Sin
from mapprojax.sip import Sip


@pytest.fixture(autouse=True)
def _x64_enabled():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


CRPIX = jnp.array([1019.5, 1019.5])
CD = jnp.array([[-1.6e-3, 1.0e-5], [2.0e-5, 1.66e-3]])
CRVAL = jnp.array([306.77, 39.38])
A = (jnp.zeros((4, 4)).at[2, 0].set(2e-6).at[0, 2].set(-1.5e-6).at[1, 1].set(1e-6)
     .at[3, 0].set(3e-10).at[1, 2].set(-2e-10))
B = (jnp.zeros((4, 4)).at[0, 2].set(1.8e-6).at[2, 0].set(-1e-6).at[1, 1].set(-1.2e-6)
     .at[0, 3].set(2.5e-10).at[2, 1].set(1e-10))
PIX = jnp.array([[300.0, 1700.0], [1500.0, 400.0], [900.0, 1100.0], [1800.0, 1900.0],
                 [1019.5, 1300.0]])
ARCSEC = 180.0 / np.pi * 3600.0

_p, _q = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
SIP_SCALE = jnp.asarray(np.where((_p + _q >= 2) & (_p + _q <= 3), 1.0 / 1000.0 ** (_p + _q), 0.0))
SCALE = dict(crpix=1.0, cd=1e-3, crval=1e-3, a=SIP_SCALE, b=SIP_SCALE, ap=SIP_SCALE, bp=SIP_SCALE)
CLASSES = [Tan, Sin]

# Per-mode check_grads: (eps, atol, rtol). See the module docstring.
CHECK_GRADS = {
    "f64": (1e-3, 1e-6, 1e-5),
    "mixed": (0.3, 3e-3, 3e-3),
    "f32": (1.0, 1e-2, 1e-2),
}

# Reported, not gated. Measured on CPU and ROCm (Nov 4 2026), max over
# Tan/Sin, both SIP cases and every path, of max|J_mode - J_f64| / max|J_f64|
# per leaf block:
#   mixed: <= 4.2e-7 (all leaves, crval <= 1.8e-7)
#   f32:   <= 3.8e-6 (crpix, cd, A/B, AP/BP), <= 1.7e-4 (crval, whose block is
#          small, ~1e-3, because crval enters offsets only through cos(lat0))
# The test only rejects a catastrophic divergence.
JACOBIAN_DIVERGENCE = 1e-2
# Forward/reverse inconsistency measured <= 4.0e-7 mixed, <= 1.8e-6 f32.
TRANSPOSE_RTOL = 1e-5


def _zeros(case):
    d = dict(crpix=jnp.zeros(2), cd=jnp.zeros((2, 2)), crval=jnp.zeros(2),
             ap=jnp.zeros((4, 4)), bp=jnp.zeros((4, 4)))
    if case == "ab_seeded":
        d.update(a=jnp.zeros((4, 4)), b=jnp.zeros((4, 4)))
    return d


def _make(cls, precision, case, d):
    ap = -A + SCALE["ap"] * d["ap"]
    bp = -B + SCALE["bp"] * d["bp"]
    if case == "ab_seeded":
        sip = Sip(a=A + SCALE["a"] * d["a"], b=B + SCALE["b"] * d["b"], ap=ap, bp=bp)
    else:
        sip = Sip(ap=ap, bp=bp)
    return cls(CRPIX + SCALE["crpix"] * d["crpix"], CD * (1.0 + SCALE["cd"] * d["cd"]),
               CRVAL + SCALE["crval"] * d["crval"], sip=sip, precision=precision)


def _paths(cls, precision, case):
    """The differentiable functions of ``delta`` for one class/mode/case."""
    base = _make(cls, "f64", case, _zeros(case))
    offset = base.unproj_offset(PIX[:, 0], PIX[:, 1])
    sky = base.unproj(PIX[:, 0], PIX[:, 1])

    def forward_offset(d):
        u, v = _make(cls, precision, case, d).unproj_offset(PIX[:, 0], PIX[:, 1])
        return jnp.stack([u, v]).astype(jnp.float64) * ARCSEC

    def inverse_offset(d):
        x, y = _make(cls, precision, case, d).proj_offset(*offset)
        return jnp.stack([x, y]).astype(jnp.float64)

    def forward_absolute(d):
        ra, dec = _make(cls, precision, case, d).unproj(PIX[:, 0], PIX[:, 1])
        out = jnp.stack([ra - jnp.radians(CRVAL[0]), dec - jnp.radians(CRVAL[1])])
        return out.astype(jnp.float64) * ARCSEC

    def inverse_absolute(d):
        x, y = _make(cls, precision, case, d).proj(*sky)
        return jnp.stack([x, y]).astype(jnp.float64)

    paths = dict(forward_offset=forward_offset, inverse_offset=inverse_offset)
    if precision != "f32":
        paths.update(forward_absolute=forward_absolute, inverse_absolute=inverse_absolute)
    return paths


PATHS = ["forward_offset", "inverse_offset", "forward_absolute", "inverse_absolute"]
CASES = ["ab_seeded", "ap_bp_only"]
# f32 has no absolute celestial path, so it is checked on the offset paths.
MODE_CASES = [
    pytest.param(cls, precision, case, path,
                 id=f"{cls.__name__}-{precision}-{case}-{path}")
    for cls in CLASSES for precision in ("mixed", "f32") for case in CASES
    for path in PATHS if precision == "mixed" or path.endswith("_offset")
]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("cls", CLASSES)
def test_f64_check_grads_every_leaf(cls, case, path):
    f = _paths(cls, "f64", case)[path]
    d0 = _zeros(case)
    assert np.all(np.isfinite(np.asarray(f(d0))))
    eps, atol, rtol = CHECK_GRADS["f64"]
    check_grads(f, (d0,), order=1, modes=("fwd", "rev"), eps=eps, atol=atol, rtol=rtol)


@pytest.mark.parametrize("cls", CLASSES)
def test_ap_bp_derivatives(cls):
    """Zero once Newton has converged on A/B; non-trivial when AP/BP is the
    inverse."""
    seeded = jax.jacrev(_paths(cls, "f64", "ab_seeded")["inverse_offset"])(_zeros("ab_seeded"))
    assert float(jnp.abs(seeded["a"]).max()) > 0.1
    for leaf in ("ap", "bp"):
        assert float(jnp.abs(seeded[leaf]).max()) < 1e-10, leaf
    direct = jax.jacrev(_paths(cls, "f64", "ap_bp_only")["inverse_offset"])(_zeros("ap_bp_only"))
    for leaf in ("ap", "bp"):
        assert float(jnp.abs(direct[leaf]).max()) > 0.1, leaf


@pytest.mark.parametrize("cls,precision,case,path", MODE_CASES)
def test_mode_check_grads_every_leaf(cls, precision, case, path):
    f = _paths(cls, precision, case)[path]
    d0 = _zeros(case)
    assert np.all(np.isfinite(np.asarray(f(d0))))
    eps, atol, rtol = CHECK_GRADS[precision]
    check_grads(f, (d0,), order=1, modes=("fwd", "rev"), eps=eps, atol=atol, rtol=rtol)


@pytest.mark.parametrize("cls,precision,case,path", MODE_CASES)
def test_mode_jacobian_against_f64_is_reported(cls, precision, case, path):
    paths = _paths(cls, precision, case)
    d0 = _zeros(case)
    jac = jax.jacrev(paths[path])(d0)
    ref = jax.jacrev(_paths(cls, "f64", case)[path])(d0)
    for leaf, block in jac.items():
        assert np.all(np.isfinite(np.asarray(block))), leaf
        scale = float(jnp.abs(ref[leaf]).max())
        error = float(jnp.abs(block - ref[leaf]).max())
        if scale < 1e-10:
            # A vanishing f64 block (AP/BP behind converged Newton steps)
            # must vanish in the mode too, relative to the leading blocks.
            assert error < 1e-10, leaf
            continue
        assert error <= JACOBIAN_DIVERGENCE * scale, (leaf, error / scale)


@pytest.mark.parametrize("cls,precision,case,path", MODE_CASES)
def test_mode_forward_and_reverse_agree(cls, precision, case, path):
    paths = _paths(cls, precision, case)
    f = paths[path]
    d0 = _zeros(case)
    rng = np.random.default_rng(0)
    tangent = {k: jnp.asarray(rng.standard_normal(v.shape)) for k, v in d0.items()}
    out, jvp = jax.jvp(f, (d0,), (tangent,))
    cotangent = jnp.asarray(rng.standard_normal(out.shape))
    _, pullback = jax.vjp(f, d0)
    (grad,) = pullback(cotangent)
    lhs = sum(float(jnp.vdot(grad[k], tangent[k])) for k in grad)
    rhs = float(jnp.vdot(cotangent, jvp))
    assert abs(lhs - rhs) <= TRANSPOSE_RTOL * abs(rhs)
