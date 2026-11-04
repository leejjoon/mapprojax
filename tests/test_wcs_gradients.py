"""Gradient smoke test: `Tan(precision="f64")` + SIP pixel->sky, differentiated with respect
to every WCS parameter (crpix, cd, crval, SIP a/b) and checked against finite
differences in forward and reverse mode. Per-mode checks are in test_gradients_modes.py.

Each parameter is perturbed as ``base + scale * delta`` and the output is the
sky offset from CRVAL in arcsec, so every Jacobian block is O(1-10) arcsec per
unit ``delta`` and one atol/rtol pair fits all of them. SIP scales are
``1 px / 1000**(p+q)`` for the allowed terms 2 <= p+q <= 3 (pixel offsets here
reach ~1000), and zero elsewhere.

Step-sweep evidence (fp64, CPU): the max |finite difference - JVP| over all
parameter groups was <=1.1e-7 at eps=1e-3, <=1.1e-6 at 1e-4 and <=7.5e-6 at
1e-5. The error grows as 1/eps, so it is round-off limited (~2e-10 arcsec of
f64 noise on ~5 rad intermediates) and truncation error is negligible. The
budget below passes at every step in the sweep.

One pixel sits on the reference column (u = 0) to cover the ``u ** 0`` term
of ``Sip._evaluate_poly`` at zero.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from mapprojax.wcs import Tan
from mapprojax.sip import Sip


@pytest.fixture(autouse=True)
def _x64_enabled():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


ATOL, RTOL = 1e-6, 1e-5


def _problem():
    crpix = jnp.array([1019.5, 1019.5])
    cd = jnp.array([[-1.6e-3, 1.0e-5], [2.0e-5, 1.66e-3]])
    crval = jnp.array([306.77, 39.38])
    a = (jnp.zeros((4, 4)).at[2, 0].set(2e-6).at[0, 2].set(-1.5e-6).at[1, 1].set(1e-6)
         .at[3, 0].set(3e-10).at[1, 2].set(-2e-10))
    b = (jnp.zeros((4, 4)).at[0, 2].set(1.8e-6).at[2, 0].set(-1e-6).at[1, 1].set(-1.2e-6)
         .at[0, 3].set(2.5e-10).at[2, 1].set(1e-10))
    pix = jnp.array([[300.0, 1700.0], [1500.0, 400.0], [900.0, 1100.0], [1800.0, 1900.0],
                     [1019.5, 1300.0]])

    p, q = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    sip_scale = jnp.asarray(np.where((p + q >= 2) & (p + q <= 3), 1.0 / 1000.0 ** (p + q), 0.0))
    scale = dict(crpix=1.0, cd=1e-3, crval=1e-3, a=sip_scale, b=sip_scale)

    def sky_offset_arcsec(d):
        wcs = Tan(
            crpix + scale["crpix"] * d["crpix"],
            cd * (1.0 + scale["cd"] * d["cd"]),
            crval + scale["crval"] * d["crval"], precision="f64",
            sip=Sip(a=a + scale["a"] * d["a"], b=b + scale["b"] * d["b"]),
        )
        ra, dec = wcs.unproj(pix[:, 0], pix[:, 1])
        offset = jnp.stack([ra - jnp.radians(crval[0]), dec - jnp.radians(crval[1])])
        return jnp.degrees(offset) * 3600.0

    delta0 = dict(crpix=jnp.zeros(2), cd=jnp.zeros((2, 2)), crval=jnp.zeros(2),
                  a=jnp.zeros((4, 4)), b=jnp.zeros((4, 4)))
    return sky_offset_arcsec, delta0


def test_jacobian_is_finite_and_well_scaled():
    f, d0 = _problem()
    jac = jax.jacrev(f)(d0)
    for name, block in jac.items():
        assert np.all(np.isfinite(block)), name
        assert 0.1 < float(jnp.abs(block).max()) < 100.0, name


@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5])
def test_check_grads_all_parameters(eps):
    f, d0 = _problem()
    check_grads(f, (d0,), order=1, modes=("fwd", "rev"), eps=eps, atol=ATOL, rtol=RTOL)
