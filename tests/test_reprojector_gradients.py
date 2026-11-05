"""Scale-aware derivative and masking checks for ``Reprojector``.

The f64 finite-difference reference uses eps=1e-3, atol=1e-6, rtol=1e-5.
For f32, parameters are scaled before entering the kernel and eps=1 is used;
the recorded atol=rtol=1e-2 accommodates the measured float32 output-rounding
floor while keeping each active parameter block O(1) per unit delta.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from mapprojax.reproject import Reprojector
from mapprojax.sip import Sip
from mapprojax.wcs import Sin, Tan


@pytest.fixture(autouse=True)
def _x64_enabled():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


CRPIX_SRC = np.asarray([30.0, 30.0], dtype=np.float64)
CRPIX_TGT = np.asarray([31.0, 28.0], dtype=np.float64)
CD_SRC = np.asarray([[-1.0e-2, 2.0e-4], [1.0e-4, 9.8e-3]], dtype=np.float64)
CD_TGT = np.asarray([[-1.02e-2, 1.0e-4], [-2.0e-4, 9.9e-3]], dtype=np.float64)
CRVAL_SRC = np.asarray([180.0, 30.0], dtype=np.float64)
CRVAL_TGT = np.asarray([180.12, 30.08], dtype=np.float64)
PIXELS = np.asarray([[42.0, 39.0], [47.0, 34.0]], dtype=np.float64)

_SIP_SCALE = np.zeros((4, 4), dtype=np.float64)
for _p in range(4):
    for _q in range(4):
        if 2 <= _p + _q <= 3:
            _SIP_SCALE[_p, _q] = 2e-4 if _p + _q == 2 else 2e-6

_A = np.zeros((4, 4), dtype=np.float64)
_A[2, 0], _A[1, 1], _A[0, 2] = 8e-5, -4e-5, 3e-5
_A[3, 0], _A[1, 2] = 1e-6, -7e-7
_B = np.zeros((4, 4), dtype=np.float64)
_B[0, 2], _B[2, 0], _B[1, 1] = -6e-5, 4e-5, 2e-5
_B[0, 3], _B[2, 1] = 8e-7, -6e-7

_CHECK = {
    "f32": (1.0, 1e-2, 1e-2),
    "f64": (1e-3, 1e-6, 1e-5),
}


def _delta_tree(precision, sip_case):
    dtype = jnp.float32 if precision == "f32" else jnp.float64
    delta = {
        "pixels": jnp.zeros((2, 2), dtype=dtype),
        "src_crpix": jnp.zeros((2,), dtype=dtype),
        "src_cd": jnp.zeros((2, 2), dtype=dtype),
        "tgt_crpix": jnp.zeros((2,), dtype=dtype),
        "tgt_cd": jnp.zeros((2, 2), dtype=dtype),
        "m_rel": jnp.zeros((3, 3), dtype=dtype),
        "src_a": jnp.zeros((4, 4), dtype=dtype),
        "src_b": jnp.zeros((4, 4), dtype=dtype),
        "tgt_ap": jnp.zeros((4, 4), dtype=dtype),
        "tgt_bp": jnp.zeros((4, 4), dtype=dtype),
    }
    if sip_case == "ab_seeded":
        delta["tgt_a"] = jnp.zeros((4, 4), dtype=dtype)
        delta["tgt_b"] = jnp.zeros((4, 4), dtype=dtype)
    return delta


def _sip_for_delta(delta, side, sip_case):
    dtype = jnp.float64
    if side == "tgt" and sip_case == "ap_bp_only":
        ap = -0.8 * jnp.asarray(_A, dtype=dtype) + jnp.asarray(_SIP_SCALE) * delta["tgt_ap"]
        bp = -0.8 * jnp.asarray(_B, dtype=dtype) + jnp.asarray(_SIP_SCALE) * delta["tgt_bp"]
        return Sip(ap=ap, bp=bp)
    a = jnp.asarray(_A, dtype=dtype) + jnp.asarray(_SIP_SCALE) * delta[f"{side}_a"]
    b = jnp.asarray(_B, dtype=dtype) + jnp.asarray(_SIP_SCALE) * delta[f"{side}_b"]
    if side == "src":
        return Sip(a=a, b=b)
    ap = -0.8 * jnp.asarray(_A, dtype=dtype) + jnp.asarray(_SIP_SCALE) * delta["tgt_ap"]
    bp = -0.8 * jnp.asarray(_B, dtype=dtype) + jnp.asarray(_SIP_SCALE) * delta["tgt_bp"]
    return Sip(a=a, b=b, ap=ap, bp=bp)


def _scaled_reprojection(delta, precision, sip_case):
    src = Tan(
        CRPIX_SRC + 0.5 * delta["src_crpix"],
        CD_SRC * (1.0 + 1e-2 * delta["src_cd"]),
        CRVAL_SRC,
        sip=_sip_for_delta(delta, "src", sip_case),
        precision="f64",
    )
    if sip_case == "ab_seeded":
        tgt_a = jnp.asarray(_A, dtype=jnp.float64) + jnp.asarray(_SIP_SCALE) * delta["tgt_a"]
        tgt_b = jnp.asarray(_B, dtype=jnp.float64) + jnp.asarray(_SIP_SCALE) * delta["tgt_b"]
        tgt_sip = Sip(
            a=tgt_a,
            b=tgt_b,
            ap=-0.8 * jnp.asarray(_A) + jnp.asarray(_SIP_SCALE) * delta["tgt_ap"],
            bp=-0.8 * jnp.asarray(_B) + jnp.asarray(_SIP_SCALE) * delta["tgt_bp"],
        )
    else:
        tgt_sip = _sip_for_delta(delta, "tgt", sip_case)
    tgt = Tan(
        CRPIX_TGT + 0.5 * delta["tgt_crpix"],
        CD_TGT * (1.0 + 1e-2 * delta["tgt_cd"]),
        CRVAL_TGT,
        sip=tgt_sip,
        precision="f64",
    )

    # The matrix is independent of these non-crval leaves. Its separate
    # scaled perturbation checks the live matrix leaf itself.
    m_rel = tgt.r_matrix @ src.r_matrix.T + 1e-4 * delta["m_rel"]
    rp = Reprojector.from_matrix(src, tgt, m_rel, matrix_precision=precision)
    pixels = PIXELS + 0.5 * delta["pixels"]
    x, y = rp.pix_to_pix(pixels[:, 0], pixels[:, 1])
    return jnp.stack((x, y))


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("sip_case", ["ab_seeded", "ap_bp_only"])
def test_reprojector_check_grads_all_applicable_leaves(precision, sip_case):
    """Check pixels, both legs' geometry/SIP leaves, and m_rel in both modes.

    The source uses A/B in the forward leg. The target uses A/B plus an
    AP/BP seed, or AP/BP as the inverse itself; source AP/BP is not used by
    this forward reprojection path.
    """
    delta = _delta_tree(precision, sip_case)
    eps, atol, rtol = _CHECK[precision]
    f = lambda d: _scaled_reprojection(d, precision, sip_case)
    assert np.isfinite(np.asarray(f(delta))).all()
    check_grads(f, (delta,), order=1, modes=("fwd", "rev"),
                eps=eps, atol=atol, rtol=rtol)


def _live_crval_reprojection(delta, precision):
    crval_src = CRVAL_SRC + 1e-2 * delta[0]
    crval_tgt = CRVAL_TGT + 1e-2 * delta[1]
    src = Tan(CRPIX_SRC, CD_SRC, crval_src, precision="f64")
    tgt = Tan(CRPIX_TGT, CD_TGT, crval_tgt, precision="f64")
    # Keep construction inside the differentiated function so both reference
    # vectors feed the dynamic matrix passed to from_matrix.
    m_rel = tgt.r_matrix @ src.r_matrix.T
    rp = Reprojector.from_matrix(src, tgt, m_rel, matrix_precision=precision)
    x, y = rp.pix_to_pix(PIXELS[:, 0], PIXELS[:, 1])
    return jnp.stack((x, y))


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_from_matrix_live_crval_dependencies_match_finite_differences(precision):
    dtype = jnp.float32 if precision == "f32" else jnp.float64
    delta = jnp.zeros((2, 2), dtype=dtype)
    eps, atol, rtol = _CHECK[precision]
    f = lambda d: _live_crval_reprojection(d, precision)
    assert np.isfinite(np.asarray(f(delta))).all()
    check_grads(f, (delta,), order=1, modes=("fwd", "rev"),
                eps=eps, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "src_projection,tgt_projection,src_crval,tgt_crval",
    [
        (Tan, Tan, (180.0, 30.0), (180.12, 30.08)),
        (Tan, Tan, (359.9, 0.0), (0.1, 0.1)),
        (Tan, Tan, (45.0, 89.0), (45.1, 89.0)),
        (Tan, Tan, (45.0, -89.0), (45.1, -89.0)),
        (Tan, Sin, (180.0, 30.0), (180.2, 30.1)),
    ],
)
def test_live_crval_gradients_cover_representative_geometry(
    src_projection, tgt_projection, src_crval, tgt_crval
):
    """Live-reference derivatives remain numerically grounded across geometry."""
    crvals = jnp.asarray([src_crval, tgt_crval], dtype=jnp.float64)
    pixels = jnp.asarray([10.0, 10.0], dtype=jnp.float64)

    def loss(values):
        src = src_projection([10.0, 10.0], CD_SRC, values[0], precision="f64")
        tgt = tgt_projection([10.0, 10.0], CD_TGT, values[1], precision="f64")
        matrix = tgt.r_matrix @ src.r_matrix.T
        rp = Reprojector.from_matrix(src, tgt, matrix, matrix_precision="f64")
        x, y = rp.pix_to_pix(*pixels)
        return jnp.where(jnp.isfinite(x) & jnp.isfinite(y), x + 0.3 * y, 0.0)

    value = loss(crvals)
    gradient = jax.grad(loss)(crvals)
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradient)).all()

    eps = 1e-4
    numerical = np.zeros((2, 2), dtype=np.float64)
    for side in range(2):
        for component in range(2):
            delta = np.zeros((2, 2), dtype=np.float64)
            delta[side, component] = eps
            numerical[side, component] = (
                float(loss(crvals + delta)) - float(loss(crvals - delta))
            ) / (2 * eps)
    np.testing.assert_allclose(
        np.asarray(gradient), numerical, rtol=2e-4, atol=2e-5
    )


def test_eager_and_from_matrix_reference_dependency_pins():
    src = Tan(CRPIX_SRC, CD_SRC, CRVAL_SRC, precision="f64")
    tgt = Tan(CRPIX_TGT, CD_TGT, CRVAL_TGT, precision="f64")
    eager = Reprojector(src, tgt, matrix_precision="f64")
    pixels = (PIXELS[:, 0], PIXELS[:, 1])

    def eager_loss(rp):
        x, y = rp.pix_to_pix(*pixels)
        return jnp.sum(x + 0.3 * y)

    eager_grad = jax.grad(eager_loss)(eager)
    np.testing.assert_array_equal(np.asarray(eager_grad.src_leg.crval), np.zeros(2))
    np.testing.assert_array_equal(np.asarray(eager_grad.tgt_leg.crval), np.zeros(2))

    fixed_matrix = eager.m_rel

    def fixed_matrix_loss(crvals):
        dynamic_src = Tan(CRPIX_SRC, CD_SRC, crvals[0], precision="f64")
        dynamic_tgt = Tan(CRPIX_TGT, CD_TGT, crvals[1], precision="f64")
        rp = Reprojector.from_matrix(
            dynamic_src, dynamic_tgt, fixed_matrix, matrix_precision="f64"
        )
        x, y = rp.pix_to_pix(*pixels)
        return jnp.sum(x + 0.3 * y)

    crvals = jnp.stack((CRVAL_SRC, CRVAL_TGT))
    fixed_grad = jax.grad(fixed_matrix_loss)(crvals)
    np.testing.assert_array_equal(np.asarray(fixed_grad), np.zeros((2, 2)))

    def eager_with_traced_reference(crvals):
        dynamic_src = Tan(CRPIX_SRC, CD_SRC, crvals[0], precision="f64")
        dynamic_tgt = Tan(CRPIX_TGT, CD_TGT, crvals[1], precision="f64")
        return Reprojector(dynamic_src, dynamic_tgt, matrix_precision="f64").m_rel

    with pytest.raises(TypeError, match="traced"):
        jax.jit(eager_with_traced_reference)(crvals)


def _simple_wcs(crval, *, precision="f64", crpix=(0.0, 0.0), cd=None):
    if cd is None:
        cd = [[-1e-2, 2e-4], [1e-4, 9.8e-3]]
    return Tan(crpix, cd, crval, precision=precision)


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_pix_to_pix_masked_loss_isolates_valid_neighbour_gradient(precision):
    src0 = _simple_wcs((0.0, 0.0))
    tgt = _simple_wcs((60.0, 0.0))
    matrix = Reprojector(src0, tgt, matrix_precision="f64").m_rel
    cd0 = CD_SRC

    def masked_loss(cd, x):
        src = _simple_wcs((0.0, 0.0), cd=cd)
        rp = Reprojector.from_matrix(src, tgt, matrix, matrix_precision=precision)
        out_x, out_y = rp.pix_to_pix(x, jnp.zeros_like(x))
        valid = jnp.isfinite(out_x) & jnp.isfinite(out_y)
        return jnp.sum(jnp.where(valid, out_x + 0.3 * out_y, 0.0))

    mixed_x = jnp.asarray([5.0, 20000.0], dtype=jnp.float32 if precision == "f32" else jnp.float64)
    single_x = mixed_x[:1]
    mixed_grad = jax.grad(lambda cd: masked_loss(cd, mixed_x))(cd0)
    single_grad = jax.grad(lambda cd: masked_loss(cd, single_x))(cd0)
    assert np.isfinite(np.asarray(mixed_grad)).all()
    assert float(jnp.abs(single_grad).max()) > 1e-4
    np.testing.assert_allclose(np.asarray(mixed_grad), np.asarray(single_grad),
                               rtol=2e-4, atol=2e-4)

    all_invalid = jnp.asarray([20000.0, 20001.0], dtype=mixed_x.dtype)
    all_invalid_grad = jax.grad(lambda cd: masked_loss(cd, all_invalid))(cd0)
    assert np.isfinite(np.asarray(all_invalid_grad)).all()
    np.testing.assert_array_equal(np.asarray(all_invalid_grad), np.zeros((2, 2)))

    invalid_x, invalid_y = Reprojector.from_matrix(
        src0, tgt, matrix, matrix_precision=precision
    ).pix_to_pix(mixed_x[1:], jnp.zeros((1,), dtype=mixed_x.dtype))
    assert np.isnan(np.asarray(invalid_x)).all() and np.isnan(np.asarray(invalid_y)).all()


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_pixel_map_bounds_mask_loss_isolates_gradients_and_all_invalid_is_finite(precision):
    wcs = _simple_wcs((12.0, 34.0))
    rp = Reprojector(wcs, wcs, matrix_precision=precision)
    dtype = jnp.float32 if precision == "f32" else jnp.float64

    def bounds_loss(points):
        x, y, valid = rp.pixel_map(
            points[:, 0], points[:, 1], bounds_shape=(10, 10)
        )
        return jnp.sum(jnp.where(valid, x + 0.3 * y, 0.0))

    mixed = jnp.asarray([[2.0, 3.0], [100.0, 3.0]], dtype=dtype)
    single = mixed[:1]
    mixed_grad = jax.grad(bounds_loss)(mixed)
    single_grad = jax.grad(bounds_loss)(single)
    assert np.isfinite(np.asarray(mixed_grad)).all()
    np.testing.assert_allclose(np.asarray(mixed_grad[0]), np.asarray(single_grad[0]),
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(mixed_grad[1]), np.zeros(2))

    all_invalid = jnp.asarray([[100.0, 3.0], [200.0, 4.0]], dtype=dtype)
    all_invalid_grad = jax.grad(bounds_loss)(all_invalid)
    assert np.isfinite(np.asarray(all_invalid_grad)).all()
    np.testing.assert_array_equal(np.asarray(all_invalid_grad), np.zeros((2, 2)))
    x, y, valid = rp.pixel_map(all_invalid[:, 0], all_invalid[:, 1], bounds_shape=(10, 10))
    assert not np.asarray(valid).any()
    np.testing.assert_array_equal(np.asarray(x), np.zeros(2, dtype=np.asarray(x).dtype))
    np.testing.assert_array_equal(np.asarray(y), np.zeros(2, dtype=np.asarray(y).dtype))


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_singular_cd_is_invalid_with_finite_masked_loss_gradient(precision):
    src = _simple_wcs((12.0, 34.0))
    eye = jnp.eye(3, dtype=jnp.float64 if precision == "f64" else jnp.float32)
    singular_cd = jnp.asarray([[1e-2, 2e-2], [2e-2, 4e-2]], dtype=jnp.float64)

    def evaluate(cd):
        tgt = _simple_wcs((12.0, 34.0), cd=cd)
        rp = Reprojector.from_matrix(src, tgt, eye, matrix_precision=precision)
        return rp.pixel_map(jnp.asarray([2.0]), jnp.asarray([3.0]))

    def masked_loss(cd):
        x, y, valid = evaluate(cd)
        return jnp.sum(jnp.where(valid, x + y, 0.0))

    x, y, valid = jax.jit(evaluate)(singular_cd)
    assert not np.asarray(valid).any()
    np.testing.assert_array_equal(np.asarray(x), np.zeros(1, dtype=np.asarray(x).dtype))
    np.testing.assert_array_equal(np.asarray(y), np.zeros(1, dtype=np.asarray(y).dtype))

    grad = jax.grad(masked_loss)(singular_cd)
    assert np.isfinite(np.asarray(grad)).all()
    np.testing.assert_array_equal(np.asarray(grad), np.zeros((2, 2)))


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_ap_bp_seed_sensitivity_decreases_with_newton_convergence(precision):
    """AP/BP are only the target inverse's seed when A/B are present.

    Their sensitivity depends on the finite Newton step count; it is expected
    to shrink as the solve converges, not to be exactly zero at a fixed count.
    """
    dtype = jnp.float32 if precision == "f32" else jnp.float64
    seed_scale = 2e-4

    def seed_loss(seed_delta):
        a = jnp.asarray(_A, dtype=dtype)
        b = jnp.asarray(_B, dtype=dtype)
        ap = -0.8 * a.at[2, 0].add(seed_scale * seed_delta)
        bp = -0.8 * b
        u, v = jnp.asarray(16.0, dtype=dtype), jnp.asarray(11.0, dtype=dtype)
        seed_sip = Sip(a=a, b=b, ap=ap, bp=bp)
        return (u + seed_sip._evaluate_poly(u, v, ap, jnp)
                + 0.3 * (v + seed_sip._evaluate_poly(u, v, bp, jnp)))

    seed_sensitivity = float(jnp.abs(jax.grad(seed_loss)(jnp.asarray(0.0, dtype=dtype))))

    def sensitivity(steps):
        def loss(seed_delta):
            a = jnp.asarray(_A, dtype=dtype)
            b = jnp.asarray(_B, dtype=dtype)
            ap = -0.8 * a.at[2, 0].add(seed_scale * seed_delta)
            sip = Sip(a=a, b=b, ap=ap, bp=-0.8 * b, inverse_steps=steps)
            # Nonconverged solves return the input stand-in. The two-step
            # case below is separately required to converge before comparing
            # its derivative with the raw seed's sensitivity.
            u, v, _ = sip._foc_to_pix_valid(
                jnp.asarray(16.0, dtype=dtype),
                jnp.asarray(11.0, dtype=dtype),
                xp=jnp,
            )
            return u + 0.3 * v

        return float(jnp.abs(jax.grad(loss)(jnp.asarray(0.0, dtype=dtype))))

    sensitivities = [sensitivity(steps) for steps in (1, 2)]
    assert np.isfinite(sensitivities).all(), sensitivities
    a, b = jnp.asarray(_A, dtype=dtype), jnp.asarray(_B, dtype=dtype)
    converged_sip = Sip(a=a, b=b, ap=-0.8 * a, bp=-0.8 * b, inverse_steps=2)
    _, _, converged = converged_sip._foc_to_pix_valid(
        jnp.asarray(16.0, dtype=dtype), jnp.asarray(11.0, dtype=dtype), xp=jnp
    )
    assert bool(converged)
    assert seed_sensitivity > 0.0
    assert max(sensitivities) < seed_sensitivity, (seed_sensitivity, sensitivities)
    if precision == "f32":
        assert sensitivities[1] < sensitivities[0], sensitivities
    else:
        assert sensitivities[1] <= sensitivities[0] + 1e-18, sensitivities
    assert sensitivities[1] < (1e-10 if precision == "f32" else 1e-18)


def test_f32_ap_bp_seed_counterexample_is_valid_and_source_seed_is_unused():
    dtype = jnp.float32
    a = jnp.zeros((3, 3), dtype=dtype).at[2, 0].set(0.01)
    zeros = jnp.zeros((3, 3), dtype=dtype)

    def target_inverse(seed, steps):
        sip = Sip(a=a, b=zeros, ap=zeros.at[2, 0].set(seed),
                  bp=zeros, inverse_steps=steps)
        u, _, valid = sip._foc_to_pix_valid(
            jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype), xp=jnp
        )
        return u, valid

    one_step = lambda seed: target_inverse(seed, 1)[0]
    two_step = lambda seed: target_inverse(seed, 2)[0]
    one_grad = float(jnp.abs(jax.grad(one_step)(jnp.asarray(0.0, dtype=dtype))))
    two_grad = float(jnp.abs(jax.grad(two_step)(jnp.asarray(0.0, dtype=dtype))))
    assert bool(target_inverse(jnp.asarray(0.0, dtype=dtype), 1)[1])
    assert one_grad > 1e-4
    assert 0.0 < two_grad < one_grad

    def source_loss(seed):
        source_sip = Sip(a=a, b=zeros, ap=zeros.at[2, 0].set(seed), bp=zeros)
        source = Tan(CRPIX_SRC, CD_SRC, CRVAL_SRC, sip=source_sip, precision="f32")
        target = Tan(CRPIX_TGT, CD_TGT, CRVAL_TGT, precision="f32")
        rp = Reprojector(source, target, matrix_precision="f32")
        x, y = rp.pix_to_pix(
            jnp.asarray([42.0], dtype=dtype), jnp.asarray([39.0], dtype=dtype)
        )
        return x[0] + 0.3 * y[0]

    source_grad = jax.grad(source_loss)(jnp.asarray(0.0, dtype=dtype))
    np.testing.assert_array_equal(np.asarray(source_grad), np.asarray(0.0, dtype=dtype))
