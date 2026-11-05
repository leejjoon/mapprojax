"""Dimensionless finite-difference sweeps for Reprojector gradient tests."""
from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp

from mapprojax.reproject import Reprojector
from mapprojax.sip import Sip
from mapprojax.wcs import Tan

jax.config.update("jax_enable_x64", True)


CRPIX_SRC = jnp.asarray([30.0, 30.0], dtype=jnp.float64)
CRPIX_TGT = jnp.asarray([31.0, 28.0], dtype=jnp.float64)
CD_SRC = jnp.asarray([[-1.0e-2, 2.0e-4], [1.0e-4, 9.8e-3]], dtype=jnp.float64)
CD_TGT = jnp.asarray([[-1.02e-2, 1.0e-4], [-2.0e-4, 9.9e-3]], dtype=jnp.float64)
CRVAL_SRC = jnp.asarray([180.0, 30.0], dtype=jnp.float64)
CRVAL_TGT = jnp.asarray([180.12, 30.08], dtype=jnp.float64)
PIXELS = jnp.asarray([[42.0, 39.0], [47.0, 34.0]], dtype=jnp.float64)

SIP_SCALE = jnp.zeros((4, 4), dtype=jnp.float64)
for _p in range(4):
    for _q in range(4):
        if 2 <= _p + _q <= 3:
            SIP_SCALE = SIP_SCALE.at[_p, _q].set(
                2e-4 if _p + _q == 2 else 2e-6
            )

A = jnp.zeros((4, 4), dtype=jnp.float64)
A = A.at[2, 0].set(8e-5).at[1, 1].set(-4e-5).at[0, 2].set(3e-5)
A = A.at[3, 0].set(1e-6).at[1, 2].set(-7e-7)
B = jnp.zeros((4, 4), dtype=jnp.float64)
B = B.at[0, 2].set(-6e-5).at[2, 0].set(4e-5).at[1, 1].set(2e-5)
B = B.at[0, 3].set(8e-7).at[2, 1].set(-6e-7)


def _sip(a, b, ap=None, bp=None):
    return Sip(a=a, b=b, ap=ap, bp=bp)


def _loss(family: str, delta, precision: str):
    dtype = jnp.float32 if precision == "f32" else jnp.float64
    src_crpix = CRPIX_SRC
    tgt_crpix = CRPIX_TGT
    src_cd = CD_SRC
    tgt_cd = CD_TGT
    src_a = A
    src_b = B
    tgt_a = A
    tgt_b = B
    tgt_ap = -0.8 * A
    tgt_bp = -0.8 * B
    m_delta = jnp.zeros((3, 3), dtype=jnp.float64)
    src_crval = CRVAL_SRC
    tgt_crval = CRVAL_TGT

    if family == "pixels":
        pixels = PIXELS.at[0, 0].add(0.5 * delta)
    else:
        pixels = PIXELS
    if family == "src_crpix":
        src_crpix = src_crpix.at[0].add(0.5 * delta)
    if family == "tgt_crpix":
        tgt_crpix = tgt_crpix.at[0].add(0.5 * delta)
    if family == "src_cd":
        src_cd = src_cd.at[0, 0].multiply(1.0 + 1e-2 * delta)
    if family == "tgt_cd":
        tgt_cd = tgt_cd.at[0, 0].multiply(1.0 + 1e-2 * delta)
    if family == "src_a":
        src_a = src_a + SIP_SCALE * delta
    if family == "src_b":
        src_b = src_b + SIP_SCALE * delta
    if family == "tgt_a":
        tgt_a = tgt_a + SIP_SCALE * delta
    if family == "tgt_b":
        tgt_b = tgt_b + SIP_SCALE * delta
    if family == "tgt_ap":
        tgt_ap = tgt_ap + SIP_SCALE * delta
    if family == "tgt_bp":
        tgt_bp = tgt_bp + SIP_SCALE * delta
    if family == "m_rel":
        m_delta = m_delta.at[0, 0].set(1e-4 * delta)
    if family == "src_crval":
        src_crval = src_crval.at[0].add(1e-2 * delta)
    if family == "tgt_crval":
        tgt_crval = tgt_crval.at[0].add(1e-2 * delta)

    src = Tan(
        src_crpix, src_cd, src_crval,
        sip=_sip(src_a, src_b),
        precision="f64",
    )
    tgt = Tan(
        tgt_crpix, tgt_cd, tgt_crval,
        sip=_sip(tgt_a, tgt_b, tgt_ap, tgt_bp),
        precision="f64",
    )
    m_rel = tgt.r_matrix @ src.r_matrix.T + m_delta
    rp = Reprojector.from_matrix(src, tgt, m_rel, matrix_precision=precision)
    x, y = rp.pix_to_pix(pixels[:, 0], pixels[:, 1])
    return jnp.sum(x + 0.3 * y).astype(dtype)


FAMILIES = (
    "pixels", "src_crpix", "tgt_crpix", "src_cd", "tgt_cd",
    "src_a", "src_b", "tgt_a", "tgt_b", "tgt_ap", "tgt_bp",
    "m_rel", "src_crval", "tgt_crval",
)
SELECTED = {
    "f32": {"eps": 1.0, "atol": 1e-2, "rtol": 1e-2},
    "f64": {"eps": 1e-3, "atol": 1e-6, "rtol": 1e-5},
}
SWEEP_EPS = {
    "f32": (0.1, 0.3, 1.0, 3.0),
    "f64": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
}


def sweep_family(family: str, precision: str) -> dict:
    dtype = jnp.float32 if precision == "f32" else jnp.float64
    zero = jnp.asarray(0.0, dtype=dtype)
    gradient = jax.grad(lambda value: _loss(family, value, precision))(zero)
    records = []
    for eps in SWEEP_EPS[precision]:
        step = jnp.asarray(eps, dtype=dtype)
        forward = _loss(family, step, precision)
        reverse = _loss(family, -step, precision)
        finite_difference = (forward - reverse) / (2 * step)
        error = jnp.abs(gradient - finite_difference)
        records.append({
            "eps": eps,
            "finite_difference": float(finite_difference),
            "gradient": float(gradient),
            "absolute_error": float(error),
        })
    selected = SELECTED[precision]
    selected_record = min(records, key=lambda item: abs(item["eps"] - selected["eps"]))
    plateau = [
        item["absolute_error"] for item in records
        if selected["eps"] / 3 <= item["eps"] <= selected["eps"] * 3
    ]
    return {
        "selected": selected,
        "selected_error": selected_record["absolute_error"],
        "plateau_min_error": min(plateau),
        "plateau_max_error": max(plateau),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("f32", "f64"), action="append")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    precisions = args.precision or ("f32", "f64")
    result = {
        "backend": jax.default_backend(),
        "devices": [repr(device) for device in jax.devices()],
        "families": {
            precision: {
                family: sweep_family(family, precision)
                for family in FAMILIES
            }
            for precision in precisions
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
