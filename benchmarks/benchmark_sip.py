"""SIP polynomial evaluation: broadcast vs Horner.

`Sip._evaluate_poly` used to build `u[..., None] ** p_range` and reduce with
`einsum`, which materialises order^2 full-frame temporaries. The Horner form
is O(order) multiplies per point.

Run with `python benchmarks/benchmark_sip.py`. JAX runs on CPU unless
`JAX_PLATFORMS` says otherwise. Steady-state timings are warmed and
synchronized under `jit`. The "first call" column is compilation plus the
first execution, a one-off cost per shape.

The Horner row times the production `Sip._evaluate_poly`; the broadcast row
is the pre-Horner form, kept here only as the baseline. The SIP inverse is
timed separately, at the step counts that converge.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import accuracy_reference as ar  # noqa: E402

from mapprojax.sip import Sip  # noqa: E402
from mapprojax.wcs_utils import load_wcs_params_from_header  # noqa: E402

REPEATS = 5
CASES = (("SPHEREx L2 (order 3, 2040x2040)", "spherex_l2.hdr", 2040),
         ("order 4, 1201x1201", None, 1201))


def _broadcast(c, xp):
    """The pre-Horner form, kept here as the comparison baseline."""
    def evaluate(u, v):
        m, n = c.shape
        u_pow = u[..., None] ** xp.arange(m, dtype=xp.int32)
        v_pow = v[..., None] ** xp.arange(n, dtype=xp.int32)
        return xp.einsum("pq,...p,...q->...", c, u_pow, v_pow)
    return evaluate


def _horner(c, xp):
    """The production evaluator, so the benchmark tracks the real code."""
    sip = Sip()

    def evaluate(u, v):
        return sip._evaluate_poly(u, v, c, xp)
    return evaluate


def _sips():
    _, _, _, l2 = load_wcs_params_from_header(ar.load_header("spherex_l2.hdr"))
    rng = np.random.default_rng(0)
    order4 = Sip(a=rng.uniform(-1e-9, 1e-9, (5, 5)),
                 b=rng.uniform(-1e-9, 1e-9, (5, 5)))
    return {CASES[0][0]: l2, CASES[1][0]: order4}


def _time(fn, args):
    jitted = jax.jit(fn)
    jax.block_until_ready(jitted(*args))
    times = []
    for _ in range(REPEATS):
        import time
        t = time.perf_counter()
        jax.block_until_ready(jitted(*args))
        times.append(time.perf_counter() - t)
    return min(times)


def main():
    sips = _sips()
    print(f"JAX {jax.__version__}, backend {jax.default_backend()}, fp64 on\n")
    print(f"{'case':32} {'form':10} {'ms':>8}  {'vs Horner':>10}")
    for label, _, n in CASES:
        sip = sips[label]
        coeff = jnp.asarray(sip.a)
        axis = jnp.linspace(-n / 2, n / 2, n)
        u, v = jnp.meshgrid(axis, axis)

        results = {}
        for form, factory in (("broadcast", _broadcast), ("horner", _horner)):
            fn = factory(coeff, jnp)
            first_start = __import__("time").perf_counter()
            jax.block_until_ready(jax.jit(fn)(u, v))
            first_ms = (__import__("time").perf_counter() - first_start) * 1e3
            ms = _time(fn, (u, v)) * 1e3
            results[form] = ms
            print(f"{label if form == 'broadcast' else '':32} {form:10} {ms:8.2f}  "
                  f"(first call {first_ms:5.0f} ms)")
        print(f"{'':32} {'horner is':10} {results['broadcast'] / results['horner']:8.1f}x\n")

    # The inverse solve, on top of the same evaluator. Before the Newton
    # solve this cost the same and was wrong by 2.9e-2 px on the L2 detector.
    sip = sips[CASES[0][0]]
    n = CASES[0][2]
    axis = jnp.linspace(-n / 2, n / 2, n)
    u, v = jnp.meshgrid(axis, axis)
    from mapprojax.sip import Sip

    def solve(steps):
        def run(u, v):
            s = Sip(a=jnp.asarray(sip.a), b=jnp.asarray(sip.b),
                    ap=jnp.asarray(sip.ap), bp=jnp.asarray(sip.bp),
                    inverse_steps=steps)
            return s.foc_to_pix(u, v)[0]
        return run

    # One step does not converge on this detector (most points come back NaN
    # by design), so it is not a usable setting and is not timed. Two is the
    # fewest that converges here; three is the default.
    print("\nSIP inverse (Newton in focal coordinates), 2040x2040, order 3:")
    for steps in (2, 3, 5):
        ms = _time(solve(steps), (u, v)) * 1e3
        x = solve(steps)(u, v)
        n_nan = int(jnp.isnan(x).sum())
        note = " (default)" if steps == 3 else ""
        print(f"  inverse_steps={steps}: {ms:7.2f} ms, {n_nan} unconverged{note}")
    print()

    # Agreement between the two forms on the real coefficients.
    sip = sips[CASES[0][0]]
    rng = np.random.default_rng(1)
    u, v = rng.uniform(-1018, 1018, 20000), rng.uniform(-1018, 1018, 20000)
    for name, c in (("A", sip.a), ("B", sip.b), ("AP", sip.ap), ("BP", sip.bp)):
        ref = _broadcast(np.asarray(c), np)(u, v)
        got = sip._evaluate_poly(u, v, c, np)
        print(f"  {name}: max |broadcast - horner| = {np.abs(got - ref).max():.2e} px")


if __name__ == "__main__":
    main()
