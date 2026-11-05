"""Precision policy tests for the explicit mapprojax.wcs API."""

import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(code, expect_ok=True):
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=REPO,
                          capture_output=True, text=True, timeout=300)
    if expect_ok and proc.returncode:
        raise AssertionError(f"expected success, got {proc.returncode}:\n{proc.stderr}")
    return proc


def test_import_is_quiet_with_x64_off():
    proc = run("""
        import warnings
        warnings.simplefilter('error')
        import mapprojax
        import jax
        assert not jax.config.jax_enable_x64
        print('ok')
    """)
    assert "ok" in proc.stdout


def test_f32_projection_and_reprojection_work_with_x64_off():
    proc = run("""
        import numpy as np
        from mapprojax import Reprojector
        from mapprojax.wcs import Tan
        src = Tan([20., 20.], [[-1e-3, 0.], [0., 1e-3]], [180., 30.], precision='f32')
        tgt = Tan([20., 20.], [[-1e-3, 0.], [0., 1e-3]], [180.01, 30.], precision='f32')
        rp = Reprojector(src, tgt, matrix_precision='f32')
        x, y = rp.pix_to_pix(np.array(20., dtype=np.float32), np.array(20., dtype=np.float32))
        assert np.isfinite(float(x)) and np.isfinite(float(y))
        print('ok')
    """)
    assert "ok" in proc.stdout


def test_f64_requests_fail_clearly_with_x64_off():
    proc = run("""
        from mapprojax.wcs import Tan
        Tan([0., 0.], [[-1e-3, 0.], [0., 1e-3]], [0., 0.], precision='f64')
    """, expect_ok=False)
    assert proc.returncode and "requires float64" in proc.stderr


_F64_WALKER = """
import jax, jax.numpy as jnp
import numpy as np


def f64_primitives(closed):
    \"\"\"Every primitive touching a float64 value, nested jaxprs included.\"\"\"
    found = set()

    # Operations only: an unused float64 leaf (the legs' host crval) is
    # data carried by the PyTree, not arithmetic in the trace.
    def walk(jx):
        for e in jx.eqns:
            for v in list(e.invars) + list(e.outvars):
                if getattr(getattr(v, "aval", None), "dtype", None) == jnp.float64:
                    found.add(str(e.primitive))
            for p in e.params.values():
                for sub in (p if isinstance(p, (list, tuple)) else [p]):
                    if hasattr(sub, "eqns"):
                        walk(sub)
                    elif hasattr(getattr(sub, "jaxpr", None), "eqns"):
                        walk(sub.jaxpr)

    walk(closed.jaxpr)
    return sorted(found)
"""


def test_f64_walker_detects_a_plain_f64_operation():
    """Mutation control: the check below must fail on real f64 arithmetic,
    including arithmetic hidden in a nested jit."""
    proc = run(_F64_WALKER + """
jax.config.update('jax_enable_x64', True)
inner = jax.jit(lambda a: a + a)
f = lambda x: x + inner(jnp.float64(1.0)).astype(jnp.float32)
assert f64_primitives(jax.make_jaxpr(f)(jnp.float32(1.0))), "walker missed f64"
assert not f64_primitives(jax.make_jaxpr(lambda x: x + x)(jnp.float32(1.0)))
print('ok')
""")
    assert "ok" in proc.stdout


def test_f32_reprojector_jaxpr_has_no_f64_ops_with_x64_on_or_off():
    """Isolate the f32 pixel_map kernel across construction and call styles.

    A live f64 relative matrix may be formed before the kernel; cast it before
    tracing so the check covers the per-pixel path rather than host setup.
    """
    for enabled in (False, True):
        proc = run(_F64_WALKER + f"""
jax.config.update('jax_enable_x64', {enabled})
from mapprojax import Reprojector
from mapprojax.sip import Sip
from mapprojax.wcs import Tan, Sin
from mapprojax.utils import compose_relative_rotation

a = np.zeros((4, 4)); a[2, 0] = 4e-6; a[1, 1] = -2e-6; a[0, 3] = 1e-9
b = np.zeros((4, 4)); b[2, 0] = -3e-6; b[1, 1] = 2.5e-6; b[3, 0] = 1e-9
sip_src = Sip(a=a, b=b, ap=-a, bp=-b, inverse_steps=2)
sip_tgt = Sip(a=0.5 * a, b=0.5 * b, ap=-0.5 * a, bp=-0.5 * b, inverse_steps=3)
cd = [[-1e-3, 0.], [0., 1e-3]]
src = Tan([100., 100.], cd, [180., 30.], sip=sip_src, precision='f32')
tgt = Sin([100., 100.], cd, [180.01, 30.], sip=sip_tgt, precision='f32')
rp = Reprojector(src, tgt, matrix_precision='f32')
x = jnp.full((8,), 3.0, jnp.float32); y = jnp.full((8,), 4.0, jnp.float32)

closed_over = jax.make_jaxpr(
    lambda x, y: rp.pixel_map(x, y, bounds_shape=(8, 8))
)(x, y)
as_argument = jax.make_jaxpr(
    lambda r, x, y: r.pixel_map(x, y, bounds_shape=(8, 8))
)(rp, x, y)
from_matrix = jax.make_jaxpr(
    lambda r, x, y: Reprojector.from_matrix(src, tgt, r.m_rel).pixel_map(
        x, y, bounds_shape=(8, 8)
    )
)(rp, x, y)
pix_to_pix_closed = jax.make_jaxpr(lambda x, y: rp.pix_to_pix(x, y))(x, y)
pix_to_pix_argument = jax.make_jaxpr(lambda r, x, y: r.pix_to_pix(x, y))(rp, x, y)
pix_to_pix_from_matrix = jax.make_jaxpr(
    lambda r, x, y: Reprojector.from_matrix(src, tgt, r.m_rel).pix_to_pix(x, y)
)(rp, x, y)

# Composition can use live host/f64 references before entering this f32
# kernel. Keep that work outside make_jaxpr and pass the already-cast leaf.
external_live_f64 = compose_relative_rotation(src, tgt)
assert external_live_f64.dtype == np.float64
external_f32 = jnp.asarray(external_live_f64.astype(np.float32))
external_matrix = jax.make_jaxpr(
    lambda x, y: Reprojector.from_matrix(src, tgt, external_f32).pixel_map(
        x, y, bounds_shape=(8, 8)
    )
)(x, y)

for name, jx in [("closed", closed_over), ("argument", as_argument),
                 ("from_matrix", from_matrix), ("external_matrix", external_matrix),
                 ("pix_to_pix_closed", pix_to_pix_closed),
                 ("pix_to_pix_argument", pix_to_pix_argument),
                 ("pix_to_pix_from_matrix", pix_to_pix_from_matrix)]:
    bad = f64_primitives(jx)
    assert not bad, (name, bad)
    assert any(e.primitive.name for e in jx.jaxpr.eqns), name
print('ok')
""")
        assert "ok" in proc.stdout
