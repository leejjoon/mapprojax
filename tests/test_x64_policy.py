"""The x64 policy, tested in fresh subprocesses.

`jax_enable_x64` is process-global and effectively latched once arrays have
been created, so each case runs in its own interpreter: the in-process
fixtures that other test modules use to toggle x64 cannot reproduce what a
user sees at import.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def run(code, expect_ok=True):
    """Run ``code`` in a fresh interpreter with x64 off (JAX's default)."""
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    if expect_ok and proc.returncode != 0:
        raise AssertionError(f"expected success, got {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    return proc


def test_import_is_quiet_with_x64_off():
    """No import-time warning: the old one fired on every user's first import."""
    proc = run("""
        import warnings
        warnings.simplefilter("error")
        import mapprojax            # must not warn
        import jax
        assert jax.config.jax_enable_x64 is False
        print("ok")
    """)
    assert "ok" in proc.stdout


def test_numpy_classes_work_with_x64_off():
    proc = run("""
        import numpy as np
        from mapprojax import Tan, Sin
        w = Tan([10., 10.], [[-1e-3, 0.], [0., 1e-3]], [180., 0.])
        ra, dec = w.unproj(np.array(12.0), np.array(9.0))
        x, y = w.proj(ra, dec)
        assert abs(float(x) - 12.0) < 1e-6, x
        d = w.unproj_offset(np.array(12.0), np.array(9.0))
        assert np.all(np.isfinite(d[0]))
        print("ok")
    """)
    assert "ok" in proc.stdout


def test_f32_reprojection_works_with_x64_off():
    proc = run("""
        import numpy as np
        from mapprojax import Tan, Reprojector
        src = Tan([200., 200.], [[-1e-3, 0.], [0., 1e-3]], [180., 30.])
        tgt = Tan([200., 200.], [[-1e-3, 0.], [0., 1e-3]], [180.01, 30.])
        rp = Reprojector(src, tgt, matrix_precision="f32")
        x, y = rp.pix_to_pix(np.array(200.0), np.array(200.0))
        assert np.isfinite(float(x))
        print(float(x))
    """)
    assert "ok" not in proc.stdout and proc.stdout.strip()


@pytest.mark.parametrize("code, needle", [
    ("""from mapprojax import Tan, Reprojector
w = Tan([0., 0.], [[-1e-3, 0.], [0., 1e-3]], [0., 0.])
Reprojector(w, w, matrix_precision="f64")""", "requires float64"),
    ("""from mapprojax import TanJaxMixed
TanJaxMixed([0., 0.], [[-1e-3, 0.], [0., 1e-3]], [0., 0.])""",
     "requires float64"),
    ("""from mapprojax import TanJaxArrayMixed
TanJaxArrayMixed([0., 0.], [[-1e-3, 0.], [0., 1e-3]], ([0.], [0.]))""",
     "requires float64"),
])
def test_explicit_f64_requests_fail_clearly(code, needle):
    proc = run(code + "\nprint('should not reach here')", expect_ok=False)
    assert proc.returncode != 0
    assert "ValueError" in proc.stderr and needle in proc.stderr, proc.stderr[-500:]
    assert "should not reach here" not in proc.stdout


def test_bare_jax_class_absolute_methods_raise_without_x64():
    proc = run("""
        import jax.numpy as jnp
        from mapprojax import TanJax
        w = TanJax(jnp.array([0., 0.]), jnp.array([[-1e-3, 0.], [0., 1e-3]]),
                   jnp.array([0., 0.]))
        try:
            w.unproj(jnp.array(1.0), jnp.array(1.0))
        except TypeError as exc:
            assert "float64" in str(exc)
            print("raised")
    """)
    assert "raised" in proc.stdout


def test_bare_jax_class_offset_path_still_works_with_x64_off():
    """The offset APIs are the supported f32 route; they must not raise."""
    proc = run("""
        import numpy as np
        import jax.numpy as jnp
        from mapprojax import TanJax
        w = TanJax(jnp.array([0., 0.]), jnp.array([[-1e-3, 0.], [0., 1e-3]]),
                   jnp.array([0., 0.]))
        d = w.unproj_offset(jnp.array(5.0), jnp.array(5.0))
        assert np.all(np.isfinite(np.asarray(d[0])))
        n = w.pix_to_native(jnp.array(5.0), jnp.array(5.0))
        assert np.all(np.isfinite(np.asarray(n[0])))
        print("ok")
    """)
    assert "ok" in proc.stdout


def test_f32_sip_reprojection_has_no_f64_ops():
    """A single f64 anywhere in the jaxpr would defeat the fp32 path."""
    proc = run("""
        import jax, jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)  # build, then inspect types
        from mapprojax import Tan, Reprojector

        src = Tan([100., 100.], [[-1e-3, 0.], [0., 1e-3]], [180., 30.],
                  sip=__import__("mapprojax.sip", fromlist=["Sip"]).Sip(
                      a=[[0., 0.], [0., 0.], [1e-5, 0.]],
                      b=[[0., 0.], [0., 0.], [0., 1e-5]]))
        tgt = Tan([100., 100.], [[-1e-3, 0.], [0., 1e-3]], [180., 30.])
        rp = Reprojector(src, tgt, matrix_precision="f32")
        jaxpr = jax.make_jaxpr(lambda x, y: rp.pix_to_pix(x, y))(
            jnp.float32(3.), jnp.float32(4.))
        bad = [str(e.primitive) for e in jaxpr.jaxpr.eqns
               if any(getattr(v, "aval", None) is not None
                      and getattr(v.aval, "dtype", None) == jnp.float64
                      for v in e.outvars)]
        assert not bad, bad
        print("ok")
    """)
    assert "ok" in proc.stdout


@pytest.mark.parametrize("cls", ["TanJax", "TanJaxArray", "TanJaxMixed", "TanJaxArrayMixed"])
def test_absolute_methods_raise_after_x64_is_switched_off(cls):
    """Built with x64 on, used with it off: the stored leaves are still
    float64 but every operation now truncates, so the check is on effective
    precision. Every absolute method goes through the same rotation guard,
    including ``unproj_xyz`` on the Mixed array class (it had none)."""
    proc = run(f"""
        import jax, jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        import mapprojax as mp
        arr = "Array" in "{cls}"
        crval = (jnp.array([10.0]), jnp.array([20.0])) if arr else jnp.array([10.0, 20.0])
        w = mp.{cls}(jnp.zeros(2), jnp.array([[1e-3, 0.], [0., 1e-3]]), crval)
        jax.config.update("jax_enable_x64", False)
        x = jnp.array([1.0]); y = jnp.array([2.0])
        calls = {{
            "proj": lambda: w.proj(x * 0.1, y * 0.1),
            "unproj": lambda: w.unproj(x, y),
            "proj_xyz": lambda: w.proj_xyz(x, x, x),
            "unproj_xyz": lambda: w.unproj_xyz(x, y),
        }}
        for name, call in calls.items():
            try:
                call()
            except TypeError:
                continue
            raise SystemExit(f"{{name}} did not raise")
        print("ok")
    """)
    assert "ok" in proc.stdout


def test_array_guard_checks_both_reference_components():
    """A float32 Dec array next to a float64 RA array is not float64."""
    proc = run("""
        import jax, jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        import mapprojax as mp
        w = mp.TanJaxArray(jnp.zeros(2), jnp.array([[1e-3, 0.], [0., 1e-3]]),
                           (jnp.array([10.0]), jnp.array([20.0])))
        w.crvals = (w.crvals[0], w.crvals[1].astype(jnp.float32))
        try:
            w.unproj(jnp.array([1.0]), jnp.array([2.0]))
        except TypeError:
            print("raised")
    """)
    assert "raised" in proc.stdout
