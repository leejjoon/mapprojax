"""
Benchmark: Performance Scaling with Resolution, and Precision-Mode Comparison

Tests:
    How JAX-accelerated geometric mapping scales with data volume compared to
    standard NumPy implementations across image resolutions from 512x512 to
    8192x8192, across the precision strategies this package now offers:

    - NumPy (f64): full double precision, the correctness reference.
    - JAX Std (f64): TanJax/SinJax at whatever precision jax_enable_x64
      allows (this script enables it, for a fair "current default" baseline).
    - JAX Mixed (Opt): MixedPrecisionWCSJaxMixin -- fp64 rotation, fp32
      everything else (see DEVELOPER.md Sec 4).
    - Reprojector (f32) [fo64_report_and_plan.md Phase 1]: composes the two
      WCSes' relative rotation once, in true float64, on the host -- then
      runs the *entire* per-pixel path in float32. No jax_enable_x64 needed
      at all, unlike every mode above.
    - Reprojector (f64): same code path, matrix_precision="f64" -- included
      to isolate Reprojector's own overhead from the fp32-vs-fp64 accuracy
      question.

    Offset-form absolute output (Phase 2: unproj_absolute/proj_absolute) is
    reported separately in the accuracy section below, not the per-size
    timing table -- it answers a different question ("give me true RA/Dec
    without global fp64", a host-side NumPy epilogue) than the others here
    ("map a pixel in one WCS to a pixel in another").

    f32x2 (Phase 1's proposed hi/lo matrix split) and a df64 rotation
    variant (Phase 3) are NOT included: f32x2 was implemented and measured
    during Phase 1 and found not to work as described (see
    mapprojax.reproject.Reprojector's docstring) -- a plain float32 add of
    the hi/lo products rounds the correction away regardless of matrix
    precision, so there is nothing distinct to benchmark. Phase 3 (a real
    double-single/df64 rotation) was never implemented.

Run twice to compare backends -- JAX picks its backend at process start and
does not support switching mid-process:

    python benchmarks/benchmark_precision.py                  # default backend (GPU if present)
    JAX_PLATFORMS=cpu python benchmarks/benchmark_precision.py # forced CPU

Accuracy, not just speed, is reported once (accuracy does not depend on
image resolution, only on the WCS pair): a speed number without an
accompanying accuracy number is not a fair comparison for this workload,
since the whole point of the faster modes is a documented accuracy
tradeoff, not a free lunch.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

# Enable x64 globally for JAX -- needed for the "JAX Std (f64)" and
# "JAX Mixed" baselines and for the NumPy/Reprojector(f64) ground truth;
# Reprojector(f32) and offset-form are the two modes in this file that do
# NOT depend on this being set (see the accuracy section's explicit check).
jax.config.update("jax_enable_x64", True)

from mapprojax import Tan, Sin, TanJax, SinJax, TanJaxMixed, SinJaxMixed
from mapprojax.reproject import Reprojector

CRPIX = [100.0, 100.0]
CD = [[-0.001, 0.0], [0.0, 0.001]]
CRVAL_SRC = [45.0, 30.0]
# A genuine (if modest) pointing difference from CRVAL_SRC, not the same
# point -- a zero-separation src/tgt pair would make M_rel exactly the
# identity, which is the one case where every precision mode looks
# identical and the whole point of this benchmark (how each mode handles
# a *real* reprojection) goes untested.
CRVAL_TGT = [45.5, 30.3]


def run_timing_benchmark():
    sizes = [512, 1024, 2048, 4096, 8192]

    print(f"backend={jax.default_backend()} devices={jax.devices()}\n")
    print(f"{'Size':<12} | {'NumPy (f64)':<15} | {'JAX Std (f64)':<15} | {'JAX Mix (Opt)':<15} | "
          f"{'Reprojector f32':<16} | {'Reprojector f64':<16}")
    print("-" * 108)

    proj2_std_jit = jax.jit(lambda ws, wt, x, y: (wt.proj(*ws.unproj(x, y))))
    proj2_mix_jit = jax.jit(lambda ws, wt, x, y: (wt.proj(*ws.unproj(x, y))))

    # Reprojector doesn't depend on image size -- built once, outside the
    # loop, like the jit wrappers above.
    wcs_src_np = Tan(CRPIX, CD, CRVAL_SRC)
    wcs_tgt_np = Sin(CRPIX, CD, CRVAL_TGT)
    rp_f32 = Reprojector(wcs_src_np, wcs_tgt_np, matrix_precision="f32")
    rp_f64 = Reprojector(wcs_src_np, wcs_tgt_np, matrix_precision="f64")
    rp_f32_jit = jax.jit(rp_f32.pix_to_pix)
    rp_f64_jit = jax.jit(rp_f64.pix_to_pix)

    for N in sizes:
        y_np, x_np = np.indices((N, N), dtype=np.float64)
        y_jax, x_jax = jnp.array(y_np, dtype=jnp.float32), jnp.array(x_np, dtype=jnp.float32)
        y_jax64, x_jax64 = jnp.array(y_np, dtype=jnp.float64), jnp.array(x_np, dtype=jnp.float64)

        # --- NumPy ---
        wcs_src_std_np = Tan(CRPIX, CD, CRVAL_SRC)
        wcs_tgt_std_np = Sin(CRPIX, CD, CRVAL_TGT)
        start = time.time()
        ra_np, dec_np = wcs_src_std_np.unproj(x_np, y_np)
        _ = wcs_tgt_std_np.proj(ra_np, dec_np)
        t_np = (time.time() - start) * 1000

        # --- JAX Standard ---
        wcs_src_std = TanJax(CRPIX, CD, CRVAL_SRC)
        wcs_tgt_std = SinJax(CRPIX, CD, CRVAL_TGT)
        out_std = proj2_std_jit(wcs_src_std, wcs_tgt_std, x_jax, y_jax)
        out_std[0].block_until_ready()
        start = time.time()
        out_std = proj2_std_jit(wcs_src_std, wcs_tgt_std, x_jax, y_jax)
        out_std[0].block_until_ready()
        t_std = (time.time() - start) * 1000

        # --- JAX Mixed ---
        wcs_src_mix = TanJaxMixed(CRPIX, CD, CRVAL_SRC)
        wcs_tgt_mix = SinJaxMixed(CRPIX, CD, CRVAL_TGT)
        out_mix = proj2_mix_jit(wcs_src_mix, wcs_tgt_mix, x_jax, y_jax)
        out_mix[0].block_until_ready()
        start = time.time()
        out_mix = proj2_mix_jit(wcs_src_mix, wcs_tgt_mix, x_jax, y_jax)
        out_mix[0].block_until_ready()
        t_mix = (time.time() - start) * 1000

        # --- Reprojector f32 ---
        out_rp32 = rp_f32_jit(x_jax, y_jax)
        out_rp32[0].block_until_ready()
        start = time.time()
        out_rp32 = rp_f32_jit(x_jax, y_jax)
        out_rp32[0].block_until_ready()
        t_rp32 = (time.time() - start) * 1000

        # --- Reprojector f64 ---
        out_rp64 = rp_f64_jit(x_jax64, y_jax64)
        out_rp64[0].block_until_ready()
        start = time.time()
        out_rp64 = rp_f64_jit(x_jax64, y_jax64)
        out_rp64[0].block_until_ready()
        t_rp64 = (time.time() - start) * 1000

        print(f"{N}x{N:<7} | {t_np:<15.2f} | {t_std:<15.2f} | {t_mix:<15.2f} | {t_rp32:<16.2f} | {t_rp64:<16.2f}")


def report_accuracy():
    print("\n=== Accuracy (max pixel error over 5000 random points, vs NumPy f64 ground truth) ===")
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 8192, 5000)
    y = rng.uniform(0, 8192, 5000)

    wcs_src_np = Tan(CRPIX, CD, CRVAL_SRC)
    wcs_tgt_np = Sin(CRPIX, CD, CRVAL_TGT)
    ra_ref, dec_ref = wcs_src_np.unproj(x, y)
    x_ref, y_ref = wcs_tgt_np.proj(ra_ref, dec_ref)

    def max_err(x_out, y_out):
        return np.max(np.hypot(np.asarray(x_out, dtype=np.float64) - x_ref, np.asarray(y_out, dtype=np.float64) - y_ref))

    wcs_src_mix = TanJaxMixed(CRPIX, CD, CRVAL_SRC)
    wcs_tgt_mix = SinJaxMixed(CRPIX, CD, CRVAL_TGT)
    ra_m, dec_m = wcs_src_mix.unproj(jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32))
    x_m, y_m = wcs_tgt_mix.proj(ra_m, dec_m)
    err_mix = max_err(x_m, y_m)

    rp_f32 = Reprojector(wcs_src_np, wcs_tgt_np, matrix_precision="f32")
    x_rp32, y_rp32 = rp_f32.pix_to_pix(x, y)
    err_rp32 = max_err(x_rp32, y_rp32)

    rp_f64 = Reprojector(wcs_src_np, wcs_tgt_np, matrix_precision="f64")
    x_rp64, y_rp64 = rp_f64.pix_to_pix(x, y)
    err_rp64 = max_err(x_rp64, y_rp64)

    # Offset-form (Phase 2): a different question -- absolute (RA, Dec)
    # output -- but round-tripping src -> (ra, dec) -> tgt through it
    # gives a directly comparable pixel-space number.
    ra_off, dec_off = wcs_src_np.unproj_absolute(x, y)
    x_off, y_off = wcs_tgt_np.proj_absolute(ra_off, dec_off)
    err_off = max_err(x_off, y_off)

    # Same, but with unproj_offset's bulk math actually run in float32
    # with jax_enable_x64 off, to demonstrate the "no x64 needed" claim
    # rather than just asserting it.
    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        from mapprojax.jax_projections import TanJax as _TanJax
        wcs_src_jax32 = _TanJax(jnp.asarray(CRPIX), jnp.asarray(CD), jnp.asarray(CRVAL_SRC))
        dalpha_cosdec, ddelta = wcs_src_jax32.unproj_offset(jnp.asarray(x), jnp.asarray(y))
        dec0 = np.radians(CRVAL_SRC[1])
        dalpha = np.asarray(dalpha_cosdec, dtype=np.float64) / np.cos(dec0)
        ddelta = np.asarray(ddelta, dtype=np.float64)
        ra_off32 = np.radians(CRVAL_SRC[0]) + dalpha
        dec_off32 = dec0 + ddelta
        x_off32, y_off32 = wcs_tgt_np.proj_absolute(ra_off32, dec_off32)
        err_off32 = max_err(x_off32, y_off32)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)

    print(f"  JAX Mixed (fp64 rotation + fp32 rest):                     {err_mix:.3e} px")
    print(f"  Reprojector (f32, no jax_enable_x64 needed):               {err_rp32:.3e} px")
    print(f"  Reprojector (f64):                                         {err_rp64:.3e} px")
    print(f"  Offset-form (unproj_absolute/proj_absolute, NumPy f64):    {err_off:.3e} px")
    print(f"  Offset-form's unproj_offset in fp32, jax_enable_x64=False: {err_off32:.3e} px")


if __name__ == "__main__":
    run_timing_benchmark()
    report_accuracy()
