"""Warm JAX pixel-mapping timings and synthetic precision comparisons.

The reference is explicit JAX f64 TAN->SIN, not an independent NumPy or
Astropy oracle. Setup, transfers and compilation are excluded from timings.
Run CPU and ROCm in separate processes; use ROCm preallocation=false on a
shared-memory device. Defaults retain the production 512..8192 size sweep.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from mapprojax import absolute_from_offset, offset_from_absolute
from mapprojax.reproject import Reprojector
from mapprojax.wcs import Tan, Sin

CRPIX = [100.0, 100.0]
CD = [[-0.001, 0.0], [0.0, 0.001]]
CRVAL_SRC = [45.0, 30.0]
CRVAL_TGT = [45.5, 30.3]
DEFAULT_SIZES = (512, 1024, 2048, 4096, 8192)
SAMPLE_COUNT = 5000
DETECTOR_SIZE = 4096
DETECTOR_CRPIX = [2048.0, 2048.0]
PLATE_CASES = tuple(("cross-field", p, 1800.0) for p in (3.6, 0.1, 0.05)) + tuple(
    ("same-pointing", p, s)
    for p in (0.1, 0.05, 0.02) for s in (1.0, 10.0, 60.0, 300.0)
)
LABELS = {
    "wcs_eager_f64": "eager JAX WCS f64",
    "wcs_jit_f64": "jitted JAX WCS f64",
    "wcs_jit_mixed": "jitted JAX WCS mixed",
    "rp_jit_f32": "jitted Reprojector f32",
    "rp_jit_f64": "jitted Reprojector f64",
}
REFERENCE = "JAX WCS f64 TAN unproj -> SIN proj with float64 coordinates (not independent truth)"


@contextmanager
def x64_mode(enabled):
    previous = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", enabled)
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def wcs_pair(precision, crpix=CRPIX, cd=CD, src=CRVAL_SRC, tgt=CRVAL_TGT):
    # Retain concrete host float64 reference points, including in x64-off mode.
    return (Tan(crpix, cd, np.asarray(src, dtype=np.float64), precision=precision),
            Sin(crpix, cd, np.asarray(tgt, dtype=np.float64), precision=precision))


def build_paths(src, tgt, mixed_src, mixed_tgt):
    def absolute(ws, wt, x, y):
        return wt.proj(*ws.unproj(x, y))

    eager = lambda x, y: absolute(src, tgt, x, y)
    mixed = lambda x, y: absolute(mixed_src, mixed_tgt, x, y)
    rp32 = Reprojector(src, tgt, matrix_precision="f32")
    rp64 = Reprojector(src, tgt, matrix_precision="f64")
    return {
        "wcs_eager_f64": (eager, "float64"),
        "wcs_jit_f64": (jax.jit(eager), "float64"),
        "wcs_jit_mixed": (jax.jit(mixed), "float32"),
        "rp_jit_f32": (jax.jit(rp32.pix_to_pix), "float32"),
        "rp_jit_f64": (jax.jit(rp64.pix_to_pix), "float64"),
    }


def coordinates(x, y, dtype):
    return jax.block_until_ready((jnp.asarray(x, dtype=dtype), jnp.asarray(y, dtype=dtype)))


def measure_warm(function, args, warmups, repeats):
    """Wait for the complete returned PyTree on every call, eager included."""
    if warmups < 1 or repeats < 1:
        raise ValueError("warmups and repeats must be positive")
    jax.block_until_ready(args)
    for _ in range(warmups):
        jax.block_until_ready(function(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = jax.block_until_ready(function(*args))
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": float(np.median(samples)),
        "min_ms": min(samples), "max_ms": max(samples),
        "std_ms": float(np.std(samples)), "samples_ms": samples,
        "warmups": warmups, "repeats": repeats,
    }, output


def error_stats(output, reference):
    x, y = (np.asarray(v, dtype=np.float64) for v in output)
    rx, ry = (np.asarray(v, dtype=np.float64) for v in reference)
    if not (x.shape == y.shape == rx.shape == ry.shape) or not x.size:
        raise ValueError("empty or mismatched coordinate shapes")
    ref_valid = np.isfinite(rx) & np.isfinite(ry)
    valid = np.isfinite(x) & np.isfinite(y)
    overlap = ref_valid & valid
    counts = {"n_samples": int(x.size), "n_reference_finite": int(ref_valid.sum()),
              "n_output_finite": int(valid.sum()), "n_overlap": int(overlap.sum())}
    if not np.all(overlap):
        raise ValueError(f"unexpected invalid samples: {counts}")
    error = np.hypot(x - rx, y - ry)
    if not np.all(np.isfinite(error)):
        raise ValueError("non-finite pixel error")
    return {**counts, "max_px": float(error.max()), "median_px": float(np.median(error))}


def output_dtypes(output):
    return [str(v.dtype) for v in output]


def run_timing_benchmark(sizes, warmups, repeats, paths):
    rows = []
    for size in sizes:
        y, x = np.indices((size, size), dtype=np.float64)
        inputs = {dtype: coordinates(x, y, dtype) for dtype in ("float32", "float64")}
        del x, y
        # The explicit f64 reference and transfers are outside all timed regions.
        reference = jax.block_until_ready(paths["wcs_jit_f64"][0](*inputs["float64"]))
        error_stats(reference, reference)
        for name, (function, dtype) in paths.items():
            timing, output = measure_warm(function, inputs[dtype], warmups, repeats)
            rows.append({"size": size, "path": name, "label": LABELS[name],
                         "input_dtype": dtype, "output_dtypes": output_dtypes(output),
                         **timing, "accuracy": error_stats(output, reference)})
            del output
        del inputs, reference
    return rows


def offset_mapping(src, tgt, x, y, src_reference, tgt_reference):
    absolute = absolute_from_offset(
        src.unproj_offset(x, y), crval=src_reference, offset_scale=src.offset_scale())
    offset = offset_from_absolute(
        absolute, crval=tgt_reference, offset_scale=tgt.offset_scale())
    return tgt.proj_offset(*offset)


def offset_f32_without_x64(x, y):
    # Save originals before disabling x64; never reconstruct a producer's cosine.
    src_reference = np.asarray(CRVAL_SRC, dtype=np.float64)
    tgt_reference = np.asarray(CRVAL_TGT, dtype=np.float64)
    with x64_mode(False):
        src, tgt = wcs_pair("f32", crpix=jnp.asarray(CRPIX, dtype=jnp.float32),
                            cd=jnp.asarray(CD, dtype=jnp.float32),
                            src=src_reference, tgt=tgt_reference)
        inputs = coordinates(x, y, "float32")
        return jax.block_until_ready(offset_mapping(
            src, tgt, *inputs, src_reference, tgt_reference))


def report_accuracy(paths, sample_count=SAMPLE_COUNT):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 8192, sample_count)
    y = rng.uniform(0, 8192, sample_count)
    inputs = {dtype: coordinates(x, y, dtype) for dtype in ("float32", "float64")}
    reference = jax.block_until_ready(paths["wcs_jit_f64"][0](*inputs["float64"]))
    rows = []
    for name, (function, dtype) in paths.items():
        output = jax.block_until_ready(function(*inputs[dtype]))
        rows.append({"path": name, "label": LABELS[name], "input_dtype": dtype,
                     "output_dtypes": output_dtypes(output), **error_stats(output, reference)})
    src, tgt = wcs_pair("f64")
    output = jax.block_until_ready(offset_mapping(
        src, tgt, *inputs["float64"], np.asarray(CRVAL_SRC, dtype=np.float64),
        np.asarray(CRVAL_TGT, dtype=np.float64)))
    rows.append({"path": "offset_host_f64", "label": "f64 offsets + host absolute helpers",
                 "input_dtype": "float64", "output_dtypes": output_dtypes(output),
                 **error_stats(output, reference)})
    output = offset_f32_without_x64(x, y)
    rows.append({"path": "offset_host_f32_x64_off",
                 "label": "f32 offsets (x64 off) + host f64 absolute helpers",
                 "input_dtype": "float32", "output_dtypes": output_dtypes(output),
                 **error_stats(output, reference)})
    return rows


def detector_samples(sample_count):
    rng = np.random.default_rng(0)
    return (rng.uniform(0, DETECTOR_SIZE, sample_count),
            rng.uniform(0, DETECTOR_SIZE, sample_count))


def report_plate_scale_sensitivity(sample_count=SAMPLE_COUNT):
    x, y = detector_samples(sample_count)
    inputs = {dtype: coordinates(x, y, dtype) for dtype in ("float32", "float64")}
    rows = []
    for regime, plate, separation in PLATE_CASES:
        deg = plate / 3600.0
        src, tgt = wcs_pair("f64", crpix=DETECTOR_CRPIX,
                            cd=[[-deg, 0.0], [0.0, deg]], src=[180.0, 30.0],
                            tgt=[180.0, 30.0 + separation / 3600.0])
        reference = jax.block_until_ready(tgt.proj(*src.unproj(*inputs["float64"])))
        for precision, dtype in (("f32", "float32"), ("f64", "float64")):
            rp = Reprojector(src, tgt, matrix_precision=precision)
            output = jax.block_until_ready(rp.pix_to_pix(*inputs[dtype]))
            rows.append({"regime": regime, "plate_arcsec_per_px": plate,
                         "separation_arcsec": separation, "matrix_precision": precision,
                         "input_dtype": dtype, "output_dtypes": output_dtypes(output),
                         **error_stats(output, reference)})
    return rows


def report_metadata(argv):
    script = Path(__file__).resolve()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=script.parent,
                              capture_output=True, text=True, check=False)
    deps = {}
    for package in ("mapprojax", "jax", "jaxlib", "numpy", "jax-rocm7-plugin", "jax-rocm7-pjrt"):
        try:
            deps[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            deps[package] = None
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "script": script.name, "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "command": shlex.join([sys.executable, str(script), *argv]),
        "python": sys.version, "os": platform.platform(), "dependencies": deps,
        "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
        "device_kinds": [d.device_kind for d in jax.devices()],
        "platform_version": getattr(jax.devices()[0].client, "platform_version", None),
        "rocm_runtime": os.environ.get("ROCM_VERSION"),
        "environment": {k: os.environ.get(k) for k in
                        ("JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE", "ROCM_PATH")},
        "reference": REFERENCE,
        "dtypes": {"reference": "float64", "f64_coordinates": "float64",
                   "mixed_and_f32_coordinates": "float32", "host_absolute_helpers": "float64"},
        "sampling": {"seed": 0, "random_samples_per_case": SAMPLE_COUNT,
                     "accuracy_span_px": [0, 8192], "accuracy_crpix": CRPIX,
                     "accuracy_cd_deg_per_px": CD, "accuracy_crval_src_deg": CRVAL_SRC,
                     "accuracy_crval_tgt_deg": CRVAL_TGT,
                     "detector_shape": [DETECTOR_SIZE, DETECTOR_SIZE],
                     "detector_span_px": [0, DETECTOR_SIZE], "detector_crpix": DETECTOR_CRPIX,
                     "detector_src_crval_deg": [180.0, 30.0],
                     "detector_target_separation_axis": "declination",
                     "projection_pair": "TAN->SIN", "plate_cases": PLATE_CASES,
                     "timing_grid": "integer pixel coordinates [0,size) on each axis"},
        "units": {"timing": "ms", "error": "px", "plate_scale": "arcsec/px",
                  "separation": "arcsec"},
        "timing_policy": "perf_counter warm median; complete input/output tree waits; "
                         "setup, transfers and compilation excluded; no cold timing",
    }


def render_markdown(report):
    lines = ["# Precision benchmark", "", report["metadata"]["reference"], "",
             "Warm median milliseconds; setup, transfers and compilation excluded.", "",
             "| Size | Path | Input | Median ms | Min ms | Max ms |",
             "|---|---|---|---:|---:|---:|"]
    for row in report["timings"]:
        lines.append(f"| {row['size']}x{row['size']} | {row['label']} | {row['input_dtype']} | "
                     f"{row['median_ms']:.6g} | {row['min_ms']:.6g} | {row['max_ms']:.6g} |")
    lines += ["", "| Accuracy path | Max px | Median px | Finite overlap / samples |",
              "|---|---:|---:|---:|"]
    for row in report["accuracy"]:
        lines.append(f"| {row['label']} | {row['max_px']:.6g} | {row['median_px']:.6g} | "
                     f"{row['n_overlap']} / {row['n_samples']} |")
    lines += ["", "## Plate scale: 4096x4096 detector, CRPIX 2048, seed 0", "",
              "| Regime | arcsec/px | Separation arcsec | Matrix | Max px | Median px | Overlap / samples |",
              "|---|---:|---:|---|---:|---:|---:|"]
    for row in report["plate_scale"]:
        lines.append(f"| {row['regime']} | {row['plate_arcsec_per_px']} | {row['separation_arcsec']} | "
                     f"{row['matrix_precision']} | {row['max_px']:.6g} | {row['median_px']:.6g} | "
                     f"{row['n_overlap']} / {row['n_samples']} |")
    lines += ["", "## Metadata", "", "```json", json.dumps(report["metadata"], indent=2), "```", ""]
    return "\n".join(lines)


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=positive_int, default=list(DEFAULT_SIZES))
    parser.add_argument("--repeats", type=positive_int, default=10)
    parser.add_argument("--warmups", type=positive_int, default=3)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    with x64_mode(True):
        src, tgt = wcs_pair("f64")
        mixed_src, mixed_tgt = wcs_pair("mixed")
        paths = build_paths(src, tgt, mixed_src, mixed_tgt)
        report = {"metadata": report_metadata(argv),
                  "timings": run_timing_benchmark(args.sizes, args.warmups, args.repeats, paths),
                  "accuracy": report_accuracy(paths),
                  "plate_scale": report_plate_scale_sensitivity()}
        report["metadata"].update(sizes=args.sizes, warmups=args.warmups, repeats=args.repeats)
    text = render_markdown(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.markdown:
        args.markdown.write_text(text)
    print(text)
    return report


if __name__ == "__main__":
    main()
