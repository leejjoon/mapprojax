"""Stage 3 performance baseline and regression harness."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import numpy as np


def _worker(kind: str, repeats: int, warmups: int, size: int,
            workload: str, order: int, steps: int, implementation: str,
            pair_precision: str) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from mapprojax.sip import Sip

    device_text = " ".join(repr(d) for d in jax.devices())
    if os.environ.get("JAX_PLATFORMS") == "rocm":
        identity = device_text + jax.devices()[0].client.platform_version
        if "rocm" not in identity.lower():
            raise RuntimeError(f"ROCm requested but device identity is {identity!r}")

    if implementation != "new":
        raise ValueError(
            "the legacy classes were retired in Stage 3 C8; measure the legacy "
            "baseline with benchmarks/benchmark_stage3.py from a d60a5b1 checkout"
        )

    if workload == "sip":
        axis = jnp.linspace(0.0, float(size - 1), size, dtype=jnp.float32)
        x, y = jnp.meshgrid(axis, axis)
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "tests"))
        from accuracy_reference import load_header
        from mapprojax.wcs_utils import load_wcs_params_from_header
        header = load_header("spherex_l2.hdr")
        crpix, cd, crval, sip = load_wcs_params_from_header(header)
        if order != 3:
            rng = np.random.default_rng(0)
            a = rng.uniform(-1e-9, 1e-9, (order + 1, order + 1))
            b = rng.uniform(-1e-9, 1e-9, (order + 1, order + 1))
            sip = Sip(a=a, b=b, inverse_steps=steps)
        sip = Sip(a=None if sip.a is None else jnp.asarray(sip.a, dtype=jnp.float32),
                  b=None if sip.b is None else jnp.asarray(sip.b, dtype=jnp.float32),
                  ap=None if sip.ap is None else jnp.asarray(sip.ap, dtype=jnp.float32),
                  bp=None if sip.bp is None else jnp.asarray(sip.bp, dtype=jnp.float32),
                  inverse_steps=steps)
        from mapprojax.wcs import Tan
        wcs = Tan(crpix, cd, jnp.asarray(crval, dtype=jnp.float64),
                  sip=sip, precision="mixed")
        if order == 3:
            # Real SPHEREx header: forward evaluation plus the SIP inverse.
            fn = jax.jit(lambda obj, px, py: (
                obj.unproj_offset(px, py),
                obj.proj_offset(*obj.unproj_offset(px, py)),
            ))
        else:
            # Synthetic higher-order coefficients: evaluation only. Their
            # inverse does not converge over the grid, so timing it would
            # measure work on samples that end up invalid.
            fn = jax.jit(lambda obj, px, py: obj.unproj_offset(px, py))
        args = (wcs, x, y)
    elif workload == "batch":
        x = jnp.linspace(0.0, 4095.0, 4096, dtype=jnp.float32)
        y = jnp.linspace(0.0, 4095.0, 4096, dtype=jnp.float32)
        from mapprojax.wcs import Tan
        one = Tan([2048.0, 2048.0], [[-1e-3, 0], [0, 1e-3]],
                  [180.0, 0.0], precision="mixed")
        objects = jax.tree_util.tree_map(
            lambda value: jnp.broadcast_to(value, (256,) + value.shape), one)
        fn = jax.jit(lambda obj, px, py: jax.vmap(
            lambda one_wcs: one_wcs.unproj_offset(px, py))(obj))
        args = (objects, x, y)
    elif workload == "pair":
        from mapprojax.wcs import Tan
        from mapprojax.reproject import Reprojector
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "tests"))
        from accuracy_reference import load_header
        from mapprojax.wcs_utils import load_wcs_params_from_header
        src = load_wcs_params_from_header(load_header("spherex_l2.hdr"))
        tgt = load_wcs_params_from_header(load_header("spherex_template.hdr"))
        source = Tan(*src, precision="f64")
        target = Tan(*tgt, precision="f64")
        from accuracy_reference import galactic_to_icrs_matrix
        frame_rotation = galactic_to_icrs_matrix().T
        reproj = Reprojector(source, target, frame_rotation=frame_rotation,
                             matrix_precision=pair_precision)
        axis = jnp.linspace(0.0, float(size - 1), size, dtype=jnp.float32)
        x, y = jnp.meshgrid(axis, axis)
        fn = jax.jit(lambda obj, px, py: obj.pix_to_pix(px, py))
        args = (reproj, x, y)
    else:
        raise ValueError(f"unknown workload {workload!r}")

    if kind == "compile":
        start = time.perf_counter()
        lowered = fn.lower(*args)
        lowering_seconds = time.perf_counter() - start
        start = time.perf_counter()
        lowered.compile()
        compile_seconds = time.perf_counter() - start
        print(json.dumps({"lowering_seconds": lowering_seconds,
                          "compile_seconds": compile_seconds,
                          "backend": jax.default_backend(),
                          "devices": device_text}))
        return
    if kind == "cold":
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        cold_seconds = time.perf_counter() - start
        print(json.dumps({"cold_seconds": cold_seconds,
                          "backend": jax.default_backend(),
                          "devices": device_text}))
        return
    jax.block_until_ready(fn(*args))
    for _ in range(warmups):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    print(json.dumps({"samples": samples, "backend": jax.default_backend(),
                      "devices": device_text}))


def _run_process(backend: str, kind: str, repeats: int, warmups: int,
                 size: int, workload: str, order: int, steps: int,
                 implementation: str, pair_precision: str) -> dict:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = backend
    env["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    cache_dir = tempfile.TemporaryDirectory(prefix="mapprojax-stage3-")
    env["JAX_COMPILATION_CACHE_DIR"] = cache_dir.name
    if backend == "rocm":
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", kind,
               "--repeats", str(repeats), "--warmups", str(warmups),
               "--size", str(size), "--workload", workload]
    command += ["--order", str(order)]
    command += ["--steps", str(steps)]
    command += ["--implementation", implementation]
    command += ["--pair-precision", pair_precision]
    with cache_dir:
        result = subprocess.run(command, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def measure(backend: str, mode: str, size: int, workload: str,
            order: int, steps: int, implementation: str,
            pair_precision: str) -> dict:
    if mode == "smoke":
        processes, warmups, repeats, compile_processes = 1, 1, 3, 1
    else:
        processes, warmups, repeats, compile_processes = 3, 5, 30, 5
    warm = [_run_process(backend, "warm", repeats, warmups, size, workload, order,
                         steps, implementation, pair_precision)
            for _ in range(processes)]
    compile_records = [_run_process(backend, "compile", 1, 0, size, workload, order,
                                    steps, implementation, pair_precision)
                for _ in range(compile_processes)]
    cold_records = [_run_process(backend, "cold", 1, 0, size, workload, order,
                                 steps, implementation, pair_precision)
                    for _ in range(compile_processes)]
    medians = [statistics.median(item["samples"]) for item in warm]
    return {
        "backend_requested": backend, "mode": mode, "workload": workload, "size": size,
        "order": order, "inverse_steps": steps,
        "batch_shape": [256, 4096] if workload == "batch" else None,
        "implementation": implementation,
        "pair_precision": pair_precision if workload == "pair" else None,
        "warm_samples_seconds": [item["samples"] for item in warm],
        "warm_process_medians_seconds": medians,
        "warm_median_of_process_medians_seconds": statistics.median(medians),
        "call": ("unproj_offset" if workload == "sip" and order != 3 else
                 "unproj_offset+proj_offset" if workload == "sip" else
                 "unproj_offset" if workload == "batch" else "pix_to_pix"),
        "lowering_samples_seconds": [item["lowering_seconds"] for item in compile_records],
        "lowering_median_seconds": statistics.median(
            item["lowering_seconds"] for item in compile_records),
        "compile_samples_seconds": [item["compile_seconds"] for item in compile_records],
        "compile_median_seconds": statistics.median(
            item["compile_seconds"] for item in compile_records),
        "cold_samples_seconds": [item["cold_seconds"] for item in cold_records],
        "cold_median_seconds": statistics.median(
            item["cold_seconds"] for item in cold_records),
        "devices": warm[0]["devices"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "rocm"), default="cpu")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--workload", choices=("sip", "batch", "pair"), default="sip")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--implementation", choices=("new", "legacy"), default="new")
    parser.add_argument("--pair-precision", choices=("f32", "f64"), default="f32")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("warm", "compile", "cold"))
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args(argv)
    if args.implementation != "new":
        parser.error("the legacy classes were retired in Stage 3 C8; measure the "
                     "legacy baseline with this script from a d60a5b1 checkout")
    if args.worker:
        _worker(args.worker, args.repeats, args.warmups, args.size or 64,
                args.workload, args.order, args.steps, args.implementation,
                args.pair_precision)
        return 0
    size = args.size or (64 if args.mode == "smoke" else 2040)
    result = {"environment": {"python": sys.version, "backend": args.backend},
              "measurement": measure(args.backend, args.mode, size, args.workload,
                                     args.order, args.steps, args.implementation,
                                     args.pair_precision)}
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
