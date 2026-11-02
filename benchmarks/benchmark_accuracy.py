"""Accuracy of mapprojax against astropy, from the shared evaluator.

Measures every case in ``tests/accuracy_reference.py`` (same sampling and
metrics as the tests) plus the real SPHEREx pair, and renders the published
table. No timing: this is an accuracy benchmark.

    JAX_PLATFORMS=cpu uv run python benchmarks/benchmark_accuracy.py \\
        --json /tmp/accuracy.json --markdown docs/benchmarks/accuracy_results.md

``--write-baselines`` regenerates ``tests/data/accuracy_baselines.json``
(regression caps = measured max x 1.2, rounded up to 2 significant figures).
It first checks the results against the *existing* caps and refuses if any
problem below is present, including a result above its old cap, so a
regression cannot be absorbed by rewriting the caps. Accepting a result
above its old cap needs ``--allow-cap-increase`` as well, and a reviewed
diff. A rewritten cap can still move up or down within that rule, since it
is recomputed as 1.2x the new measurement; review the diff either way.
New entries (a cap that did not exist before) are always allowed.

Exits non-zero if: the astropy oracle does not converge or its own round trip
exceeds 1e-8 px; any mapprojax output on a valid pixel or in a pair overlap is
non-finite; an fp64 result breaks its budget; an fp32 row or capped pair run
lacks a cap or exceeds it; an exact (fp64) pair run misses 1e-6 px; or the
transposed-rotation control is not grossly wrong.

Runs offline on the header fixtures in ``tests/data``; it does not depend on
the working directory or the original FITS files. JAX runs on CPU unless
``JAX_PLATFORMS`` says otherwise.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import astropy  # noqa: E402
import numpy as np  # noqa: E402
from astropy.wcs import NoConvergence  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import accuracy_reference as ar  # noqa: E402

BASELINE_MARGIN = 1.2
ORACLE_ROUNDTRIP_PX = 1e-8


def _cap(value):
    x = value * BASELINE_MARGIN
    e = math.floor(math.log10(x))
    return float(f"{math.ceil(x / 10 ** (e - 1)) * 10 ** (e - 1):.2g}")


def measure():
    problems = []
    oracle, rows = [], []
    for case in ar.cases():
        try:
            check = ar.oracle_self_check(case)
        except NoConvergence as exc:
            problems.append(f"oracle did not converge on {case.name}: {exc}")
            continue
        oracle.append(check)
        rt = check["roundtrip_px"]
        if rt["n_nonfinite"] or not rt["max"] < ORACLE_ROUNDTRIP_PX:
            problems.append(f"oracle round trip on {case.name}: {rt['max']:.3e} px")
        for impl in ar.implementations_for(case):
            r = ar.evaluate(case, impl)
            for metric, budget in (("forward", ar.FORWARD_MAS), ("inverse", ar.INVERSE_PX)):
                stats = r[ar.METRIC_KEYS[metric][0]]
                # fp32 paths are guarded by measured caps, not fp64 budgets.
                budget = ar.budget(impl, metric)
                within = budget is not None and stats["max"] < budget
                stats["budget"] = budget
                if budget is None:
                    stats["status"] = "cap"
                    stats["cap"] = None  # filled in by check_caps
                else:
                    stats["status"] = "pass" if within else "FAIL"
                if stats["n_nonfinite"]:
                    problems.append(f"{case.name}/{impl} {metric}: {stats['n_nonfinite']} non-finite")
                if stats["status"] == "FAIL":
                    problems.append(f"{case.name}/{impl} {metric}: {stats['max']:.3e} > {budget}")
            rows.append(r)

    pairs = []
    for direction, precision in ar.PAIR_RUNS:
        p = ar.evaluate_pair(precision, direction)
        s = p["pixel_error_px"]
        if s["n_nonfinite"] or not p["n_overlap"] or s["n"] != p["n_overlap"]:
            problems.append(f"pair {direction}/{precision}: empty overlap or non-finite output")
        if (direction, precision) in ar.PAIR_EXACT and not s["max"] < ar.TARGET_PX:
            problems.append(f"pair {direction}/{precision}: {s['max']:.3e} > target {ar.TARGET_PX}")
        pairs.append(p)

    controls = {}
    for d in ("forward", "reverse"):
        c = controls[d] = ar.transposed_rotation_control(d)
        if not c["detected"]:
            problems.append(f"transposed-rotation control {d} not detected: "
                            f"{c['n_caught']}/{c['n']} samples caught")
    return {"oracle": oracle, "cases": rows, "pairs": pairs,
            "transposed_rotation": controls}, problems


def _fp32_rows(result):
    """(metric, key, stats) for the fp32 path (``jax_mixed``), which is a
    different precision contract rather than a defect and so is capped on its
    own measured floor."""
    for r in result["cases"]:
        for metric, (field, _) in ar.METRIC_KEYS.items():
            if r[field].get("status") == "cap":
                yield metric, f"{r['case']}/{r['impl']}", r[field]


def _capped_pairs(result):
    for p in result["pairs"]:
        if (p["direction"], p["precision"]) not in ar.PAIR_EXACT:
            yield f"{p['direction']}/{p['precision']}", p


def check_caps(result, baselines):
    """Every fp32 row and every non-exact pair run must have a cap and stay
    within it; a missing cap is itself a problem."""
    problems = []
    for metric, key, stats in _fp32_rows(result):
        entry = baselines.get("fp32", {}).get(metric, {}).get(key)
        if entry is None:
            problems.append(f"no regression cap for fp32 {key} ({metric})")
            continue
        stats["cap"] = entry["cap"]
        if stats["max"] > entry["cap"]:
            problems.append(f"{key} {metric} {stats['max']:.3e} > cap {entry['cap']} {entry['unit']}")
    for key, p in _capped_pairs(result):
        entry = baselines["pair"].get(key)
        if entry is None:
            problems.append(f"no regression cap for pair {key}")
            continue
        s = p["pixel_error_px"]
        s["cap"] = entry["cap"]
        if s["max"] > entry["cap"]:
            problems.append(f"pair {key} {s['max']:.3e} > cap {entry['cap']} px")
    return problems


def baselines_from(result):
    def entry(stats, unit, sample, **extra):
        return {"sample": sample, **extra, "unit": unit, "measured_max": stats["max"],
                "measured_rms": stats["rms"], "cap": _cap(stats["max"])}

    grid = f"{ar.GRID_N}x{ar.GRID_N} grid + CRPIX"
    fp32 = {metric: {} for metric in ar.METRIC_KEYS}
    for metric, key, stats in _fp32_rows(result):
        fp32[metric][key] = entry(stats, ar.METRIC_KEYS[metric][1], grid)
    pair = {key: entry(p["pixel_error_px"], "px", f"{grid} on the source, astropy-decided overlap",
                       n_overlap=p["n_overlap"])
            for key, p in _capped_pairs(result)}
    env = environment()
    return {
        "_provenance": {
            "generated_by": "benchmarks/benchmark_accuracy.py --write-baselines",
            "versions": env["versions"],
            "backend": env["backend"],
            "dtype": ("float64, except the jax_mixed rows (fp32 pixels, crpix, cd and SIP; fp64 "
                      "crval and rotation) and pair runs keyed f32 (Reprojector matrix_precision='f32')"),
            "margin": f"cap = measured max x {BASELINE_MARGIN}, rounded up to 2 significant figures",
            "meaning": ("Regression caps on the jax_mixed fp32 floor and on the fp32 "
                        "Reprojector. Not accuracy claims; the fp64 paths are held to the "
                        "budgets in tests/accuracy_reference.py instead."),
        },
        "fp32": fp32,
        "pair": pair,
    }


def environment():
    return {"backend": jax.default_backend(),
            "versions": {"jax": jax.__version__, "numpy": np.__version__,
                         "astropy": astropy.__version__}}


def _g(x):
    return f"{x:.2e}"


def _status(stats):
    if stats.get("cap") is not None:
        return f"fp32 cap ({stats['cap']:g})"
    return stats["status"]


def render_markdown(result, env):
    v = env["versions"]
    out = [
        "# Accuracy against astropy",
        "",
        "Generated by `benchmarks/benchmark_accuracy.py` from the shared evaluator",
        "`tests/accuracy_reference.py`; the tests in `tests/test_astropy_reference.py` and",
        "`tests/test_reproject_astropy.py` assert the same numbers. Do not edit by hand.",
        "",
        f"- Backend: `{env['backend']}`, fp64 (`jax_enable_x64`) unless stated. "
        f"JAX {v['jax']}, NumPy {v['numpy']}, astropy {v['astropy']}.",
        f"- Sampling: {ar.GRID_N}x{ar.GRID_N} pixel centres spanning `0 .. NAXIS-1` inclusive, plus CRPIX.",
        "- Forward: great-circle separation between mapprojax `unproj` and astropy",
        "  `all_pix2world(origin=0)`, in milliarcseconds.",
        "- Inverse: mapprojax `proj` of the astropy sky position against the original pixel.",
        f"  The oracle's own converged inverse (`all_world2pix`, tol {ar.ORACLE_TOL:g}) is reported too.",
        f"- Budgets (fp64 paths): forward < {ar.FORWARD_MAS:g} mas; inverse < {ar.INVERSE_PX:g} px.",
        "- `numpy_absolute` is `unproj_absolute` / `proj_absolute`, measured on every case,",
        "  including the exact-pole `1904-66_*` fixtures (CRVAL2 = -90).",
        "- `jax_mixed` is `TanJaxMixed` / `SinJaxMixed` fed f32 pixels: fp32 pixels, `crpix`,",
        "  `cd` and SIP; fp64 `crval` and rotation. That is a different precision contract, so it is",
        "  held to a measured cap per case and metric (`fp32 cap X`) from the `fp32` section of",
        "  `tests/data/accuracy_baselines.json`, not to the fp64 budgets.",
        "- No timings: this is an accuracy benchmark.",
        "",
        "## Single WCS",
        "",
        "| case | impl | fwd max (mas) | fwd rms (mas) | fwd | inv max (px) | inv rms (px) | inv |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["cases"]:
        f, i = r["forward_mas"], r["inverse_px"]
        out.append(f"| {r['case']} | {r['impl']} | {_g(f['max'])} | {_g(f['rms'])} | {_status(f)} "
                   f"| {_g(i['max'])} | {_g(i['rms'])} | {_status(i)} |")

    out += ["", "Oracle self-consistency (astropy pixel -> sky -> pixel, converged):", "",
            "| case | max (px) |", "|---|---|"]
    out += [f"| {o['case']} | {_g(o['roundtrip_px']['max'])} |" for o in result["oracle"]]

    out += [
        "",
        "## Real SPHEREx pair (`Reprojector`)",
        "",
        "- forward: Galactic template pixel -> L2 exposure pixel (rotation Galactic -> ICRS).",
        "  Ends in the L2 SIP inverse, a Newton solve against A/B seeded by AP/BP.",
        "- reverse: L2 pixel -> template pixel (rotation ICRS -> Galactic). Uses only the SIP",
        "  forward and a SIP-free inverse, so in fp64 it isolates the frame rotation.",
        "- The fp32 runs are limited by float32 precision, not by the solver.",
        "- Overlap is decided from astropy's destination pixel. `cap` is the regression cap from",
        "  `tests/data/accuracy_baselines.json`, not an accuracy claim.",
        "",
        "| direction | matrix_precision | overlap | max (px) | rms (px) | cap (px) |",
        "|---|---|---|---|---|---|",
    ]
    for p in result["pairs"]:
        s = p["pixel_error_px"]
        cap = s.get("cap", f"target {ar.TARGET_PX:g}")
        out.append(f"| {p['direction']} | {p['precision']} | {p['n_overlap']}/{p['n_samples']} "
                   f"| {_g(s['max'])} | {_g(s['rms'])} | {cap} |")
    def _ctl(c):
        if c["n_rejected"] == c["n"]:
            return "all samples rejected as invalid"
        return (f"min error {_g(c['min_finite_px'])} px, "
                f"{c['n_rejected']} of {c['n']} rejected")

    c = result["transposed_rotation"]
    out += ["", "Negative control, transposed rotation (fp64), caught on every overlap sample "
                "(rejected, or wrong by more than max(1 px, 100x the correct run)): "
                f"forward {_ctl(c['forward'])}; reverse {_ctl(c['reverse'])}.", ""]

    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, help="write machine-readable results here")
    ap.add_argument("--markdown", type=Path, help="write the rendered table here")
    ap.add_argument("--write-baselines", action="store_true",
                    help=f"regenerate {ar.BASELINES_PATH.relative_to(REPO)} (review the diff)")
    ap.add_argument("--allow-cap-increase", action="store_true",
                    help="with --write-baselines: accept results above their old caps")
    args = ap.parse_args(argv)
    if args.allow_cap_increase and not args.write_baselines:
        ap.error("--allow-cap-increase only applies with --write-baselines")

    result, problems = measure()
    env = environment()
    if args.write_baselines:
        old_cap_problems = [p for p in check_caps(result, ar.load_baselines())
                            if not p.startswith("no regression cap")]
        blocking = problems + ([] if args.allow_cap_increase else old_cap_problems)
        if blocking:
            for p in blocking:
                print(f"PROBLEM: {p}", file=sys.stderr)
            print("refusing to write baselines", file=sys.stderr)
            return 1
        ar.BASELINES_PATH.write_text(json.dumps(baselines_from(result), indent=2) + "\n")
    problems += check_caps(result, ar.load_baselines())

    if args.json:
        args.json.write_text(json.dumps({"environment": env, **result}, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(render_markdown(result, env))
    if not (args.json or args.markdown):
        print(render_markdown(result, env))

    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
