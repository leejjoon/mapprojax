"""Compare matched Stage 3 baseline/candidate timing artifacts.

usage: compare_stage3.py [--output FILE] BASELINE CANDIDATE [BASELINE CANDIDATE ...]

Exits 1 if any pair fails a rule, 2 if a pair is not matched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (metric, relative limit, absolute floor in seconds). The compile limit also
# applies to lowering, which the harness records separately.
RULES = (
    ("warm_median_of_process_medians_seconds", 1.10, 0.05e-3),
    ("lowering_median_seconds", 1.20, 0.05),
    ("compile_median_seconds", 1.20, 0.05),
    ("cold_median_seconds", 1.20, 0.05),
)
HISTORICAL_FIELDS = (
    "backend_requested", "mode", "workload", "call", "size", "order",
    "inverse_steps", "batch_shape", "pair_precision", "devices",
)
CURRENT_FIELDS = tuple(field for field in HISTORICAL_FIELDS if field != "call") + (
    "input_dtype", "kernel_precision", "sip", "bounds_shape", "bounds",
    "direction", "semantic_workload", "backend_actual",
)


def compare(baseline: Path, candidate: Path) -> dict:
    b_root = json.loads(baseline.read_text())
    c_root = json.loads(candidate.read_text())
    b, c = b_root["measurement"], c_root["measurement"]
    roles = (b.get("implementation"), c.get("implementation"))
    if roles == ("baseline", "new"):
        fields = CURRENT_FIELDS
        historical = False
    elif roles == ("legacy", "new"):
        fields = HISTORICAL_FIELDS
        historical = True
    else:
        raise ValueError("pair roles must be current baseline/new or historical legacy/new")
    if not historical:
        if b.get("workload") == "pixel_map":
            expected_calls = {
                "baseline": "pix_to_pix+finite_half_open_bounds+zero_selection",
                "new": "pixel_map",
            }
            valid_calls = (
                b.get("call") == expected_calls["baseline"]
                and c.get("call") == expected_calls["new"]
                and b.get("semantic_call") == expected_calls["baseline"]
                and c.get("semantic_call") == expected_calls["new"]
            )
        elif b.get("workload") == "pair":
            valid_calls = (
                b.get("call") == c.get("call") == "pix_to_pix"
                and b.get("semantic_workload") == c.get("semantic_workload") == "pair_pixel_mapping"
            )
        else:
            valid_calls = False
        if not valid_calls:
            raise ValueError("current baseline/new pair has incompatible semantics")
        missing = [field for field in fields if field not in b or field not in c]
        if missing:
            raise ValueError(f"missing current workload metadata: {', '.join(missing)}")
    if any(b.get(field) != c.get(field) for field in fields):
        raise ValueError("baseline and candidate workload metadata differ")
    if b_root.get("environment") != c_root.get("environment"):
        raise ValueError("baseline and candidate runtime environments differ")
    values = {}
    for name, limit, floor in RULES:
        if name not in b or name not in c:
            if historical and name == "lowering_median_seconds":
                continue  # artifacts recorded before lowering was measured
            raise ValueError(f"missing {name}")
        before, after = b[name], c[name]
        budget = max(limit * before, before + floor)
        values[name] = {"baseline": before, "candidate": after,
                        "budget": budget, "passed": after <= budget}
    metadata = {k: b.get(k) for k in fields}
    if not historical:
        metadata["calls"] = {"baseline": b["call"], "candidate": c["call"]}
        metadata["provenance"] = {
            "baseline": b.get("source_identity"),
            "candidate": c.get("source_identity"),
        }
    return {"baseline": str(baseline), "candidate": str(candidate),
            "metadata": metadata,
            "rules": values, "all_passed": all(v["passed"] for v in values.values())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if len(args.paths) % 2:
        parser.error("paths must come in BASELINE CANDIDATE pairs")
    try:
        results = [compare(args.paths[i], args.paths[i + 1])
                   for i in range(0, len(args.paths), 2)]
    except ValueError as exc:
        print(f"unmatched pair: {exc}", file=sys.stderr)
        return 2
    text = json.dumps({"timing": results}, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    failed = [r for r in results if not r["all_passed"]]
    for r in failed:
        print(f"FAILED: {r['candidate']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
