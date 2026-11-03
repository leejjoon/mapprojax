"""Compare matched Stage 3 baseline/candidate timing artifacts."""
from __future__ import annotations
import json
import sys
from pathlib import Path


def compare(baseline: Path, candidate: Path) -> dict:
    b = json.loads(baseline.read_text())["measurement"]
    b_root = json.loads(baseline.read_text())
    c_root = json.loads(candidate.read_text())
    c = c_root["measurement"]
    if b.get("implementation") != "legacy" or c.get("implementation") != "new":
        raise ValueError("baseline must be legacy and candidate must be new")
    fields = ("backend_requested", "mode", "workload", "size", "order",
              "inverse_steps", "batch_shape", "pair_precision", "devices")
    if any(b.get(field) != c.get(field) for field in fields):
        raise ValueError("baseline and candidate workload metadata differ")
    if b_root.get("environment") != c_root.get("environment"):
        raise ValueError("baseline and candidate runtime environments differ")
    values = {}
    for name, limit in (("warm_median_of_process_medians_seconds", 1.10),
                        ("compile_median_seconds", 1.20),
                        ("cold_median_seconds", 1.20)):
        before, after = b[name], c[name]
        budget = max(limit * before, before + (0.05e-3 if name == "warm_median_of_process_medians_seconds" else 0.05))
        values[name] = {"baseline": before, "candidate": after,
                        "budget": budget, "passed": after <= budget}
    return {"baseline": str(baseline), "candidate": str(candidate),
            "metadata": {k: b.get(k) for k in fields},
            "rules": values, "all_passed": all(v["passed"] for v in values.values())}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_stage3.py BASELINE CANDIDATE")
    print(json.dumps(compare(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
