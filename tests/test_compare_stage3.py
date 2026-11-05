import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_COMPARATOR_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "compare_stage3.py"
_SPEC = importlib.util.spec_from_file_location("compare_stage3", _COMPARATOR_PATH)
_COMPARATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMPARATOR)
RULES = _COMPARATOR.RULES
compare = _COMPARATOR.compare


HISTORICAL_FIELDS = {
    "backend_requested": "cpu",
    "mode": "smoke",
    "workload": "pair",
    "call": "pix_to_pix",
    "size": 4,
    "order": 3,
    "inverse_steps": 3,
    "batch_shape": None,
    "pair_precision": "f32",
    "devices": "cpu:0",
}


def _artifact(path, role, *, current=True, include_lowering=True, warm=1.0):
    measurement = {**HISTORICAL_FIELDS, "implementation": role}
    if current:
        measurement["workload"] = "pixel_map"
        measurement["call"] = (
            "pix_to_pix+finite_half_open_bounds+zero_selection"
            if role == "baseline" else "pixel_map"
        )
        measurement.update({
            "input_dtype": "float32",
            "kernel_precision": "f32",
            "sip": {"source": {"order": 3, "inverse_steps": 3}, "target": None},
            "bounds_shape": [1201, 1201],
            "bounds": "half_open_pixel_centres",
            "direction": {"source": "source", "target": "target"},
            "semantic_workload": "bounded_pair_pixel_mapping",
            "semantic_call": measurement["call"],
            "backend_actual": "cpu",
            "source_identity": {"mapprojax_sha256": "a", "benchmark_sha256": "b"},
        })
    for name, _, _ in RULES:
        if name == "lowering_median_seconds" and not include_lowering:
            continue
        measurement[name] = warm
    path.write_text(json.dumps({
        "environment": {"python": "test-python", "backend": "cpu"},
        "measurement": measurement,
    }))
    return path


def test_accepts_current_and_historical_role_pairs(tmp_path):
    current_baseline = _artifact(tmp_path / "current-baseline.json", "baseline")
    current_new = _artifact(tmp_path / "current-new.json", "new")
    candidate_data = json.loads(current_new.read_text())
    candidate_data["measurement"]["source_identity"]["mapprojax_sha256"] = "different"
    current_new.write_text(json.dumps(candidate_data))
    historical_legacy = _artifact(
        tmp_path / "historical-legacy.json", "legacy", current=False
    )
    historical_new = _artifact(
        tmp_path / "historical-new.json", "new", current=False
    )

    result = compare(current_baseline, current_new)
    assert result["all_passed"]
    assert result["metadata"]["provenance"]["candidate"]["mapprojax_sha256"] == "different"
    assert compare(historical_legacy, historical_new)["all_passed"]


def test_rejects_current_metadata_mismatch(tmp_path):
    baseline = _artifact(tmp_path / "baseline.json", "baseline")
    candidate = _artifact(tmp_path / "candidate.json", "new")
    data = json.loads(candidate.read_text())
    data["measurement"]["direction"]["target"] = "different-target"
    candidate.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="metadata differ"):
        compare(baseline, candidate)


def test_missing_lowering_is_only_allowed_for_historical_pair(tmp_path):
    legacy = _artifact(
        tmp_path / "legacy.json", "legacy", current=False, include_lowering=False
    )
    old_new = _artifact(
        tmp_path / "old-new.json", "new", current=False, include_lowering=False
    )
    assert "lowering_median_seconds" not in compare(legacy, old_new)["rules"]

    baseline = _artifact(
        tmp_path / "baseline.json", "baseline", include_lowering=False
    )
    new = _artifact(tmp_path / "new.json", "new", include_lowering=False)
    with pytest.raises(ValueError, match="missing lowering_median_seconds"):
        compare(baseline, new)


def test_over_budget_timing_fails_without_changing_limits(tmp_path):
    baseline = _artifact(tmp_path / "baseline.json", "baseline")
    candidate = _artifact(tmp_path / "candidate.json", "new", warm=1.2)

    result = compare(baseline, candidate)
    warm_rule = result["rules"]["warm_median_of_process_medians_seconds"]
    assert not result["all_passed"]
    assert warm_rule["budget"] == pytest.approx(1.1)
    assert not warm_rule["passed"]


def test_comparator_cli_reports_mismatch_and_budget_exit_codes(tmp_path):
    baseline = _artifact(tmp_path / "baseline.json", "baseline")
    candidate = _artifact(tmp_path / "candidate.json", "new", warm=1.2)
    mismatch = json.loads(candidate.read_text())
    mismatch["measurement"]["direction"]["target"] = "other"
    candidate.write_text(json.dumps(mismatch))
    command = [sys.executable, str(_COMPARATOR_PATH), str(baseline), str(candidate)]
    mismatch_result = subprocess.run(command, capture_output=True, text=True)
    assert mismatch_result.returncode == 2
    assert "unmatched pair" in mismatch_result.stderr

    candidate = _artifact(tmp_path / "over-budget.json", "new", warm=1.2)
    budget_result = subprocess.run(
        [sys.executable, str(_COMPARATOR_PATH), str(baseline), str(candidate)],
        capture_output=True,
        text=True,
    )
    assert budget_result.returncode == 1
    assert "FAILED:" in budget_result.stderr
