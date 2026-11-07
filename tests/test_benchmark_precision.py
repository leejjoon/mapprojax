"""Deterministic harness contracts, without hardware-dependent speed limits."""

import importlib.util
import json
from pathlib import Path
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_precision.py"


def load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_precision", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bp = load_benchmark()


@pytest.fixture(autouse=True)
def restore_x64():
    previous = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("enabled", [False, True])
def test_import_does_not_change_x64_or_filter_warnings(enabled):
    jax.config.update("jax_enable_x64", enabled)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        before = list(warnings.filters)
        load_benchmark()
        assert jax.config.jax_enable_x64 == enabled
        assert warnings.filters == before


def test_measurement_waits_on_every_leaf_and_uses_warm_median(monkeypatch):
    args = (object(), object())
    outputs = []
    events = []

    def function(*received):
        assert received == args
        output = {"x": object(), "nested": (object(), {"y": object()})}
        outputs.append(output)
        events.append(("call", output))
        return output

    def wait(tree):
        events.append(("wait", tree))
        return tree

    ticks = iter([1.0, 1.004, 2.0, 2.002, 3.0, 3.009])
    monkeypatch.setattr(bp.jax, "block_until_ready", wait)
    monkeypatch.setattr(bp.time, "perf_counter", lambda: next(ticks))
    result, output = bp.measure_warm(function, args, warmups=2, repeats=3)
    assert events[0] == ("wait", args)
    assert len(outputs) == 5
    for index, returned in enumerate(outputs):
        assert events[1 + 2 * index] == ("call", returned)
        assert events[2 + 2 * index] == ("wait", returned)
    assert output is outputs[-1]
    assert result["samples_ms"] == pytest.approx([4.0, 2.0, 9.0])
    assert result["median_ms"] == pytest.approx(4.0)
    assert result["warmups"] == 2 and result["repeats"] == 3


@pytest.mark.parametrize("warmups,repeats", [(0, 1), (1, 0), (-1, 1)])
def test_reject_empty_measurements(warmups, repeats):
    with pytest.raises(ValueError, match="positive"):
        bp.measure_warm(lambda: (), (), warmups, repeats)


def test_offset_mapping_forwards_exact_producer_scales(monkeypatch):
    src_scale, tgt_scale = object(), object()
    src_reference = np.array([45.0, 89.99999], dtype=np.float64)
    tgt_reference = np.array([45.1, 89.99998], dtype=np.float64)
    source_offset, absolute, target_offset = object(), object(), (object(), object())
    x, y = object(), object()

    class Source:
        def offset_scale(self):
            return src_scale

        def unproj_offset(self, a, b):
            assert a is x and b is y
            return source_offset

    class Target:
        def offset_scale(self):
            return tgt_scale

        def proj_offset(self, a, b):
            assert a is target_offset[0] and b is target_offset[1]
            return "mapped"

    def to_absolute(offset, *, crval, offset_scale):
        assert offset is source_offset
        assert crval is src_reference and offset_scale is src_scale
        return absolute

    def from_absolute(value, *, crval, offset_scale):
        assert value is absolute
        assert crval is tgt_reference and offset_scale is tgt_scale
        return target_offset

    monkeypatch.setattr(bp, "absolute_from_offset", to_absolute)
    monkeypatch.setattr(bp, "offset_from_absolute", from_absolute)
    assert bp.offset_mapping(Source(), Target(), x, y, src_reference, tgt_reference) == "mapped"


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_x64_off_offset_types_warnings_and_error_restoration(monkeypatch, initial, fail):
    jax.config.update("jax_enable_x64", initial)
    original = bp.offset_mapping

    def mapping(src, tgt, x, y, src_ref, tgt_ref):
        assert not jax.config.jax_enable_x64
        assert src.precision == tgt.precision == "f32"
        assert x.dtype == y.dtype == jnp.float32
        assert src_ref.dtype == tgt_ref.dtype == np.float64
        assert src.offset_scale().dtype == tgt.offset_scale().dtype == jnp.float32
        if fail:
            raise RuntimeError("injected producer failure")
        return original(src, tgt, x, y, src_ref, tgt_ref)

    monkeypatch.setattr(bp, "offset_mapping", mapping)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filters = list(warnings.filters)
        if fail:
            with pytest.raises(RuntimeError, match="injected producer failure"):
                bp.offset_f32_without_x64(np.array([100.0]), np.array([101.0]))
        else:
            output = bp.offset_f32_without_x64(np.array([100.0]), np.array([101.0]))
            assert bp.output_dtypes(output) == ["float32", "float32"]
            assert all(np.isfinite(v).all() for v in output)
        assert warnings.filters == filters
    assert jax.config.jax_enable_x64 == initial


def test_path_labels_types_and_small_accuracy():
    with bp.x64_mode(True), warnings.catch_warnings():
        warnings.simplefilter("error")
        src, tgt = bp.wcs_pair("f64")
        ms, mt = bp.wcs_pair("mixed")
        paths = bp.build_paths(src, tgt, ms, mt)
        assert list(paths) == list(bp.LABELS)
        assert list(bp.LABELS.values()) == [
            "eager JAX WCS f64", "jitted JAX WCS f64", "jitted JAX WCS mixed",
            "jitted Reprojector f32", "jitted Reprojector f64"]
        rows = bp.report_accuracy(paths, sample_count=7)
    for row in rows:
        assert row["n_samples"] == row["n_reference_finite"] == row["n_output_finite"] == row["n_overlap"] == 7
        assert np.isfinite(row["max_px"])
        expected = "float32" if row["path"] in ("wcs_jit_mixed", "rp_jit_f32", "offset_host_f32_x64_off") else "float64"
        assert row["input_dtype"] == expected
        assert row["output_dtypes"] == [expected, expected]


def test_detector_geometry_seed_and_preserved_plate_grid(monkeypatch):
    assert bp.DEFAULT_SIZES == (512, 1024, 2048, 4096, 8192)
    assert bp.DETECTOR_SIZE == 4096 and bp.DETECTOR_CRPIX == [2048.0, 2048.0]
    rng = np.random.default_rng(0)
    x, y = bp.detector_samples(13)
    np.testing.assert_array_equal(x, rng.uniform(0, 4096, 13))
    np.testing.assert_array_equal(y, rng.uniform(0, 4096, 13))
    assert bp.PLATE_CASES[:3] == tuple(("cross-field", p, 1800.0) for p in (3.6, 0.1, 0.05))
    assert bp.PLATE_CASES[3:] == tuple(("same-pointing", p, s) for p in (0.1, 0.05, 0.02) for s in (1.0, 10.0, 60.0, 300.0))
    original = bp.wcs_pair
    geometry = []

    def pair(precision, **kwargs):
        geometry.append(kwargs)
        result = original(precision, **kwargs)
        assert isinstance(result[0], bp.Tan) and isinstance(result[1], bp.Sin)
        return result

    monkeypatch.setattr(bp, "wcs_pair", pair)
    with bp.x64_mode(True):
        rows = bp.report_plate_scale_sensitivity(sample_count=5)
    assert len(rows) == 30
    assert all(row["n_overlap"] == row["n_samples"] == 5 for row in rows)
    for setup, (_, plate, separation) in zip(geometry, bp.PLATE_CASES):
        assert setup["crpix"] == [2048.0, 2048.0]
        assert setup["cd"] == [[-plate / 3600.0, 0.0], [0.0, plate / 3600.0]]
        assert setup["src"] == [180.0, 30.0]
        assert setup["tgt"] == [180.0, 30.0 + separation / 3600.0]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_outputs_and_references_are_rejected(bad):
    good = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    invalid = (np.array([bad, 2.0]), good[1])
    with pytest.raises(ValueError, match="invalid samples"):
        bp.error_stats(invalid, good)
    with pytest.raises(ValueError, match="invalid samples"):
        bp.error_stats(good, invalid)
    stats = bp.error_stats(good, good)
    assert stats["n_overlap"] == 2 and stats["max_px"] == 0.0


@pytest.mark.parametrize("initial", [False, True])
def test_main_restores_x64_on_measurement_failure(monkeypatch, initial):
    jax.config.update("jax_enable_x64", initial)
    monkeypatch.setattr(bp, "report_metadata", lambda argv: {})

    def fail(*args):
        assert jax.config.jax_enable_x64
        raise RuntimeError("timing failed")

    monkeypatch.setattr(bp, "run_timing_benchmark", fail)
    with pytest.raises(RuntimeError, match="timing failed"):
        bp.main(["--sizes", "2", "--warmups", "1", "--repeats", "1"])
    assert jax.config.jax_enable_x64 == initial


def test_small_cli_report_contract(tmp_path, monkeypatch, capsys):
    # Keep CLI coverage small while exercising every real path and plate case.
    accuracy, plates = bp.report_accuracy, bp.report_plate_scale_sensitivity
    monkeypatch.setattr(bp, "report_accuracy", lambda paths: accuracy(paths, sample_count=5))
    monkeypatch.setattr(bp, "report_plate_scale_sensitivity", lambda: plates(sample_count=5))
    json_path, markdown_path = tmp_path / "precision.json", tmp_path / "precision.md"
    jax.config.update("jax_enable_x64", False)
    bp.main(["--sizes", "2", "--warmups", "1", "--repeats", "2",
             "--json", str(json_path), "--markdown", str(markdown_path)])
    assert not jax.config.jax_enable_x64
    report = json.loads(json_path.read_text())
    assert len(report["timings"]) == 5
    assert all(len(row["samples_ms"]) == 2 for row in report["timings"])
    assert all(row["accuracy"]["n_overlap"] == 4 for row in report["timings"])
    meta = report["metadata"]
    assert len(meta["script_sha256"]) == 64
    assert meta["sampling"]["detector_shape"] == [4096, 4096]
    assert meta["reference"] == bp.REFERENCE
    assert {"revision", "devices", "dependencies", "dtypes", "units", "command"} <= meta.keys()
    text = markdown_path.read_text()
    assert "4096x4096" in text and "not independent truth" in text
    assert capsys.readouterr().out.strip() == text.strip()
