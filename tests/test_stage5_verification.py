"""Pure artifact/command tests; never recursively run the installed-suite gate."""

import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_installed_artifacts.py"
SPEC = importlib.util.spec_from_file_location("check_installed_artifacts", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def distributions(tmp_path):
    directory = tmp_path / "dist"
    directory.mkdir()
    (directory / "mapprojax-0.3.0-py3-none-any.whl").write_bytes(b"wheel fixture")
    (directory / "mapprojax-0.3.0.tar.gz").write_bytes(b"source fixture")
    return directory


def test_manifest_roundtrip_and_cli(distributions, tmp_path):
    output = tmp_path / "manifest.json"
    RUNNER.main(["manifest", "--dist-dir", str(distributions), "--output", str(output)])
    manifest = json.loads(output.read_text())
    assert manifest["version"] == "0.3.0"
    assert manifest["artifacts"]["mapprojax-0.3.0.tar.gz"] == {
        "size": len(b"source fixture"), "sha256": hashlib.sha256(b"source fixture").hexdigest()}
    RUNNER.main(["verify", "--dist-dir", str(distributions), "--manifest", str(output)])
    assert len(RUNNER.verify_manifest(distributions, manifest)) == 2


@pytest.mark.parametrize("mutation", ["hash", "size", "missing", "extra", "version", "field"])
def test_manifest_rejects_changed_inventory(distributions, mutation):
    manifest = RUNNER.make_manifest(distributions)
    wheel = distributions / "mapprojax-0.3.0-py3-none-any.whl"
    if mutation == "hash":
        wheel.write_bytes(b"wrong fixture")
    elif mutation == "size":
        manifest["artifacts"][wheel.name]["size"] += 1
    elif mutation == "missing":
        wheel.unlink()
    elif mutation == "extra":
        (distributions / "unexpected.txt").write_text("unexpected")
    elif mutation == "version":
        manifest["version"] = "99"
    else:
        manifest["unknown"] = True
    with pytest.raises(ValueError):
        RUNNER.verify_manifest(distributions, manifest)


@pytest.mark.parametrize("name", ["../evil", "/tmp/evil", "a/../b", "a//b", "./a", "a\\b", "C:evil", ""])
def test_unsafe_paths_rejected(name):
    with pytest.raises(ValueError, match="unsafe path"):
        RUNNER.safe_name(name)


def test_manifest_rejects_nested_path_and_symlink(distributions, tmp_path):
    manifest = RUNNER.make_manifest(distributions)
    manifest["artifacts"]["nested/wheel.whl"] = {"size": 0, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="basename"):
        RUNNER.verify_manifest(distributions, manifest)
    wheel = distributions / "mapprojax-0.3.0-py3-none-any.whl"
    wheel.unlink()
    wheel.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError):
        RUNNER.make_manifest(distributions)


def make_wheel(path, content=b"package", record_mutation=None, reverse=False):
    members = {"mapprojax/__init__.py": content,
               "mapprojax-0.3.0.dist-info/METADATA": b"Name: mapprojax\nVersion: 0.3.0\n"}
    rows = []
    for name, data in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([name, "sha256=" + digest, str(len(data))])
    record = "mapprojax-0.3.0.dist-info/RECORD"
    rows.append([record, "", ""])
    if record_mutation:
        record_mutation(rows, members)
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n" if reverse else "\r\n").writerows(
        reversed(rows) if reverse else rows)
    members[record] = stream.getvalue().encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED if reverse else zipfile.ZIP_STORED) as archive:
        for name, data in reversed(list(members.items())) if reverse else members.items():
            info = zipfile.ZipInfo(name, date_time=(2025 if reverse else 2024, 1, 1, 0, 0, 0))
            archive.writestr(info, data)
    return path


def test_wheel_comparison_ignores_container_metadata_and_record_order(tmp_path):
    direct = make_wheel(tmp_path / "direct.whl")
    derived = make_wheel(tmp_path / "derived.whl", reverse=True)
    RUNNER.compare_wheels(direct, derived)
    RUNNER.main(["compare-wheels", str(direct), str(derived)])


def test_wheel_comparison_rejects_valid_but_changed_content(tmp_path):
    direct = make_wheel(tmp_path / "direct.whl")
    derived = make_wheel(tmp_path / "derived.whl", content=b"different")
    with pytest.raises(ValueError, match="differ"):
        RUNNER.compare_wheels(direct, derived)


@pytest.mark.parametrize("mutation", ["hash", "size", "missing", "duplicate", "self", "unlisted", "unsafe"])
def test_record_validation(tmp_path, mutation):
    def mutate(rows, members):
        if mutation == "hash":
            rows[0][1] = "sha256=" + "A" * 43
        elif mutation == "size":
            rows[0][2] = "999"
        elif mutation == "missing":
            del members[rows[0][0]]
        elif mutation == "duplicate":
            rows.append(rows[0])
        elif mutation == "self":
            rows[-1][2] = "0"
        elif mutation == "unlisted":
            members["unlisted.txt"] = b"data"
        else:
            members["../escape"] = b"bad"
    path = make_wheel(tmp_path / "bad.whl", record_mutation=mutate)
    with pytest.raises(ValueError):
        RUNNER.wheel_contents(path)


def test_metadata_license_and_urls():
    project = {"name": "mapprojax", "version": "0.3.0", "description": "Kernel",
               "requires-python": ">=3.11", "urls": {"Repository": "https://example.org/repo"}}
    metadata = (b"Name: mapprojax\nVersion: 0.3.0\nSummary: Kernel\nRequires-Python: >=3.11\n"
                b"License-Expression: MIT\nLicense-File: LICENSE\n"
                b"Project-URL: Repository, https://example.org/repo\n\n")
    license_text = b"complete license fixture\n"
    assert RUNNER.validate_metadata(metadata, license_text, project, license_text) == "0.3.0"
    for bad in (metadata.replace(b"0.3.0", b"0.2.0"),
                metadata.replace(b"MIT", b"BSD"),
                metadata.replace(b"LICENSE", b"OTHER"),
                metadata.replace(b"https://example.org/repo", b"https://wrong.org")):
        with pytest.raises(ValueError):
            RUNNER.validate_metadata(bad, license_text, project, license_text)
    with pytest.raises(ValueError, match="LICENSE"):
        RUNNER.validate_metadata(metadata, b"truncated", project, license_text)


def test_tar_rejects_traversal_links_and_duplicates(tmp_path):
    for index, kind in enumerate(("traversal", "symlink", "duplicate")):
        path = tmp_path / f"{index}.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            info = tarfile.TarInfo("../escape" if kind == "traversal" else "root/file")
            if kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                archive.addfile(info)
            else:
                info.size = 4
                archive.addfile(info, io.BytesIO(b"data"))
                if kind == "duplicate":
                    archive.addfile(info, io.BytesIO(b"data"))
        with pytest.raises(ValueError):
            RUNNER.tar_contents(path)


def test_tar_extracts_only_validated_regular_files(tmp_path):
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("mapprojax-0.3.0/tests/example.txt")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"data"))
    target = RUNNER.extract_source(path, tmp_path / "source")
    assert (target / "tests/example.txt").read_bytes() == b"data"


def test_clean_environment_removes_source_and_interpreter_overrides(monkeypatch, tmp_path):
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTHONWARNINGS",
                 "PYTEST_ADDOPTS", "VIRTUAL_ENV", "JAX_ENABLE_X64", "JAX_DISABLE_JIT",
                 "MAPPROJAX_EXPECTED_SOURCE_ROOT", "MAPPROJAX_EXPECTED_INSTALL_ROOT"):
        monkeypatch.setenv(name, "contaminated")
    env = RUNNER.clean_environment("rocm", tmp_path)
    assert not {"PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTHONWARNINGS",
                "PYTEST_ADDOPTS", "VIRTUAL_ENV", "JAX_ENABLE_X64", "JAX_DISABLE_JIT",
                "MAPPROJAX_EXPECTED_SOURCE_ROOT"} & env.keys()
    assert env["MAPPROJAX_EXPECTED_INSTALL_ROOT"] == str(tmp_path)
    assert env["JAX_PLATFORMS"] == "rocm"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_command_has_bounded_timeout_exact_cwd_env_and_failure_semantics(monkeypatch, tmp_path):
    calls = []
    code = 0

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(RUNNER.subprocess, "run", fake_run)
    env = {"PATH": "/test"}
    log = tmp_path / "command.log"
    RUNNER.run_command(["python", "-m", "pytest"], tmp_path, env, log)
    command, kwargs = calls.pop()
    assert command == ["python", "-m", "pytest"]
    assert kwargs["cwd"] == tmp_path and kwargs["env"] is env
    assert kwargs["timeout"] == RUNNER.TIMEOUT and kwargs["timeout"] > 0
    assert kwargs["stderr"] == subprocess.STDOUT
    assert "shell" not in kwargs
    code = 3
    with pytest.raises(RuntimeError, match="exit 3"):
        RUNNER.run_command(["python"], tmp_path, env, log)
    assert RUNNER.run_command(["python"], tmp_path, env, log, expect_failure=True) == 3
    code = 0
    with pytest.raises(RuntimeError, match="exit 0"):
        RUNNER.run_command(["python"], tmp_path, env, log, expect_failure=True)
    with pytest.raises(ValueError):
        RUNNER.run_command([], tmp_path, env, log)


@pytest.mark.parametrize("text,valid", [
    ("build==1.3.0\nsetuptools==80.9.0\nwheel==0.45.1\n", True),
    ("build==1.3.0\nsetuptools==76.0.0\nwheel==0.45.1\n", False),
    ("build==1.3.0\nsetuptools>=77\nwheel==0.45.1\n", False),
    ("setuptools==80.9.0\nwheel==0.45.1\n", False),
])
def test_backend_build_tools_must_be_effectively_pinned(tmp_path, text, valid):
    path = tmp_path / "build.txt"
    path.write_text(text)
    if valid:
        RUNNER.validate_build_constraints(path)
    else:
        with pytest.raises(ValueError):
            RUNNER.validate_build_constraints(path)


def seed_payload(root):
    for name in (*RUNNER.SUPPORT_FILES, *("tests/" + n for n in RUNNER.REQUIRED_TESTS)):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("payload fixture\n")
    (root / "tests/data").mkdir()
    (root / "tests/data/fixture.json").write_text("{}")


def test_payload_inventory_is_external_exact_and_has_no_source_reachback(tmp_path):
    root = tmp_path / "root"
    seed_payload(root)
    (root / "src/mapprojax").mkdir(parents=True)
    (root / "src/mapprojax/__init__.py").write_text("forbidden")
    (root / ".git").mkdir()
    (root / "scratch.txt").write_text("forbidden")
    (root / "tests/__pycache__").mkdir()
    (root / "tests/__pycache__/junk.pyc").write_bytes(b"cache")
    destination = tmp_path / "payload"
    inventory = RUNNER.copy_payload(root, destination)
    assert set(inventory) == set(RUNNER.payload_inventory(root))
    assert "tests/data/fixture.json" in inventory
    assert "scripts/check_installed_artifacts.py" in inventory
    assert not (destination / "src").exists()
    assert not (destination / ".git").exists()
    assert not (destination / "scratch.txt").exists()
    assert not (destination / "tests/__pycache__").exists()
    for name, identity in inventory.items():
        assert RUNNER.file_identity(destination / name) == identity
    (root / "README.md").unlink()
    with pytest.raises(ValueError, match="support"):
        RUNNER.payload_inventory(root)


def test_payload_rejects_link_and_missing_required_test(tmp_path):
    root = tmp_path / "root"
    seed_payload(root)
    test = root / "tests" / RUNNER.REQUIRED_TESTS[0]
    test.unlink()
    with pytest.raises(ValueError, match="required test"):
        RUNNER.payload_inventory(root)
    test.symlink_to(root / "README.md")
    with pytest.raises(ValueError, match="symlink"):
        RUNNER.payload_inventory(root)
