"""Pure byte/schema and workflow-shape checks, not GitHub enforcement tests."""

from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
from urllib.request import Request
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_artifact", ROOT / "scripts/release_artifact.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_literal_workflow_x64_smoke():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    smoke = workflow.split("- name: Fresh x64-off source smoke", 1)[1]
    code = re.search(r"<<'PY'\n(.*?)\n          PY", smoke, re.DOTALL).group(1)
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0"}
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def record():
    now = datetime.now(timezone.utc)
    files = {"mapprojax-0.3.0-py3-none-any.whl": b"wheel",
             "mapprojax-0.3.0.tar.gz": b"source"}
    manifest = {"schema_version": 1, "project": "mapprojax", "version": "0.3.0",
                "artifacts": {name: {"size": len(data), "sha256": release.sha256(data)}
                              for name, data in files.items()}}
    inputs = [["pyproject.toml", "100644", "a" * 64]]
    result = {"schema_version": 1, "repository": "owner/mapprojax", "version": "0.3.0",
              "producer_sha": "b" * 40, "input_sha256": release.sha256(release.canonical(inputs)),
              "run_id": 1, "run_attempt": 1, "artifact_id": 2,
              "artifact_name": "release-original", "artifact_digest": None,
              "created_at": (now - timedelta(hours=1)).isoformat(),
              "expires_at": (now + timedelta(days=89)).isoformat(),
              "expired": False, "deleted": False, "manifest": manifest}
    return result, files, inputs


def fence(value):
    return "```stage5-artifact-json\n" + json.dumps(value) + "\n```"


def make_bundle(path):
    value, files, inputs = record()
    (path / "dist").mkdir(parents=True)
    for name, data in files.items():
        (path / "dist" / name).write_bytes(data)
    (path / "manifest.json").write_text(json.dumps(value["manifest"]))
    (path / "inputs.json").write_text(json.dumps({"files": inputs, "sha256": value["input_sha256"]}))
    return value


def test_record_requires_latest_fixed_complete_fence():
    value, _, _ = record()
    assert release.parse_record("prose\n" + fence(value) + "\n") == value
    assert release.parse_record(fence(value) + "\n" + fence(value)) == value
    for text in ("", "```json\n{}\n```", fence(value) + "\n```stage5-artifact-json\n{}"):
        with pytest.raises(ValueError):
            release.parse_record(text)
    for field in release.RECORD_KEYS:
        incomplete = dict(value)
        del incomplete[field]
        with pytest.raises(ValueError):
            release.parse_record(fence(incomplete))


def selection_checkout(tmp_path, monkeypatch, value, text=None, producer_text=None):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.3.0"\n')
    if text is not None:
        closure = tmp_path / release.CLOSURE
        closure.parent.mkdir(parents=True)
        closure.write_text(text)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/mapprojax")
    monkeypatch.setattr(release, "inputs", lambda root, revision=None:
                        {"sha256": value["input_sha256"]})

    def git(root, *args):
        if args[0] == "ls-tree":
            return release.CLOSURE.encode() if producer_text is not None else b""
        if args == ("show", "HEAD:" + release.CLOSURE):
            return text.encode()
        if args == ("show", value["producer_sha"] + ":" + release.CLOSURE):
            return producer_text.encode()
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(release, "git", git)


def test_initial_produce_and_ordinary_reuse(tmp_path, monkeypatch):
    value, _, _ = record()
    selection_checkout(tmp_path, monkeypatch, value)
    result = release.select(tmp_path)
    assert result["mode"] == "produce" and result["reason"] == "closure-absent"
    closure = tmp_path / release.CLOSURE
    closure.parent.mkdir(parents=True)
    closure.write_text(fence(value))
    assert release.select(tmp_path) == {"mode": "reuse", **value}


@pytest.mark.parametrize("historical_state", ["valid", "expired", "deleted", "digest-drift"])
def test_explicit_pending_candidate_allows_new_producer(tmp_path, monkeypatch, historical_state):
    value, _, _ = record()
    if historical_state == "expired":
        value["expired"] = True
        value["expires_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    elif historical_state == "deleted":
        value["deleted"] = True
    elif historical_state == "digest-drift":
        value["input_sha256"] = "c" * 64
    text = fence(value) + "\n" + release.NEW_CANDIDATE + "\n"
    selection_checkout(tmp_path, monkeypatch, value, text)
    monkeypatch.setattr(release, "inputs", lambda root, revision=None: {"sha256": "d" * 64})
    assert release.select(tmp_path) == {"mode": "produce", "input_sha256": "d" * 64,
                                        "reason": "new-candidate-request"}
    with pytest.raises(ValueError, match="pending"):
        release.parse_record(text)
    with pytest.raises(ValueError, match="pending"):
        release.select(tmp_path, "v0.3.0")


@pytest.mark.parametrize("command", ["download", "verify"])
def test_pending_candidate_never_downloads_or_verifies_old_bytes(tmp_path, monkeypatch, command):
    value, _, _ = record()
    value["expired"] = True
    selection_checkout(tmp_path, monkeypatch, value,
                       fence(value) + "\n" + release.NEW_CANDIDATE + "\n")

    def unexpected_api(path, binary=False):
        raise AssertionError("pending request must not access historical artifact")

    monkeypatch.setattr(release, "api", unexpected_api)
    with pytest.raises(ValueError, match="requires closure"):
        release.main([command, "--root", str(tmp_path)])


def test_pending_request_must_be_committed(tmp_path, monkeypatch):
    value, _, _ = record()
    selection_checkout(tmp_path, monkeypatch, value,
                       fence(value) + "\n" + release.NEW_CANDIDATE + "\n")
    monkeypatch.setattr(release, "git", lambda root, *args: fence(value).encode())
    with pytest.raises(ValueError, match="must be committed"):
        release.select(tmp_path)


def test_appended_valid_record_supersedes_expired_history_and_request(tmp_path, monkeypatch):
    fresh, _, _ = record()
    historical = {**fresh, "expired": True, "deleted": True, "input_sha256": "c" * 64}
    request = fence(historical) + "\n" + release.NEW_CANDIDATE + "\n"
    text = request + fence(fresh) + "\n"
    selection_checkout(tmp_path, monkeypatch, fresh, text, producer_text=request)
    assert release.parse_record(text) == fresh
    assert release.select(tmp_path) == {"mode": "reuse", **fresh}


def test_producer_with_existing_closure_requires_explicit_request(tmp_path, monkeypatch):
    value, _, _ = record()
    selection_checkout(tmp_path, monkeypatch, value, fence(value), producer_text=fence(value))
    with pytest.raises(ValueError, match="explicit new candidate request"):
        release.select(tmp_path)


@pytest.mark.parametrize("mutation", ["changed", "deleted", "truncated", "crlf"])
def test_renewal_rejects_changed_producer_closure_prefix(tmp_path, monkeypatch, mutation):
    value, _, _ = record()
    history = "Original historical evidence\n" + fence({**value, "expired": True})
    producer = history + "\n" + release.NEW_CANDIDATE + "\n"
    if mutation == "changed":
        prefix = producer.replace("Original", "Rewritten", 1)
    elif mutation == "deleted":
        prefix = ""
    elif mutation == "truncated":
        prefix = producer[len("Original historical evidence\n"):]
    else:
        prefix = producer.replace("\n", "\r\n")
    text = prefix + fence(value) + "\n"
    selection_checkout(tmp_path, monkeypatch, value, text, producer_text=producer)
    # Write exact bytes so text-mode newline handling cannot mask the CRLF mutation.
    (tmp_path / release.CLOSURE).write_bytes(text.encode())
    assert release.parse_record((tmp_path / release.CLOSURE).read_text()) == value
    with pytest.raises(ValueError, match="unchanged producer closure prefix"):
        release.select(tmp_path)


@pytest.mark.parametrize("newlines", ["mixed", "crlf"])
def test_committed_pending_request_preserves_crlf_bytes(tmp_path, monkeypatch, newlines):
    value, _, _ = record()
    producer = "Historical prose\r\n" + fence({**value, "expired": True})
    producer += "\n" + release.NEW_CANDIDATE + "\n"
    if newlines == "crlf":
        producer = producer.replace("\r\n", "\n").replace("\n", "\r\n")
    selection_checkout(tmp_path, monkeypatch, value, producer, producer_text=producer)
    (tmp_path / release.CLOSURE).write_bytes(producer.encode())
    assert release.select(tmp_path)["mode"] == "produce"


@pytest.mark.parametrize("newlines", ["mixed", "crlf"])
def test_renewal_preserves_exact_historical_crlf_bytes(tmp_path, monkeypatch, newlines):
    value, _, _ = record()
    producer = "Historical prose\r\n" + fence({**value, "expired": True})
    producer += "\n" + release.NEW_CANDIDATE + "\n"
    if newlines == "crlf":
        producer = producer.replace("\r\n", "\n").replace("\n", "\r\n")
    text = producer + fence(value) + "\n"
    selection_checkout(tmp_path, monkeypatch, value, text, producer_text=producer)
    (tmp_path / release.CLOSURE).write_bytes(text.encode())
    assert release.select(tmp_path) == {"mode": "reuse", **value}


@pytest.mark.parametrize("latest", ["{}", "not json", "expired", "incomplete", "bad-opening"])
def test_invalid_latest_never_falls_back_to_valid_history(tmp_path, monkeypatch, latest):
    value, _, _ = record()
    historical = fence(value) + "\n" + release.NEW_CANDIDATE + "\n"
    if latest == "expired":
        appended = fence({**value, "expired": True})
    elif latest == "incomplete":
        appended = "```stage5-artifact-json\n{}"
    elif latest == "bad-opening":
        appended = "```stage5-artifact-json extra\n{}\n```"
    else:
        appended = "```stage5-artifact-json\n" + latest + "\n```"
    text = historical + appended
    selection_checkout(tmp_path, monkeypatch, value, text)
    with pytest.raises(ValueError):
        release.select(tmp_path)


@pytest.mark.parametrize("state", ["expired", "deleted", "digest-drift"])
def test_ordinary_record_without_request_fails_closed(tmp_path, monkeypatch, state):
    value, _, _ = record()
    if state != "digest-drift":
        value[state] = True
    selection_checkout(tmp_path, monkeypatch, value, fence(value))
    if state == "digest-drift":
        monkeypatch.setattr(release, "inputs", lambda root, revision=None: {"sha256": "c" * 64})
    with pytest.raises(ValueError):
        release.select(tmp_path)


@pytest.mark.parametrize("marker", ["prefix <!-- stage5-new-candidate -->",
                                    " <!-- stage5-new-candidate -->",
                                    "<!-- stage5-new-candidate --> suffix"])
def test_candidate_marker_requires_whole_line_after_latest_fence(marker):
    value, _, _ = record()
    assert not release.pending_candidate(fence(value) + "\n" + marker + "\n")
    assert not release.pending_candidate(release.NEW_CANDIDATE + "\n" + fence(value))


@pytest.mark.parametrize("field,bad", [
    ("artifact_id", 0), ("run_id", "1"), ("run_attempt", True),
    ("producer_sha", "main"), ("input_sha256", "abc"),
    ("artifact_name", "latest"), ("artifact_digest", "a" * 64),
    ("expired", True), ("deleted", True),
])
def test_record_rejects_bad_id_hash_or_state(field, bad):
    value, _, _ = record()
    value[field] = bad
    with pytest.raises(ValueError):
        release.parse_record(fence(value))


def test_expiry_is_actual_bounded_and_timezone_aware():
    value, _, _ = record()
    now = datetime.now(timezone.utc)
    release.check_expiry(value, now)
    for field, bad in (("expires_at", (now - timedelta(seconds=1)).isoformat()),
                       ("expires_at", (now + timedelta(days=91)).isoformat()),
                       ("created_at", (now + timedelta(seconds=1)).isoformat()),
                       ("expires_at", "2099-01-01T00:00:00")):
        with pytest.raises(ValueError):
            release.check_expiry({**value, field: bad}, now)
    for field in ("expires_at", "created_at", "expired"):
        missing = dict(value)
        del missing[field]
        with pytest.raises((KeyError, ValueError)):
            release.check_expiry(missing, now)


def test_bundle_exact_file_hashes_and_input_binding(tmp_path):
    value = make_bundle(tmp_path)
    release.verify_bundle(tmp_path, value)
    wheel = next((tmp_path / "dist").glob("*.whl"))
    wheel.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash/size"):
        release.verify_bundle(tmp_path, value)
    wheel.unlink()
    with pytest.raises(ValueError, match="missing or unexpected"):
        release.verify_bundle(tmp_path, value)


def test_bundle_rejects_extra_files_and_changed_manifests(tmp_path):
    value = make_bundle(tmp_path)
    extra = tmp_path / "dist" / "log.txt"
    extra.write_text("not a distribution")
    with pytest.raises(ValueError):
        release.verify_bundle(tmp_path, value)
    extra.unlink()
    (tmp_path / "inputs.json").write_text(json.dumps({"files": [], "sha256": value["input_sha256"]}))
    with pytest.raises(ValueError, match="input manifest"):
        release.verify_bundle(tmp_path, value)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        release.verify_bundle(tmp_path, value)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "dist/../../outside", "dist\\outside", "logs.txt"])
def test_zip_rejects_unsafe_or_unexpected_paths(tmp_path, name):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, "bad")
    with pytest.raises(ValueError):
        release.extract_bundle(stream.getvalue(), tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()


def test_zip_rejects_symlinks(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        info = zipfile.ZipInfo("dist/mapprojax-0.3.0.tar.gz")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "../../outside")
    with pytest.raises(ValueError, match="unsafe"):
        release.extract_bundle(stream.getvalue(), tmp_path / "fresh")


@pytest.mark.parametrize("field", ["size", "sha256"])
def test_manifest_requires_individual_file_identity(field):
    value, _, _ = record()
    identity = next(iter(value["manifest"]["artifacts"].values()))
    del identity[field]
    with pytest.raises(ValueError, match="file identity"):
        release.validate_manifest(value["manifest"], "0.3.0")


@pytest.mark.parametrize("version", [True, 1.0, False, 2, "1"])
def test_manifest_schema_requires_exact_integer_one(version):
    value, _, _ = record()
    value["manifest"]["schema_version"] = version
    with pytest.raises(ValueError, match="version mismatch"):
        release.validate_manifest(value["manifest"], "0.3.0")


@pytest.mark.parametrize("original,replacement", [
    ("mapprojax-0.3.0.tar.gz", "mapprojax-0.3.0.extra.tar.gz"),
    ("mapprojax-0.3.0.tar.gz", "mapprojax-0.3.0-other.tar.gz"),
    ("mapprojax-0.3.0-py3-none-any.whl", "mapprojax-0.3.0-py3-none.whl"),
    ("mapprojax-0.3.0-py3-none-any.whl", "mapprojax-0.3.0-1-py3-none-any.whl"),
    ("mapprojax-0.3.0-py3-none-any.whl", "mapprojax-0.3.1-py3-none-any.whl"),
    ("mapprojax-0.3.0-py3-none-any.whl", "mapprojax-0.3.0-py3-none-an:y.whl"),
    ("mapprojax-0.3.0-py3-none-any.whl", "mapprojax-0.3.0-py3-none-an\x00y.whl"),
])
def test_manifest_distribution_names_match_installed_validator(original, replacement):
    value, _, _ = record()
    files = value["manifest"]["artifacts"]
    files[replacement] = files.pop(original)
    with pytest.raises(ValueError, match="distribution name"):
        release.validate_manifest(value["manifest"], "0.3.0")


@pytest.mark.parametrize("prevent_self_review", [False, True])
@pytest.mark.parametrize("login", ["leejjoon", "LeeJjoon"])
def test_protection_accepts_actual_operator_user(monkeypatch, prevent_self_review, login):
    settings = {"protection_rules": [{"type": "required_reviewers",
                "prevent_self_review": prevent_self_review,
                "reviewers": [{"type": "User", "reviewer": {"login": login}}]}]}
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/mapprojax")
    paths = []

    def api(path):
        paths.append(path)
        return settings

    monkeypatch.setattr(release, "api", api)
    release.protection()
    assert paths == ["repos/owner/mapprojax/environments/pypi"]


@pytest.mark.parametrize("prevent_self_review", [False, True])
@pytest.mark.parametrize("settings", [
    {}, {"protection_rules": []},
    {"protection_rules": [{"type": "wait_timer"}]},
    {"protection_rules": [{"type": "required_reviewers"}]},
    {"protection_rules": [{"type": "required_reviewers", "reviewers": []}]},
    {"protection_rules": [{"type": "required_reviewers", "reviewers": [
        {"type": "User", "reviewer": {"login": "another-user"}}]}]},
    {"protection_rules": [{"type": "required_reviewers", "reviewers": [
        {"type": "Team", "reviewer": {"login": "leejjoon"}}]}]},
    {"protection_rules": [{"type": "required_reviewers", "reviewers": [
        {"type": "User", "reviewer": {}}]}]},
])
def test_protection_rejects_missing_or_wrong_operator(monkeypatch, settings, prevent_self_review):
    settings = json.loads(json.dumps(settings))
    for rule in settings.get("protection_rules", []):
        rule["prevent_self_review"] = prevent_self_review
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/mapprojax")
    monkeypatch.setattr(release, "api", lambda path: settings)
    with pytest.raises(ValueError, match="User reviewer leejjoon"):
        release.protection()


@pytest.mark.parametrize("target", ["https://storage.example/artifact",
                                   "https://api.github.com:8443/artifact",
                                   "https://api.github.com/artifact",
                                   "https://api.github.com:443/artifact"])
def test_every_https_redirect_strips_authorization(target):
    request = Request("https://api.github.com/artifact", headers={
        "Authorization": "Bearer test-credential", "Accept": "application/octet-stream"})
    redirected = release.TokenSafeRedirect().redirect_request(
        request, None, 302, "Found", {}, target)
    assert redirected.full_url == target
    assert not any(name.lower() == "authorization" for name in redirected.headers)
    assert redirected.get_header("Accept") == "application/octet-stream"
    assert request.get_header("Authorization") == "Bearer test-credential"


@pytest.mark.parametrize("target", ["http://api.github.com/artifact",
                                   "http://storage.example/artifact"])
def test_redirect_rejects_http_boundary(target):
    request = Request("https://api.github.com/artifact", headers={"Authorization": "Bearer test"})
    with pytest.raises(ValueError, match="insecure API redirect"):
        release.TokenSafeRedirect().redirect_request(request, None, 302, "Found", {}, target)


def test_tag_requires_exact_version_and_closure(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.3.0"\n')
    with pytest.raises(ValueError, match="exactly match"):
        release.select(tmp_path, "v0.3.1")
    with pytest.raises(ValueError, match="requires closure"):
        release.select(tmp_path, "v0.3.0")


def test_original_producer_checks_both_distributions_with_pinned_twine():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    producer = ci.split("- name: Single candidate build from clean archive", 1)[1]
    producer = producer.split("- name: Reuse original artifact", 1)[0]
    check = ('"$GITHUB_WORKSPACE/.venv/bin/python" -m twine check '
             '"$RUNNER_TEMP/bundle/dist/"*.whl "$RUNNER_TEMP/bundle/dist/"*.tar.gz')
    assert check in producer
    assert producer.index("-m build") < producer.index(check) < producer.index(" manifest ")
    assert ci.index(check) < ci.index("name: Persist immutable original distributions")


def test_workflow_event_permissions_pins_and_inert_publication_shape():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    publish = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "on:\n  push:\n    branches: [rocm-extra, main]\n  pull_request:" in ci
    assert "github.event.pull_request.head.sha || github.sha" in ci
    assert "on:\n  push:\n    tags: ['v*']\n" in publish
    assert "pull_request" not in publish and "workflow_dispatch" not in publish
    for text in (ci, publish):
        assert text.count("runs-on: ubuntu-24.04") == 2
        assert "runs-on: ubuntu-latest" not in text
        assert "permissions:\n  contents: read\n  actions: read\n" in text
        assert "env:\n  JAX_PLATFORMS: cpu\n" in text
        uses = re.findall(r"uses: ([^\s]+)", text)
        assert uses and all(re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", action) for action in uses)
        assert "password:" not in text and "secrets." not in text
        assert "retention-days: 90" in text
        assert "--constraints ci/latest.txt --build-constraints ci/build.txt --backend cpu" in text
    assert ci.count("-m build --no-isolation --sdist --wheel") == 1
    assert "if: steps.selection.outputs.mode == 'produce'" in ci
    assert "if: steps.selection.outputs.mode == 'reuse'" in ci
    assert "--artifact-id \"$ARTIFACT_ID\"" in ci
    assert "--destination \"$RUNNER_TEMP/bundle\"" in ci
    assert "python: '3.11.15'\n            constraints: ci/minimum.txt" in ci
    assert "python: '3.14.4'\n            constraints: ci/latest.txt" in ci
    assert "freethreaded: false" in ci and "JAX_ENABLE_X64: '0'" in ci
    assert "namespace['_F64_WALKER']" in ci and "src.unproj_offset" in ci and "src.proj_offset" in ci
    assert "benchmark_accuracy.py --json" in ci
    assert "-m build" not in publish
    assert publish.count("id-token: write") == 1 and "id-token:" not in ci
    assert "needs: validate" in publish and "environment: pypi" in publish
    assert "--protection" in publish
    assert re.search(r"name: Upload original bytes with trusted publishing\n        if: success\(\)\n        uses: pypa/gh-action-pypi-publish@[0-9a-f]{40}", publish)
    # The x64 flag belongs only to the dedicated smoke step, never job/global env.
    assert ci.count("JAX_ENABLE_X64:") == 1
    assert "JAX_ENABLE_X64:" not in publish
