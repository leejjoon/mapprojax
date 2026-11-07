"""Select and verify the one retained release artifact (stdlib only).

Closure schema: append-only ```stage5-artifact-json fenced JSON objects in
benchmarks/results/stage5-closure.md. Only the latest fence is selected; an
invalid latest record never falls back to historical records. Keys are RECORD_KEYS below; manifest is
exactly the check_installed_artifacts.py manifest. artifact_digest is the API
container digest (null only when the API reports none). All timestamps are API
ISO-8601 timestamps; expired/deleted must be false. IDs are positive integers.
schema_version is integer 1; repository is owner/repo; version equals pyproject;
producer_sha is 40 lowercase hex; input_sha256 is 64 lowercase hex; artifact_name
is release-original; run_id/run_attempt/artifact_id are positive integers.
The operator records the JSON printed by provenance after the producer upload.
input_sha256 hashes a deterministic JSON list of [tracked path, git mode,
SHA256(blob)] for every tracked file EXCEPT the evidence-only closure file.
No other documentation, test, workflow, lock or build input is excluded.
To request a fresh fully validated producer, append the whole line
<!-- stage5-new-candidate --> AFTER the latest artifact fence, committing and
pushing only under existing operator authority. choose then emits mode=produce
and reason=new-candidate-request even if historical bytes expired or inputs
changed. download/verify/tag reject this pending state. Append a fresh complete
artifact fence after validation to supersede the marker and historical records;
then require exact closure CI/main as usual. Never rewrite historical closure.

CLI (all paths explicit):
  choose --root ROOT --output selection.json [--tag vVERSION]
  provenance --root ROOT --artifact-id ID --manifest FILE --output record.json
  download --root ROOT --destination BUNDLE [--tag vVERSION]
  verify --root ROOT --bundle BUNDLE [--tag vVERSION] [--protection]
  inputs --root ROOT --output inputs.json
GH_TOKEN and GITHUB_REPOSITORY supply read-only API access; download/verify
always query the actual artifact and workflow run, never a 'latest' selector.
A publish tag additionally requires exact remote main and prior successful
push CI on that closure SHA. --protection checks the actual required User
reviewer leejjoon (case insensitive), without imposing prevent_self_review;
it does not establish PyPI publisher configuration or operator approval.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tomllib
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

CLOSURE = "benchmarks/results/stage5-closure.md"
NEW_CANDIDATE = "<!-- stage5-new-candidate -->"
RECORD_KEYS = {"schema_version", "repository", "version", "producer_sha",
               "input_sha256", "run_id", "run_attempt", "artifact_id",
               "artifact_name", "artifact_digest", "created_at", "expires_at",
               "expired", "deleted", "manifest"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def inputs(root, revision=None):
    entries = []
    listing = (git(root, "ls-files", "--stage", "-z") if revision is None else
               git(root, "ls-tree", "-rz", "--full-tree", revision))
    for line in listing.split(b"\0"):
        if not line:
            continue
        metadata, raw_path = line.split(b"\t", 1)
        path = raw_path.decode("utf-8")
        if path == CLOSURE:
            continue
        if revision is None:
            mode, oid, stage = metadata.decode().split()
            require(stage == "0", "unmerged release input")
            source = Path(root) / path
            require(source.is_file() and not source.is_symlink(), "missing or linked release input")
            data = source.read_bytes()
            require(bool(source.stat().st_mode & 0o111) == (mode == "100755"),
                    "release input executable mode mismatch")
        else:
            mode, kind, oid = metadata.decode().split()
            require(kind == "blob", "submodules are not release inputs")
            data = git(root, "cat-file", "blob", oid)
        require(mode in ("100644", "100755"), "unsupported release input mode")
        entries.append([path, mode, sha256(data)])
    entries.sort(key=lambda entry: entry[0])
    require(entries, "empty release inputs")
    return {"files": entries, "sha256": sha256(canonical(entries))}


def latest_fence(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    openings = list(re.finditer(r"^```stage5-artifact-json[^\n]*$", text, re.MULTILINE))
    require(openings, "closure requires a stage5-artifact-json record")
    latest = text[openings[-1].start():]
    match = re.match(r"```stage5-artifact-json\n(.*?)\n```[ \t]*(?:\n|$)", latest,
                     re.DOTALL)
    require(match is not None, "latest artifact fence is incomplete or malformed")
    return match.group(1), latest[match.end():]


def pending_candidate(text):
    _, tail = latest_fence(text)
    return NEW_CANDIDATE in tail.splitlines()


def parse_record(text):
    payload, _ = latest_fence(text)
    require(not pending_candidate(text), "new candidate request is pending")
    record = json.loads(payload)
    require(isinstance(record, dict) and set(record) == RECORD_KEYS,
            "unexpected closure record schema")
    require(type(record["schema_version"]) is int and record["schema_version"] == 1,
            "unsupported closure record version")
    for field in ("run_id", "run_attempt", "artifact_id"):
        require(type(record[field]) is int and record[field] > 0, "invalid " + field)
    for field, length in (("producer_sha", 40), ("input_sha256", 64)):
        require(isinstance(record[field], str) and
                re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", record[field]),
                "invalid " + field)
    require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", record["repository"]),
            "invalid repository")
    require(record["artifact_name"] == "release-original", "wrong artifact name")
    require(record["artifact_digest"] is None or
            re.fullmatch(r"sha256:[0-9a-f]{64}", record["artifact_digest"]),
            "invalid artifact digest")
    check_expiry(record)
    validate_manifest(record["manifest"], record["version"])
    return record


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "timestamp must include timezone")
    return parsed


def check_expiry(metadata, now=None):
    now = now or datetime.now(timezone.utc)
    require(metadata.get("expired") is False and metadata.get("deleted", False) is False,
            "artifact expired or deleted")
    created, expires = timestamp(metadata["created_at"]), timestamp(metadata["expires_at"])
    require(created <= now < expires and timedelta(0) < expires - created <= timedelta(days=90),
            "artifact expiry outside allowed live retention")


def validate_manifest(manifest, version):
    require(isinstance(manifest, dict) and set(manifest) ==
            {"schema_version", "project", "version", "artifacts"}, "invalid manifest")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest["project"] == "mapprojax"
            and manifest["version"] == version, "manifest version mismatch")
    files = manifest["artifacts"]
    require(isinstance(files, dict) and len(files) == 2, "expected wheel and sdist")
    require(sum(name.endswith(".whl") for name in files) == 1 and
            sum(name.endswith(".tar.gz") for name in files) == 1, "expected wheel and sdist")
    for name, identity in files.items():
        require(PurePosixPath(name).name == name and "\\" not in name
                and ":" not in name and "\x00" not in name and
                ((re.fullmatch(r"mapprojax-[^-]+-[^-]+-[^-]+-[^-]+\.whl", name)
                  and name.split("-")[1] == version) if name.endswith(".whl") else
                 (re.fullmatch(r"mapprojax-[^-]+\.tar\.gz", name)
                  and name == f"mapprojax-{version}.tar.gz")), "unsafe distribution name")
        require(isinstance(identity, dict) and set(identity) == {"size", "sha256"}
                and type(identity["size"]) is int and identity["size"] > 0 and
                re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]), "invalid file identity")


class TokenSafeRedirect(HTTPRedirectHandler):
    """Never forward the API credential on any redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require(urlsplit(newurl).scheme == "https", "insecure API redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def api(path, binary=False):
    token = os.environ.get("GH_TOKEN")
    require(token, "GH_TOKEN is required for read-only GitHub API verification")
    request = Request("https://api.github.com/" + path, headers={
        "Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"})
    with build_opener(TokenSafeRedirect()).open(request, timeout=120) as response:
        data = response.read()
    return data if binary else json.loads(data)


def repository():
    value = os.environ.get("GITHUB_REPOSITORY", "")
    require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value),
            "GITHUB_REPOSITORY owner/repo is required")
    return value


def select(root, tag=None):
    root = Path(root)
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    if tag is not None:
        require(tag == "v" + version, "tag must exactly match project version")
    closure = root / CLOSURE
    closure_bytes = closure.read_bytes() if closure.exists() else None
    text = closure_bytes.decode() if closure_bytes is not None else None
    pending = text is not None and pending_candidate(text)
    if text is None or pending:
        require(tag is None, "tag requires closure record without pending candidate request")
        digest = inputs(root)["sha256"]
        require(digest == inputs(root, "HEAD")["sha256"], "producer checkout is not clean")
        if pending:
            require(git(root, "show", "HEAD:" + CLOSURE) == closure_bytes,
                    "new candidate request must be committed")
        return {"mode": "produce", "input_sha256": digest,
                "reason": "new-candidate-request" if pending else "closure-absent"}
    record = parse_record(text)
    require(record["repository"] == repository(), "repository mismatch")
    require(record["version"] == version, "closure version mismatch")
    digest = inputs(root)["sha256"]
    require(record["input_sha256"] == digest == inputs(root, record["producer_sha"])["sha256"],
            "producer/current input digest mismatch")
    if git(root, "ls-tree", "--name-only", record["producer_sha"], "--", CLOSURE).strip():
        producer_closure = git(root, "show", record["producer_sha"] + ":" + CLOSURE)
        require(closure_bytes.startswith(producer_closure),
                "closure must preserve the entire unchanged producer closure prefix")
        require(pending_candidate(producer_closure.decode()),
                "producer closure requires explicit new candidate request")
    if tag is not None:
        head = git(root, "rev-parse", "HEAD").decode().strip()
        main = api("repos/" + repository() + "/git/ref/heads/main")
        require(main["object"]["sha"] == head, "tag closure must be exact current main")
        runs = api("repos/" + repository() + "/actions/workflows/ci.yml/runs?head_sha=" + head + "&event=push&per_page=100")
        require(any(run["head_sha"] == head and run["head_branch"] == "rocm-extra"
                    and run["conclusion"] == "success" and run["status"] == "completed"
                    for run in runs["workflow_runs"]), "no successful exact closure candidate CI")
    return {"mode": "reuse", **record}


def actual_metadata(record):
    repo = repository()
    require(record["repository"] == repo, "repository mismatch")
    metadata = api(f"repos/{repo}/actions/artifacts/{record['artifact_id']}")
    check_expiry(metadata)
    for field in ("created_at", "expires_at", "expired"):
        require(metadata[field] == record[field], "artifact metadata changed: " + field)
    require(metadata["id"] == record["artifact_id"] and
            metadata["name"] == record["artifact_name"] and
            metadata.get("digest") == record["artifact_digest"], "artifact identity mismatch")
    require(metadata["workflow_run"]["id"] == record["run_id"] and
            metadata["workflow_run"]["head_sha"] == record["producer_sha"], "artifact source mismatch")
    run = api(f"repos/{repo}/actions/runs/{record['run_id']}")
    require(run["head_sha"] == record["producer_sha"] and run["run_attempt"] == record["run_attempt"]
            and run["path"] == ".github/workflows/ci.yml" and run["event"] == "push"
            and run["head_branch"] == "rocm-extra", "producer run identity mismatch")
    return metadata, run


def extract_bundle(data, destination):
    destination = Path(destination)
    require(not destination.exists(), "download destination must be fresh")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate zip members")
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename
                    and ":" not in item.filename and "\x00" not in item.filename
                    and path.as_posix() + ("/" if item.is_dir() else "") == item.filename
                    and not stat.S_ISLNK(item.external_attr >> 16), "unsafe artifact zip path")
            require(item.filename in ("manifest.json", "inputs.json", "dist/") or
                    (len(path.parts) == 2 and path.parts[0] == "dist" and
                     (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))),
                    "unexpected artifact zip member")
        archive.extractall(destination)


def verify_bundle(bundle, record):
    bundle = Path(bundle)
    require(bundle.is_dir() and not bundle.is_symlink() and
            all(not path.is_symlink() for path in bundle.iterdir()) and
            (bundle / "dist").is_dir(), "linked or malformed artifact bundle")
    manifest = json.loads((bundle / "manifest.json").read_text())
    require(manifest == record["manifest"], "artifact manifest differs from closure")
    validate_manifest(manifest, record["version"])
    require({p.name for p in bundle.iterdir()} == {"dist", "manifest.json", "inputs.json"},
            "unexpected bundle content")
    require({p.name for p in (bundle / "dist").iterdir()} == set(manifest["artifacts"]),
            "missing or unexpected distribution")
    for name, identity in manifest["artifacts"].items():
        path = bundle / "dist" / name
        require(path.is_file() and not path.is_symlink() and
                {"size": path.stat().st_size, "sha256": sha256(path.read_bytes())} == identity,
                "distribution hash/size mismatch: " + name)
    payload = json.loads((bundle / "inputs.json").read_text())
    require(payload["sha256"] == sha256(canonical(payload["files"])) == record["input_sha256"],
            "artifact input manifest mismatch")


def protection():
    settings = api("repos/" + repository() + "/environments/pypi")
    reviewers = [rule for rule in settings.get("protection_rules", [])
                 if rule.get("type") == "required_reviewers"]
    require(any(entry.get("type") == "User" and
                str(entry.get("reviewer", {}).get("login", "")).casefold() == "leejjoon"
                for rule in reviewers for entry in rule.get("reviewers", [])),
            "pypi environment requires User reviewer leejjoon")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["choose", "inputs", "provenance", "download", "verify"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-id", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--protection", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "inputs":
        result = inputs(args.root)
    elif args.command == "provenance":
        require(args.artifact_id and args.manifest, "provenance needs artifact ID and manifest")
        repo = repository()
        metadata = api(f"repos/{repo}/actions/artifacts/{args.artifact_id}")
        check_expiry(metadata)
        run = api(f"repos/{repo}/actions/runs/{metadata['workflow_run']['id']}")
        result = {"schema_version": 1, "repository": repo,
                  "version": tomllib.loads((args.root / "pyproject.toml").read_text())["project"]["version"],
                  "producer_sha": git(args.root, "rev-parse", "HEAD").decode().strip(),
                  "input_sha256": inputs(args.root)["sha256"], "run_id": run["id"],
                  "run_attempt": run["run_attempt"], "artifact_id": metadata["id"],
                  "artifact_name": metadata["name"], "artifact_digest": metadata.get("digest"),
                  "created_at": metadata["created_at"], "expires_at": metadata["expires_at"],
                  "expired": metadata["expired"], "deleted": False,
                  "manifest": json.loads(args.manifest.read_text())}
        parse_record("```stage5-artifact-json\n" + json.dumps(result) + "\n```")
        actual_metadata(result)
    else:
        result = select(args.root, args.tag)
        if args.command in ("download", "verify"):
            require(result["mode"] == "reuse", "artifact operation requires closure")
            metadata, run = actual_metadata(result)
            require(run["status"] == "completed" and run["conclusion"] == "success",
                    "producer CI is not successful")
            if args.command == "download":
                require(args.destination is not None, "download needs destination")
                data = api(f"repos/{repository()}/actions/artifacts/{result['artifact_id']}/zip", binary=True)
                if metadata.get("digest"):
                    require("sha256:" + sha256(data) == metadata["digest"], "artifact container hash mismatch")
                extract_bundle(data, args.destination)
                verify_bundle(args.destination, result)
            else:
                require(args.bundle is not None, "verify needs bundle")
                verify_bundle(args.bundle, result)
            if args.protection:
                protection()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    else:
        print(json.dumps(result, indent=2))
    if args.command == "choose" and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("mode=" + result["mode"] + "\n")


if __name__ == "__main__":
    main()
