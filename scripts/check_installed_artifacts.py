"""Bounded validation of the original mapprojax wheel and source distribution."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser


SUPPORT_FILES = (
    "benchmarks/compare_stage3.py", "benchmarks/benchmark_precision.py",
    "scripts/check_installed_artifacts.py", "README.md", "docs/user_manual.md",
    "pyproject.toml", "LICENSE", "CHANGELOG.md", "uv.lock",
    "ci/minimum.txt", "ci/latest.txt", "ci/reference-cpu.txt",
    "ci/reference-rocm.txt", "ci/build.txt",
    ".github/workflows/ci.yml", ".github/workflows/publish.yml",
    "scripts/release_artifact.py",
)
REQUIRED_TESTS = (
    "test_benchmark_precision.py", "test_doc_examples.py",
    "test_installed_provenance.py", "test_stage5_verification.py",
    "test_x64_policy.py", "test_wcs_new.py",
)
TIMEOUT = 3600


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or
            any(part in ("", ".", "..") for part in name.split("/")) or
            ":" in name or "\x00" in name):
        raise ValueError(f"unsafe path: {name!r}")
    return path


def file_identity(path):
    return {"size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def distribution_files(dist_dir):
    paths = sorted(Path(dist_dir).iterdir())
    if len(paths) != 2 or any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("dist directory must contain exactly one wheel and one sdist")
    for path in paths:
        safe_name(path.name)
    wheel = [p for p in paths if re.fullmatch(r"mapprojax-[^-]+-[^-]+-[^-]+-[^-]+\.whl", p.name)]
    source = [p for p in paths if re.fullmatch(r"mapprojax-[^-]+\.tar\.gz", p.name)]
    if len(wheel) != 1 or len(source) != 1:
        raise ValueError("unexpected distribution names")
    version = wheel[0].name.split("-")[1]
    if source[0].name != f"mapprojax-{version}.tar.gz":
        raise ValueError("wheel and sdist versions differ")
    return wheel[0], source[0], version


def make_manifest(dist_dir):
    wheel, source, version = distribution_files(dist_dir)
    return {"schema_version": 1, "project": "mapprojax", "version": version,
            "artifacts": {p.name: file_identity(p) for p in (wheel, source)}}


def verify_manifest(dist_dir, manifest):
    if set(manifest) != {"schema_version", "project", "version", "artifacts"}:
        raise ValueError("invalid manifest fields")
    if (type(manifest["schema_version"]) is not int or
            manifest["schema_version"] != 1 or manifest["project"] != "mapprojax"):
        raise ValueError("unsupported manifest")
    if not isinstance(manifest["artifacts"], dict):
        raise ValueError("invalid artifact inventory")
    for name, identity in manifest["artifacts"].items():
        if len(safe_name(name).parts) != 1:
            raise ValueError("artifact must be a basename")
        if (not isinstance(identity, dict) or set(identity) != {"size", "sha256"} or
                type(identity["size"]) is not int or identity["size"] < 0 or
                not isinstance(identity["sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"])):
            raise ValueError("invalid artifact identity")
    if manifest != make_manifest(dist_dir):
        raise ValueError("distribution inventory, version, size or SHA-256 mismatch")
    return distribution_files(dist_dir)[:2]


def wheel_contents(path):
    with zipfile.ZipFile(path) as archive:
        members = {}
        for info in archive.infolist():
            safe_name(info.filename.rstrip("/"))
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("wheel contains a symlink")
            if info.is_dir():
                continue
            if info.filename in members:
                raise ValueError("duplicate wheel member")
            members[info.filename] = archive.read(info)
    records = [name for name in members if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("wheel must have exactly one RECORD")
    record = records[0]
    normalized = {}
    for row in csv.reader(io.StringIO(members[record].decode("utf-8"))):
        if len(row) != 3:
            raise ValueError("invalid RECORD row")
        name, digest, size = row
        safe_name(name)
        if name in normalized or name not in members:
            raise ValueError("duplicate or missing RECORD member")
        if name == record:
            if digest or size:
                raise ValueError("RECORD self-entry must be unhashed")
        else:
            expected = "sha256=" + base64.urlsafe_b64encode(
                hashlib.sha256(members[name]).digest()).rstrip(b"=").decode("ascii")
            if digest != expected or size != str(len(members[name])):
                raise ValueError(f"RECORD hash/size mismatch: {name}")
        normalized[name] = (digest, size)
    if set(normalized) != set(members):
        raise ValueError("wheel has unrecorded members")
    del members[record]
    return members, normalized


def compare_wheels(direct, derived):
    if wheel_contents(direct) != wheel_contents(derived):
        raise ValueError("wheel contents or normalized RECORD differ")


def tar_contents(path):
    members = {}
    with tarfile.open(path, "r:gz") as archive:
        for info in archive:
            safe_name(info.name.rstrip("/"))
            if info.isdir():
                continue
            if not info.isfile() or info.name in members:
                raise ValueError("archive contains links, special or duplicate files")
            members[info.name] = archive.extractfile(info).read()
    roots = {PurePosixPath(name).parts[0] for name in members}
    if len(roots) != 1:
        raise ValueError("archive must have one root directory")
    root = roots.pop()
    if any(len(PurePosixPath(name).parts) < 2 for name in members):
        raise ValueError("archive has root-level files")
    return {name[len(root) + 1:]: data for name, data in members.items()}


def extract_source(archive, destination):
    members = tar_contents(archive)
    destination.mkdir()
    for name, data in members.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return destination


def validate_metadata(metadata, license_data, project, license_text):
    message = BytesParser().parsebytes(metadata)
    expected = {"Name": project["name"], "Version": project["version"],
                "Summary": project["description"],
                "Requires-Python": project["requires-python"], "License-Expression": "MIT"}
    for key, value in expected.items():
        if message.get_all(key) != [value]:
            raise ValueError(f"distribution metadata mismatch: {key}")
    if message.get_all("License-File") != ["LICENSE"]:
        raise ValueError("missing LICENSE metadata")
    urls = set(message.get_all("Project-URL", []))
    if not {f"{key}, {value}" for key, value in project["urls"].items()} <= urls:
        raise ValueError("project URLs mismatch")
    requirements = {re.sub(r"\s+", "", value) for value in message.get_all("Requires-Dist", [])}
    if not {re.sub(r"\s+", "", value) for value in project.get("dependencies", [])} <= requirements:
        raise ValueError("runtime dependency metadata mismatch")
    if set(message.get_all("Provides-Extra", [])) != set(project.get("optional-dependencies", {})):
        raise ValueError("optional dependency metadata mismatch")
    if license_data != license_text or not license_text.strip():
        raise ValueError("distribution LICENSE differs from source")
    return message["Version"]


def validate_distributions(wheel, source, root):
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    license_text = (root / "LICENSE").read_bytes()
    members, _ = wheel_contents(wheel)
    prefix = f"mapprojax-{project['version']}.dist-info/"
    validate_metadata(members[prefix + "METADATA"], members[prefix + "licenses/LICENSE"],
                      project, license_text)
    members = tar_contents(source)
    validate_metadata(members["PKG-INFO"], members["LICENSE"], project, license_text)


def payload_inventory(root):
    root = Path(root).resolve()
    tests = root / "tests"
    if tests.is_symlink() or not tests.is_dir():
        raise ValueError("missing regular tests directory")
    paths = []
    for path in tests.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"payload symlink: {path}")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_file():
            paths.append(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError(f"payload special file: {path}")
    for name in REQUIRED_TESTS:
        if f"tests/{name}" not in paths:
            raise ValueError(f"missing required test: {name}")
    for name in SUPPORT_FILES:
        path = root / name
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"missing or linked payload support: {name}")
        paths.append(name)
    return sorted(paths)


def copy_payload(root, destination):
    paths = payload_inventory(root)
    destination.mkdir()
    inventory = {}
    for name in paths:
        safe_name(name)
        source = root / name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inventory[name] = file_identity(target)
    return inventory


def clean_environment(backend="cpu", install_root=None):
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTHONWARNINGS",
                 "VIRTUAL_ENV", "JAX_ENABLE_X64", "JAX_DISABLE_JIT",
                 "MAPPROJAX_EXPECTED_SOURCE_ROOT", "MAPPROJAX_EXPECTED_INSTALL_ROOT"):
        env.pop(name, None)
    for name in list(env):
        if name.startswith("PYTEST_"):
            del env[name]
    env.update(PYTHONNOUSERSITE="1", JAX_PLATFORMS=backend,
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    if install_root is not None:
        env["MAPPROJAX_EXPECTED_INSTALL_ROOT"] = str(install_root)
    return env


def validate_build_constraints(path):
    pins = dict(re.findall(r"(?m)^([A-Za-z0-9_.-]+)==([^\s;\\]+)", path.read_text()))
    pins = {name.lower().replace("_", "-"): version for name, version in pins.items()}
    if not {"build", "setuptools", "wheel"} <= pins.keys():
        raise ValueError("build constraints must exactly pin build, setuptools and wheel")
    major = re.match(r"\d+", pins["setuptools"])
    if major is None or int(major.group()) < 77:
        raise ValueError("PEP 639 requires setuptools >=77")


def run_command(command, cwd, env, log, expect_failure=False):
    if not command or any(not isinstance(arg, str) for arg in command):
        raise ValueError("command must be a nonempty string argument list")
    with Path(log).open("w") as stream:
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, timeout=TIMEOUT, check=False)
    if (proc.returncode != 0) != expect_failure:
        raise RuntimeError(f"unexpected command exit {proc.returncode}: {command!r}; see {log}")
    return proc.returncode


def check(args):
    root = args.source_root.resolve()
    wheel, source = verify_manifest(args.dist_dir, json.loads(args.manifest.read_text()))
    wheel, source = wheel.resolve(), source.resolve()
    validate_distributions(wheel, source, root)
    out = args.output_dir.resolve()
    if out == root or root in out.parents or out in root.parents:
        raise ValueError("output must be outside source tree")
    out.mkdir(parents=True, exist_ok=False)
    payload = out / "payload"
    inventory = copy_payload(root, payload)
    (out / "payload-manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
    constraints = args.constraints.resolve(strict=True)
    build_constraints = args.build_constraints.resolve(strict=True)
    validate_build_constraints(build_constraints)
    # Exact requirements, not merely a constrained frontend, govern every backend.
    for path in (constraints, build_constraints):
        if not path.read_text().strip():
            raise ValueError(f"empty constraints: {path}")
    shutil.copyfile(constraints, out / "constraints.txt")
    shutil.copyfile(build_constraints, out / "build-constraints.txt")
    constraints = out / "constraints.txt"
    build_constraints = out / "build-constraints.txt"
    counter = 0

    def command(argv, env, cwd=payload, fail=False):
        nonlocal counter
        counter += 1
        return run_command([str(arg) for arg in argv], cwd, env,
                           out / f"{counter:02d}.log", fail)

    def environment(name):
        venv = out / name
        env = clean_environment(args.backend, venv)
        command(["uv", "venv", "--python", sys.executable, venv], env)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command(["uv", "pip", "install", "--python", python, "-c", build_constraints,
                 "-r", build_constraints], env)
        return python, env

    def install(python, env, artifact):
        requirement = str(artifact) + ("[test,rocm]" if args.backend == "rocm" else "[test]")
        command(["uv", "pip", "install", "--python", python, "-c", constraints,
                 "--no-build-isolation", requirement], env)

    audit = (
        "import importlib.util, pathlib; "
        "p=pathlib.Path('tests/test_installed_provenance.py'); "
        "s=importlib.util.spec_from_file_location('provenance',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "m.assert_provenance()"
    )
    backend_audit = (
        "import jax; devices=jax.devices(); print(devices); "
        + ("assert jax.default_backend() == 'cpu' and all(d.platform == 'cpu' for d in devices)"
           if args.backend == "cpu" else
           "assert jax.default_backend() == 'gpu' and all(d.platform == 'gpu' for d in devices); "
           "assert any('rocm' in str(d.client.platform_version).lower() for d in devices)")
    )
    direct_python = direct_env = None
    artifacts = [("direct", wheel)]
    if args.backend == "cpu":
        artifacts.append(("sdist", source))
    for name, artifact in artifacts:
        python, env = environment(name)
        install(python, env, artifact)
        command([python, "-c", audit], env)
        command([python, "-c", backend_audit], env)
        command([python, "-m", "pytest", "-ra", "-s", "tests"], env)
        if name == "direct":
            direct_python, direct_env = python, env
            shadow = out / "shadow"
            (shadow / "mapprojax").mkdir(parents=True)
            (shadow / "mapprojax/__init__.py").write_text("# intentional provenance negative control\n")
            negative_env = {**env, "PYTHONPATH": str(shadow)}
            command([python, "-c", audit], negative_env, fail=True)
            log = (out / f"{counter:02d}.log").read_text()
            if "AssertionError" not in log or "mapprojax origin" not in log:
                raise RuntimeError("negative control failed for a reason other than provenance")

    if args.backend == "cpu":
        python, env = environment("comparison-build")
        extracted = extract_source(source, out / "comparison-source")
        derived = out / "comparison-wheel"
        command([python, "-m", "build", "--wheel", "--no-isolation", "--outdir", derived,
                 extracted], env)
        wheels = list(derived.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("comparison build did not produce exactly one wheel")
        compare_wheels(wheel, wheels[0])

    if args.reprojax_archive is not None:
        sibling = extract_source(args.reprojax_archive.resolve(strict=True), out / "reprojax")
        project = tomllib.loads((sibling / "pyproject.toml").read_text())["project"]
        if project["name"] != "reprojax" or not (sibling / "tests").is_dir():
            raise ValueError("not a reprojax test source archive")
        deps = project.get("dependencies", []) + project["optional-dependencies"]["test"]
        deps = [dep for dep in deps if not re.match(r"mapprojax\b", dep, re.I)]
        command(["uv", "pip", "install", "--python", direct_python, "-c", constraints,
                 *deps], direct_env)
        command(["uv", "pip", "install", "--python", direct_python,
                 "--no-deps", "--no-build-isolation", sibling], direct_env)
        command([direct_python, "-c", audit], direct_env)
        command([direct_python, "-c", backend_audit], direct_env)
        sibling_audit = (
            "import importlib.metadata as md, pathlib, sysconfig, reprojax; "
            "p=pathlib.Path(sysconfig.get_path('purelib')).resolve(); "
            "assert pathlib.Path(reprojax.__file__).resolve().is_relative_to(p); "
            f"assert md.version('reprojax') == {project['version']!r}; print(reprojax.__file__)"
        )
        command([direct_python, "-c", sibling_audit], direct_env)
        # Keep collection external to the sibling's src tree as well.
        sibling_payload = out / "reprojax-tests"
        sibling_payload.mkdir()
        for item in sibling.iterdir():
            if item.name in ("src", ".git", ".venv", "__pycache__"):
                continue
            if item.is_dir():
                shutil.copytree(item, sibling_payload / item.name)
            else:
                shutil.copyfile(item, sibling_payload / item.name)
        command([direct_python, "-m", "pytest", "-ra", "-s", "tests"],
                direct_env, cwd=sibling_payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("manifest", "verify", "check"):
        sub = commands.add_parser(name)
        sub.add_argument("--dist-dir", required=True, type=Path)
        if name == "manifest":
            sub.add_argument("--output", required=True, type=Path)
        else:
            sub.add_argument("--manifest", required=True, type=Path)
        if name == "check":
            for option in ("source-root", "output-dir", "constraints", "build-constraints"):
                sub.add_argument("--" + option, required=True, type=Path)
            sub.add_argument("--backend", choices=("cpu", "rocm"), default="cpu")
            sub.add_argument("--reprojax-archive", type=Path)
    compare = commands.add_parser("compare-wheels")
    compare.add_argument("direct", type=Path)
    compare.add_argument("derived", type=Path)
    args = parser.parse_args(argv)
    if args.command == "manifest":
        args.output.write_text(json.dumps(make_manifest(args.dist_dir), indent=2) + "\n")
    elif args.command == "verify":
        verify_manifest(args.dist_dir, json.loads(args.manifest.read_text()))
    elif args.command == "compare-wheels":
        compare_wheels(args.direct, args.derived)
    else:
        check(args)


if __name__ == "__main__":
    main()
