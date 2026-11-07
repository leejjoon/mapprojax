"""Audit the parent process and the two unchanged fresh-interpreter helpers."""

import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import pkgutil
import sys
import sysconfig
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def assert_provenance(expected_executable=None, expected_prefix=None):
    """Check actual module files, not just distribution metadata or sys.path."""
    import mapprojax

    installed = os.environ.get("MAPPROJAX_EXPECTED_INSTALL_ROOT")
    source = os.environ.get("MAPPROJAX_EXPECTED_SOURCE_ROOT")
    assert not (installed and source), "conflicting provenance modes"
    prefix = Path(sys.prefix).resolve()
    executable = Path(os.path.abspath(sys.executable))
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    if expected_executable is not None:
        assert executable == Path(expected_executable), "child executable differs"
    if expected_prefix is not None:
        assert prefix == Path(expected_prefix), "child prefix differs"
    if installed:
        expected = Path(installed).resolve()
        assert prefix == expected, "unexpected installed sys.prefix"
        assert executable.parent in (expected / "bin", expected / "Scripts"), "unexpected venv executable"
        assert purelib.is_relative_to(expected), "purelib outside expected venv"
        package_root = purelib / "mapprojax"
    else:
        package_root = Path(source).resolve() / "src/mapprojax" if source else ROOT / "src/mapprojax"
    # Check the parent before walking its modules: a shadow package must fail here.
    origin = Path(mapprojax.__file__).resolve()
    assert origin == package_root / "__init__.py", f"mapprojax origin: {origin}, expected {package_root}"
    for info in pkgutil.walk_packages(mapprojax.__path__, "mapprojax."):
        importlib.import_module(info.name)
    origins = {}
    for name, module in list(sys.modules.items()):
        if name != "mapprojax" and not name.startswith("mapprojax."):
            continue
        filename = getattr(module, "__file__", None)
        assert filename, f"package module without origin: {name}"
        path = Path(filename).resolve()
        assert path.is_relative_to(package_root), f"mapprojax origin: {name}: {path}"
        if installed:
            assert path.is_relative_to(purelib), f"module outside purelib: {name}"
        origins[name] = str(path)
    version = importlib.metadata.version("mapprojax")
    metadata_root = ROOT if installed else Path(source).resolve() if source else ROOT
    expected_version = tomllib.loads((metadata_root / "pyproject.toml").read_text())["project"]["version"]
    assert version == expected_version, "installed distribution version differs from test subject"
    audit = {"executable": str(executable), "prefix": str(prefix),
             "purelib": str(purelib), "version": version, "origins": origins}
    print(json.dumps(audit, sort_keys=True))
    return audit


def test_parent_provenance():
    assert_provenance()


def _load_subject(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("module_name,helper_name", [
    ("test_x64_policy", "run"), ("test_wcs_new", "_run_fresh"),
])
def test_original_helper_child_provenance(module_name, helper_name):
    parent = assert_provenance()
    snippet = f"""
import importlib.util
spec = importlib.util.spec_from_file_location('child_provenance', {str(Path(__file__).resolve())!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.assert_provenance({parent['executable']!r}, {parent['prefix']!r})
"""
    result = getattr(_load_subject(module_name), helper_name)(snippet)
    output = result.stdout if hasattr(result, "stdout") else result
    child = json.loads(output.strip().splitlines()[-1])
    assert child == parent
