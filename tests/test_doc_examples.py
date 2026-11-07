"""Run every documented Python fence verbatim in a fresh CPU process."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


PAYLOAD_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (Path("README.md"), Path("docs/user_manual.md"))
PYTHON_FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def _examples():
    examples = []
    for document in DOCUMENTS:
        text = (PAYLOAD_ROOT / document).read_text(encoding="utf-8")
        matches = list(PYTHON_FENCE.finditer(text))
        assert matches, f"No Python examples found in {document}"
        for match in matches:
            line = text.count("\n", 0, match.start()) + 1
            examples.append(pytest.param(match.group(1), id=f"{document}:{line}"))
    return examples


@pytest.fixture(scope="module")
def example_process(tmp_path_factory):
    cwd = tmp_path_factory.mktemp("doc-examples")
    assert not cwd.resolve().is_relative_to(PAYLOAD_ROOT), (
        "Documentation examples require a temporary cwd outside the payload root"
    )
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "JAX_ENABLE_X64"):
        env.pop(name, None)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_PLATFORM_NAME"] = "cpu"

    # Source runs use the current interpreter's installed/editable package.
    # Artifact runs must instead prove that the same environment resolves the
    # package from the expected venv's actual purelib, never from the payload.
    if env.get("MAPPROJAX_EXPECTED_INSTALL_ROOT"):
        probe = subprocess.run(
            [sys.executable, "-c", """
import os
from pathlib import Path
import sys
import sysconfig
import mapprojax

expected = Path(os.environ["MAPPROJAX_EXPECTED_INSTALL_ROOT"]).resolve()
purelib = Path(sysconfig.get_path("purelib")).resolve()
origin = Path(mapprojax.__file__).resolve()
assert Path(sys.prefix).resolve() == expected, (sys.prefix, expected)
assert purelib.is_relative_to(expected), (purelib, expected)
assert origin.is_relative_to(purelib), (origin, purelib)
print(origin)
"""],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=300,
        )
        assert probe.returncode == 0, (
            f"Installed documentation import provenance failed:\n"
            f"stdout:\n{probe.stdout}\nstderr:\n{probe.stderr}"
        )
    return cwd, env


@pytest.mark.parametrize("code", _examples())
def test_python_fence(code, example_process, request):
    cwd, env = example_process
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{request.node.callspec.id} timed out after {exc.timeout}s")
    assert proc.returncode == 0, (
        f"{request.node.callspec.id} exited with {proc.returncode}:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
