# Developer Guide

`mapprojax` is a JAX coordinate kernel for FITS TAN and SIN projections with SIP distortion. It owns projection-domain validity, the pixel/native/celestial transforms, precision policy, and coordinate-only reprojection. It does not own general FITS WCS parsing or image interpolation; use Astropy for broad WCS support and `reprojax` for image resampling.

## Architecture

- `src/mapprojax/wcs.py` defines the public `Tan` and `Sin` classes. Each is a JAX PyTree with an explicit `precision` (`"f32"`, `"mixed"`, or `"f64"`). There are no separate NumPy, Array, or Mixed class families.
- `src/mapprojax/projections.py` contains the backend-independent TAN/SIN formula mixins used by the precision-explicit WCS classes.
- `src/mapprojax/base.py` contains the shared validity-aware pixel/native pipeline and host offset-composition helpers.
- `src/mapprojax/jax_base.py` contains JAX precision-policy utilities and the PyTree registration for `Sip`.
- `src/mapprojax/sip.py` implements SIP forward and inverse distortion. Its coefficient shapes and `inverse_steps` are static PyTree structure/configuration for batching and compilation.
- `src/mapprojax/reproject.py` composes two conforming projection pipelines into a coordinate map. It does not resample data.

The public root exports are `Tan`, `Sin`, `Reprojector`, `absolute_from_offset`, `offset_from_absolute`, and `compose_relative_rotation`. Do not add aliases for retired APIs.

## Precision contract

| Mode | Native and pixel math | Celestial rotation | Absolute `proj` / `unproj` |
| --- | --- | --- | --- |
| `f32` | float32 | unavailable | No; use the offset methods and host helper composition. |
| `mixed` | float32 | float64 | Yes; JAX x64 must be enabled. |
| `f64` | float64 | float64 | Yes; JAX x64 must be enabled. |

`crval` is retained at the supplied reference dtype. Do not assume that casting a truncated reference back to float64 restores its information. Host composition of the reference rotation uses concrete float64 references; it must reject tracers and non-float64 references. Precision trade-offs are workload-dependent. The [corrected Stage 5 measurements](docs/benchmarks/precision_results_stage5_2026-11-07.md) give sampled Reprojector f32 maxima of 5.28e-4–8.44e-3 px on CPU/ROCm across 4096² plate-scale/separation cases against a JAX f64 absolute path, using matching typed coordinates. F64 agreement on those cases is at most 4.75e-9 px (CPU) / 3.88e-9 px (ROCm), not an independent accuracy bound. The separate [Astropy worked pair](docs/benchmarks/accuracy_results.md) gives f32 maxima 2.49e-4 / 2.03e-4 px on 1009/1090 / 320/1090 overlapping samples, not every pixel. Mixed WCS forward 0.17–17 mas values are per-case regression caps, not measured maxima. The [August precision report](docs/benchmarks/precision_results.md) remains historical.

All pixel coordinates are 0-based. `crval` is in degrees; projection methods take and return radians in `(longitude, latitude)` order. TAN excludes the horizon and back hemisphere; SIN includes the front-hemisphere boundary and its inverse includes the unit-disc boundary. Public projection outputs are NaN when invalid; `Reprojector.pixel_map` instead returns finite zero stand-ins and a separate boolean validity mask. Internal validity flags keep invalid values from poisoning downstream arithmetic or gradients.

## Public API patterns

Construct a projection with an explicit precision mode:

```python
import jax
from mapprojax import Tan

jax.config.update("jax_enable_x64", True)
wcs = Tan(crpix, cd, crval, precision="mixed")
lon, lat = wcs.unproj(x, y)
```

Batch compatible WCS PyTrees with `jax.tree_util.tree_map` and `jax.vmap`, rather than adding per-centre `*Array` classes. A batch must share projection type, precision, PyTree structure, SIP coefficient shapes, and `inverse_steps`. For many WCS objects over the same pixel grid, use `jax.vmap(lambda w: w.unproj(x, y), in_axes=0, out_axes=1)` to return `(pixels, centres)`.

For true absolute coordinates from the offset path, compose the producer-typed offsets on the host:

```python
from mapprojax import absolute_from_offset

offset = wcs.unproj_offset(x, y)
absolute = absolute_from_offset(offset, crval=wcs.crval, offset_scale=wcs.offset_scale())
```

This helper is not JIT- or autodiff-compatible. `crval` and `offset_scale` are keyword-only. Preserve the offset component and scale dtypes returned by the producer.

Use `Reprojector(src, tgt)` for source-to-target pixel coordinates. Its `matrix_precision="f32"` mode removes fp64 from the per-pixel path; the host-composed rotation is not differentiable with respect to the two WCS `crval` values. When gradients with respect to `crval` are required, build the relative matrix from the live references inside the traced function and pass it to `Reprojector.from_matrix`. After a transformation that leaves `crval` traced or float32, compose from retained host references with `compose_relative_rotation(src, tgt, crval_src=..., crval_tgt=...)` and pass the result to `from_matrix`.

`pixel_map(x, y, *, bounds_shape=None)` returns mapped coordinates and a validity mask. Coordinates are finite and zero where invalid; a supplied `(ny, nx)` bounds tuple additionally applies the half-open target bounds `[-0.5, nx-0.5) × [-0.5, ny-0.5)`. The tuple must be static and contain two positive built-in Python integers. Use `jnp.where` to mask before nonlinear losses, and choose regular stand-ins before downstream computations that are unsafe at invalid coordinates. `pix_to_pix` is unchanged and keeps its NaN invalid-output contract. See the [user manual](docs/user_manual.md#differentiability-contract) for the complete differentiation table, including the distinction between eager composition and `from_matrix`.

Serialization uses the tested JSON-safe `to_dict()` / `from_dict()` contract. It preserves projection type, precision, reference dtype, and SIP state and rejects mismatched types or legacy dictionaries. ASDF support and the undocumented byte-serialization claims have been retired.

## Validation and changes

Use the focused tests while iterating, followed by the project test suite on CPU and the supported ROCm setup before release. Independent Astropy agreement is the projection-correctness check; do not replace it with only a mapprojax round trip. The regression suite also covers domain edges, masked gradients, PyTree transforms, SIP inverse behavior, CD validation, and JSON restoration.

When changing numerical behavior, update the accuracy harness/baselines and regenerate the corresponding results using the existing scripts in `benchmarks/`. Do not change projection formulas without independent reference coverage.

## 0.3.0 release verification

Stage 5 changes only source comments/docstrings, not executable ASTs (strip only genuine module/class/function docstring nodes before comparison). Preserve the original `run` and `_run_fresh` test helpers, all runtime strings, protected baselines, the complete committed accuracy report and existing Stage 4 evidence. Do not normalize a changed reference report or relax a cap. Compare protected paths against `e5b588a` with `git diff --quiet` **before archiving**, generate their SHA-256 list before extraction, and verify that list inside each archive without requiring `.git`. Run `benchmark_accuracy.py` for numerical acceptance everywhere; `cmp` the regenerated markdown only in the recorded reference CPU environment.

Reference inputs are the original `e5b588a` `pyproject.toml` and `uv.lock` (SHA-256 `01b888bd6224dedd8b0486aec767adb37537ebad01feb4ce623af14ce6a8f3f4` and `6c2331ea97458144b198538b4e03752bb0cccfcedf18a622c06c8ea23217f7c6`). The reference constraints were exported with uv 0.11.8 from a clean archive of that revision, **not** the refreshed lock:

```bash
uv export --frozen --extra test --no-dev --no-emit-project --format requirements-txt --output-file /tmp/reference-cpu.txt
uv export --frozen --extra test --extra rocm --no-dev --no-emit-project --format requirements-txt --output-file /tmp/reference-rocm.txt
```

The interpreter/runtime are separate: reference Python 3.12.13, JAX/jaxlib 0.11.1, NumPy 2.5.1, Astropy 8.0.2; ROCm plugins 0.11.1 on Strix Halo, client `rocm 70253211`. The universal exports retain their original markers/hashes. `ci/minimum.txt` records the passing Python 3.11.15 / JAX+jaxlib 0.10.2 / NumPy 2.4.6 combination; `ci/latest.txt` is a recorded exact combination for standard (not freethreaded) Python 3.14.4. Exact build frontend/backend pins are in `ci/build.txt`; apply them to an explicit isolated build venv and use `python -m build --no-isolation`, not an unconstrained backend environment.

### Candidate artifact and local gate

Commits and pushes require their own authorization. CI runs on `rocm-extra` and `main`; the exact pre-closure payload SHA must be green. It builds the publication wheel and sdist **once**, from a clean git archive. Record the original `release-original` artifact's run ID/attempt, artifact ID, source/input identities, actual creation/expiry and per-file SHA-256. Requested retention is at most 90 days; the live API expiry may be earlier. A missing/deleted/expired artifact requires a newly validated candidate, even if local copies still exist.

To renew a candidate after closure exists, append the whole-line marker `<!-- stage5-new-candidate -->` **after** its last artifact fence, and obtain the applicable commit/push authorization. This selects a fresh CI producer without deleting historical closure evidence. Download, tag validation and publication reject the pending state rather than falling back to old bytes. After full CI/local validation, append a new complete `stage5-artifact-json` record after the marker. Only the latest record is operative; an invalid latest record never falls back to historical ones. Recheck the new exact closure SHA on CI and main before any separately authorized tag/publication.

`scripts/release_artifact.py provenance` generates the fixed `stage5-artifact-json` record for the closure from actual upload/API evidence; no placeholder record belongs in a closure. After downloading that exact artifact, run the bounded installer with its `manifest.json` and `dist/`:

```bash
git -C ../reprojax archive --format=tar.gz --prefix=reprojax/ --output="$REPROJAX_TAR" "$REPROJAX_SHA"
python scripts/check_installed_artifacts.py verify --dist-dir "$BUNDLE/dist" --manifest "$BUNDLE/manifest.json"
python scripts/check_installed_artifacts.py check --dist-dir "$BUNDLE/dist" --manifest "$BUNDLE/manifest.json" --source-root "$MAP_ARCHIVE" --output-dir "$OUT/cpu" --constraints ci/reference-cpu.txt --build-constraints ci/build.txt --backend cpu --reprojax-archive "$REPROJAX_TAR"
python scripts/check_installed_artifacts.py check --dist-dir "$BUNDLE/dist" --manifest "$BUNDLE/manifest.json" --source-root "$MAP_ARCHIVE" --output-dir "$OUT/rocm" --constraints ci/reference-rocm.txt --build-constraints ci/build.txt --backend rocm --reprojax-archive "$REPROJAX_TAR"
```

The CPU check installs the original wheel and sdist separately and runs the complete external fixture suite, including literal README/manual Python blocks in fresh CPU children. The sdist-derived comparison wheel is checked against the direct wheel's RECORD/member contents, **not** given a third full-suite run. Installed ROCm is run once on the direct wheel; source-archive suites on both backends remain separate gates. The supplied tests, non-package subjects and explicit support allowlist contain no runtime source checkout. Parent and dedicated child-provenance tests audit actual venv purelib paths; the shadow-package negative control must fail. Ordinary pytest counts/warnings are evidence; no per-node receipts are required. Reprojax verification uses its exact clean archive and the same mapprojax wheel, without source edits or resolver substitution. ROCm backend/device assertions must identify ROCm rather than assuming JAX's backend string is `rocm` (it may be `gpu`).

### Closure, main and publication

Only after the real local/CI gates pass, write append-only `benchmarks/results/stage5-closure.md` with actual evidence and the producer record. The evidence-only closure commit is the release commit; it cites the tested payload and downstream pair, never itself. Closure/main/tag CI reuses the exact original artifact and verifies unchanged inputs; it does not rebuild release bytes. No release-ready claim is justified by working-tree tests or local preparation distributions alone.

Promotion is a separately authorized **fast-forward only**. Recheck live remote/main identities and `git merge-base --is-ancestor main "$RELEASE_SHA"`; stop if main diverged or cannot fast-forward. Never rebase, create a merge commit, reset or force-push as a substitute. An authorized `git merge --ff-only "$RELEASE_SHA"` and normal main push preserve the exact closure SHA; require green CI and applicable branch checks on that SHA before separately authorized tagging of `v0.3.0` as it sits on main.

`publish.yml` accepts only `v*` tag pushes. Its upload step is armed with `if: success()` after recorded operator setup; tagging and protected upload approval remain independent operator gates. Record dated **actual** environment/publisher evidence (for example `gh api repos/leejjoon/mapprojax/environments/pypi`) in closure; verify the exact owner/repository/workflow/environment tuple and required reviewer, and recheck protection/artifact expiry immediately before arming and publication. Workflow lint and pure hash tests cannot prove GitHub enforcement. Publish only the original validated files after immediate per-file hash checks. Missing authority/setup retains publication pending; it is not permission for an alternate upload path. Finally, separately authorized PyPI installation verifies release-file hashes and the quick start; reprojax's re-pin/release is separate work.

The reviewer rule must include operator `leejjoon`. No extra second-person requirement is imposed: GitHub's `prevent_self_review` may be enabled or disabled by the operator; the real protected environment approval remains mandatory either way. The code checks identity/settings, not whether a local simulation thinks approval happened.

Arming changes the tracked workflow and therefore the release input digest. Separately authorized arming must precede production/validation of the final publication candidate. When arming after an existing closure, append a new-candidate request and revalidate the newly produced artifact; never expect an old artifact record to survive a workflow change. Readiness remains publication-pending after arming until separately authorized tagging and protected upload approval.
