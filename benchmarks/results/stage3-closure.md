# Stage 3 closure manifest

Evidence for closing Stage 3 (C5–C8 in mapprojax, C2 and C7 in reprojax), measured Nov 4 2026 on the working trees described below. It repairs the nine blocking items from the committee's closure review (Fable and Astra, Nov 4 2026). Frozen C1 and C4 artifacts are unchanged from `d60a5b1`.

## Identity

- **mapprojax:** branch `rocm-extra`, base `d60a5b1`, uncommitted working tree. The commit SHAs are recorded in the plan's Stage 3 closure section once committed.
- **reprojax:** base `40221bf`, uncommitted working tree.
- **Legacy timing baseline:** `git archive d60a5b1 | tar -x -C <dir>`, outside the repository, run as `PYTHONPATH=<dir>/src python <dir>/benchmarks/benchmark_stage3.py --backend {cpu,rocm} --mode full <workload args> --implementation legacy`, back to back with each candidate (`python benchmarks/benchmark_stage3.py ... --implementation new` from this tree). Workload args: `--workload sip --order 3 --steps {2,3,5}`; `--workload sip --order 4 --steps 3 --size 1201`; `--workload batch`; `--workload pair --pair-precision {f32,f64}`.
- **Changes after timing:** the re-review repairs touched `base.py` and `utils.py` only in host-side reference validation (`_check_sequence_components`), which runs outside every timed `jit` kernel, so the timing evidence still applies.

| File | sha256 (first 16) |
| --- | --- |
| `src/mapprojax/base.py` | `10fadd9d5e6fd857` (`a6a789547e63562d` when timed) |
| `src/mapprojax/wcs.py` | `f38a92278e922396` |
| `src/mapprojax/reproject.py` | `52c0ccc5b801a02a` |
| `src/mapprojax/utils.py` | `74bf75980b36cca8` (`8702e7276e6a05c9` when timed) |
| `src/mapprojax/jax_base.py` | `e7f594952c6776a2` |
| `benchmarks/benchmark_stage3.py` | `28e3d4b06c32ac61` |
| `benchmarks/compare_stage3.py` | `96dbfcbea05aaa5b` (unchanged since C4) |

## Environment

- **Machine:** Strix Halo APU, CPU and ROCm.
- **mapprojax:** Python 3.12.13, JAX/jaxlib 0.11.1 with the jax-rocm7 plugin, NumPy 2.5.1, Astropy 8.0.2.
- **reprojax:**
  - CPU: its own `.venv`, JAX 0.12.0.1.
  - ROCm: a separate Python 3.12 venv with JAX 0.11.1 and the ROCm plugin, importing both working trees, because the reprojax `.venv` has no ROCm plugin.

## Results

| Gate | Result |
| --- | --- |
| mapprojax suite, CPU | 737 passed, 0 skipped |
| mapprojax suite, ROCm | 737 passed, 0 skipped |
| reprojax suite, CPU | 79 passed; 2 skipped (GPU-only batch tests) |
| reprojax suite, ROCm | 81 passed, 0 skipped |
| Accuracy, CPU | `benchmark_accuracy.py` exit 0. A second run is byte-identical to `docs/benchmarks/accuracy_results.md`. Every `jax`, `jax_mixed`, pair, control and oracle row matches C4 exactly. The only changes: `numpy` rows removed and `numpy_absolute` replaced by the regenerated `offset_host` rows. |
| Accuracy, ROCm | exit 0. The same rows match `stage3-c4r-accuracy-rocm.md` exactly. |
| Timing | `compare_stage3.py --output stage3-c8-comparison.json` over 14 pairs: exit 0, every pair passes all four rules |
| No recompilation on numerical leaf changes | `test_numeric_leaf_changes_reuse_the_compiled_function` (trace count, all modes, positive control) |

## Timing (`stage3-c8-*`)

Rules: warm `max(1.10·B, B+0.05 ms)`; lowering, compile and cold `max(1.20·B, B+50 ms)`. Warm is the median of 3 process medians of 30 repetitions. Lowering, compile and cold are medians of 5 fresh processes with the persistent cache disabled. B is the legacy `d60a5b1` implementation, measured in the same session.

| Backend | Workload | Warm ms (B → N) | Lowering ms | Compile ms | Cold ms |
| --- | --- | --- | --- | --- | --- |
| CPU | SPHEREx order 3, 2040², steps 2 | 77.5 → 71.8 | 82.2 → 79.9 | 281 → 264 | 449 → 409 |
| CPU | same, steps 3 | 89.1 → 85.9 | 93.2 → 89.8 | 342 → 312 | 516 → 488 |
| CPU | same, steps 5 | 121 → 112 | 160 → 111 | 437 → 392 | 712 → 596 |
| CPU | SIP order 4, 1201², evaluation only | 6.83 → 6.69 | 27.1 → 27.5 | 92.0 → 90.5 | 126 → 125 |
| CPU | batch 256×4096, mixed | 4.52 → 4.23 | 32.3 → 32.0 | 70.6 → 67.0 | 108 → 105 |
| CPU | `Reprojector` pair, f32, 2040² | 14.4 → 11.8 | 26.2 → 28.2 | 94.0 → 93.8 | 132 → 135 |
| CPU | `Reprojector` pair, f64, 2040² | 28.9 → 27.5 | 26.3 → 27.9 | 87.0 → 96.8 | 135 → 147 |
| ROCm | SPHEREx order 3, 2040², steps 2 | 16.0 → 6.29 | 81.9 → 78.7 | 991 → 915 | 1090 → 1000 |
| ROCm | same, steps 3 | 19.2 → 6.28 | 93.9 → 91.0 | 1170 → 1110 | 1290 → 1200 |
| ROCm | same, steps 5 | 24.8 → 6.45 | 164 → 111 | 1530 → 1400 | 1730 → 1530 |
| ROCm | SIP order 4, 1201², evaluation only | 0.854 → 0.910 | 27.9 → 27.7 | 175 → 163 | 200 → 207 |
| ROCm | batch 256×4096, mixed | 0.659 → 0.624 | 33.4 → 32.9 | 105 → 101 | 135 → 142 |
| ROCm | `Reprojector` pair, f32, 2040² | 0.373 → 0.369 | 25.0 → 27.9 | 148 → 159 | 180 → 197 |
| ROCm | `Reprojector` pair, f64, 2040² | 3.26 → 3.27 | 25.3 → 30.2 | 153 → 173 | 192 → 205 |

- **Tightest margin:** ROCm SIP order 4 warm, at 0.97 of its limit (0.910 ms against 0.939 ms). Every other metric is at most 0.91 of its limit.
- **SPHEREx on ROCm:** the 2.5–3.9× warm speedup is the precision change recorded at C4. Legacy `TanJaxMixed` promoted the SIP inverse to f64; the new mixed mode runs it in f32. See `stage3-c4-checkpoint.md`, "Reading the SPHEREx speedup".
- **Noise and drift.** The CPU batch legacy baseline has a 21.8% process-median dispersion. Astra's full-protocol repeat in the re-review measured 4.020 → 4.052 ms (batch) and 6.355 → 5.881 ms (order 4), and both passed every rule against both the repeat and these budgets. Against the C1 files, the CPU warm pair comparison would fail by 17–31%, but the legacy kernel itself re-measures 20–34% slower today (Fable, from a `d60a5b1` worktree), so that is machine drift. That is why baselines are re-measured in the same session.
- **Pair workloads:** these compare the whole legacy `Reprojector` (NumPy `Tan` legs) with the new one (`Tan(precision="f64")` legs, converted by `with_precision`) on the real SPHEREx L2 → Galactic template pair, using the same frame rotation.

## Closure repairs (committee blocking list, Nov 4 2026)

1. **Frozen evidence and report.**
   - Every edit under `benchmarks/results` from the C5–C8 session is reverted to `d60a5b1`. The edited copies are kept outside the repository.
   - `docs/benchmarks/accuracy_results.md` is regenerated by the script and reviewed (see Results).
2. **Retirement.**
   - Removed: `WCSBase`, `WCSArrayBase` and the absolute methods (`base.py`), and `WCSJax`, `WCSJaxArray`, `_JaxRotationPrecision` and both Mixed mixins (`jax_base.py`, which now holds only the x64 utilities and the `Sip` PyTree registration).
   - `_PrecisionWCSBase` now derives from `_PipelineBase`.
   - Also removed: the empty `io/asdf` and `resources` directories.
   - Carry-overs also done: the unreachable f32 branch in `r_matrix`, the duplicated `sip_steps` aux entry, and `_CLASS_NAMES`.
3. **Composition.**
   - `compose_relative_rotation(..., *, crval_src=None, crval_tgt=None)` validates each override like `crval`, and requires a finite `(3, 3)` frame.
   - `Reprojector.__init__` calls it; the duplicate in `reproject.py` is gone.
   - **Re-review repair:** a sequence such as `[np.float32(x), 30]` used to be promoted to float64 by `np.asarray` before the dtype check, so it was accepted. `_check_sequence_components` (`utils.py`) now rejects any traced or non-float64 floating component first, for the overrides, `wcs.crval` composition and the helpers' `crval`. Python floats and genuine float64 sequences are still accepted.
   - Tests cover: x64-off `jit` recovery that is bit-identical to the eager result, both overrides, bad and traced overrides, NaN/Inf/wrong-shape frames, and the public composition being used by both precisions.
4. **Regression net.** `tests/test_offset_projection.py` is rebuilt as helper compositions with the original budgets (49 tests):
   - fp64 matches `unproj`/`proj`;
   - round trips;
   - RA wrap and unnormalized input;
   - near-pole fp64;
   - SIP;
   - fp32 without x64 at 1e-6 rad;
   - the separation sweep to 45°;
   - f32 at ±89.9999° and ±90°, both projections, with a measured envelope. The maximum is 0.82–0.87 mas against a 2 mas limit at ±89.9999°, and 18.3–18.4 mas against 30 mas at ±90°. A host-cosine negative control shows 2.78e3 mas and 4.5e6 mas;
   - compiled producer scale;
   - dtype mismatch, Python-float scale, f64 absolute input with an f32 scale, broadcasting, NaN/Inf, zero scale, and tracer and bad-`crval` rejection.

   The recursive no-f64 walker is restored in `tests/test_x64_policy.py`:
   - It checks nested SIP with AP/BP on both legs, a closed-over `Reprojector`, a `Reprojector` passed as an argument, and `from_matrix`, with x64 on and off.
   - A mutation control proves it catches a plain f64 add inside a nested `jit`.
5. **Derivatives.** `tests/test_gradients_modes.py` (66 tests, no skips):
   - **f64:** `check_grads` forward and reverse on every leaf, including AP/BP. It covers Tan and Sin, A/B with seeded AP/BP and AP/BP-only SIP, and the forward and inverse paths, both offset and absolute.
   - **f32 and mixed:** the AD Jacobian is compared with the f64 Jacobian, and forward and reverse mode are checked for consistency.
   - **Recorded mixed-versus-f64 difference:** at most 4.2e-7 relative. f32: at most 3.8e-6, or 1.7e-4 on the small `crval` block.
   - **Re-review repair:** the Jacobian comparison alone did not meet "check_grads passes per mode with recorded, scale-aware tolerances", and it gated a difference the plan says to report. Now `check_grads` runs forward and reverse in mixed (eps 0.3, atol = rtol = 3e-3) and f32 (eps 1, atol = rtol = 1e-2), on every applicable path and both SIP cases. The minimum passing tolerance was measured at 1e-3, or 3e-3 for the f32 AP/BP-only forward path. That outlier was traced to the float32 rounding of ~5300 arcsec outputs divided by 2·eps, while the f64 difference at the same step agrees with AD to 1e-7. The cross-mode comparison is kept as a report with only a 1e-2 divergence guard. 90 tests.
6. **Downstream gate (reprojax).**
   - **A/B-only:** the case strips AP/BP from whichever header carries SIP, in both directions. It asserts the keys are gone and the transform is not the cached full one. It is numerically identical to the full case on this data, because AP/BP only seeds the Newton inverse.
   - **Limits:** overlap counts are frozen exactly at 4673 forward and 1568 reverse. The f32 caps are 6e-4 px forward and 5e-4 px reverse, about twice the measured 3.04e-4 and 2.59e-4. The f32 precision test is max < 6e-4 and median < 1.5e-4 px, replacing 0.05 and 0.01.
   - **Finite-difference tolerance:** the "interpolation kinks" relaxation is replaced by a step sweep showing ordinary finite-difference error. Moving the step from 1e-9 to 1e-7 allows rtol 1e-5, tighter than the original 1e-4.
   - **Cleanup:** the ignored `refine_steps` parameter, the duplicate tests and the stale `TanJax`/refinement docstrings are removed. `benchmarks/benchmark_backends.py` is migrated.
7. **Performance.** The timing table above. In the migrated tree, `benchmark_stage3.py --implementation legacy` now exits with an error that points to a `d60a5b1` checkout.
8. **Docs and packaging.**
   - **User manual:** the first example is fixed, and `Reprojector`, `from_matrix` and the retained-reference workflow are documented. The migration table covers names, precision replacements, Array → `vmap`, both absolute replacements, host references, return types and x64. A compatibility statement is added.
   - **Execution:** all manual examples run, in order, in one fresh process.
   - **Helper signatures:** `absolute_from_offset(offset, *, crval, offset_scale)` and `offset_from_absolute(absolute, *, crval, offset_scale)`. These replace the plan's separate-component spelling, which the operator approved on Nov 4 2026, because keyword-only arguments block a silent swap of `crval` and a scalar offset.
   - **Versions:** mapprojax 0.3.0 (`uv.lock` refreshed offline). reprojax is 0.2.0 and declares `mapprojax>=0.3.0,<0.4`.
   - **DEVELOPER.md:** keyword-only helper call, and gradient guidance through `from_matrix` and the retained-reference workflow.
   - **Historical documents:** `docs/jax_migration.md` and `docs/benchmarks/precision_results.md` are marked as historical.
9. **Commit and pin.** Pending operator authorization. C2 is to be verified once against a `d60a5b1` checkout when its commit is made, and the committed pair retested.

## Artifacts

- **Canonical, closure:**
  - `stage3-c8-{legacy,new}-{cpu,rocm}-*.json` (28 files)
  - `stage3-c8-comparison.json`
  - `stage3-c8-accuracy-{cpu,rocm}.{json,md}`
  - this manifest
- **Canonical, C4:** as listed in `stage3-c4-checkpoint.md`.
- **Canonical, C1:** as listed in `stage3-c4-checkpoint.md`.
- **Superseded:**
  - `stage3-g012-qa.json` is the C5–C8 session's QA report. Its "no blocking regression" verdict is withdrawn by the closure review, and it is kept untracked.
  - The superseded C4-era files are listed in `stage3-c4-checkpoint.md`. They are kept as provenance, not pruned.

## Non-blocking follow-ups

- **Batched leaves:** the constructor accepts batched `crpix`/`cd`/`crval` leaves (`wcs.py`), which reprojax's batched construction uses. This is documented here rather than reverted.
- **Pruning:** superseded artifacts can be removed once nothing cites them.
