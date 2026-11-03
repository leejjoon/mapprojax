# Stage 3 C4 checkpoint manifest

Nov 3 2026. This is the evidence packet for the go/no-go decision before C5. It covers the C4 repairs that the committee (Fable, Astra) made blocking on the first packet.

## Identity

| Item | Value |
| --- | --- |
| Plan | `pending-approval.md` revision 4, sha256 `f6dff8eef8672ab1…` |
| Committed base | `338f47e` (C3/C4), on `b32fdc1` (C1), on `5b1c955` |
| Candidate | `338f47e` plus the uncommitted C4 repairs, at the file hashes below |
| `src/mapprojax/wcs.py` | `44263ced22e00d30…` |
| `tests/test_wcs_new.py` | `029a8abd8e53e741…` |
| `tests/accuracy_reference.py` | `8dda9ed6719edf00…` (unchanged since `338f47e`) |
| `benchmarks/benchmark_stage3.py` | `2b67a13833a5ae5e…` |
| `benchmarks/compare_stage3.py` | `96dbfcbea05aaa5b…` |
| `tests/data/accuracy_baselines.json` | `aadd23e50b454395…` (frozen; unchanged since C1) |
| `docs/benchmarks/accuracy_results.md` | `a6d3fbd9a19e0c6d…` (unchanged) |
| Legacy sources and fixtures | All 20 files in `stage3-c1-provenance.json` match their C1 hashes |

## Environment

- Measured interpreter: Python 3.12.13, the project `.venv`. `stage3-c1-provenance.json` records `"python": "3.14.4"`, but that is the interpreter that ran the collector script, not the measured one. Every timing artifact records 3.12.13.
- Packages: jax 0.11.1, jaxlib 0.11.1, jax-rocm7-plugin and jax-rocm7-pjrt 0.11.1, NumPy 2.5.1, Astropy 8.0.2.
- CPU runs used `JAX_PLATFORMS=cpu`. ROCm runs used `JAX_PLATFORMS=rocm` on `RocmDevice(id=0)` (Strix Halo APU, PJRT C API, platform `rocm 70253211`), with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Results

| Check | Command | Result |
| --- | --- | --- |
| Full suite, CPU | `JAX_PLATFORMS=cpu uv run pytest -q` | 647 passed |
| Full suite, ROCm | `JAX_PLATFORMS=rocm uv run pytest -q` | 647 passed |
| New-class tests | `uv run pytest tests/test_wcs_new.py` | 147 passed. On `338f47e` the same file fails 3: two integer-`crval` cases and `[f32, int]` |
| Accuracy, CPU | `benchmark_accuracy.py --json … --markdown …` | Exit 0. The Markdown is byte-identical to `docs/benchmarks/accuracy_results.md`. The JSON equals both the C1 baseline (`stage3-c1-accuracy.json`) and the C4 run |
| Accuracy, ROCm | same, `JAX_PLATFORMS=rocm` | Exit 0. JSON and Markdown are byte-identical to the C4 ROCm run, which Fable had matched against the legacy classes in a `5b1c955` worktree |
| `jax`/`jax_mixed` rows from the new classes | the CPU and ROCm runs above | 14/14 and 14/14, no differences |
| Timing gate | `compare_stage3.py --output stage3-c4r-comparison.json` over 10 pairs | Exit 0; all 10 pairs pass all four rules |

Collection grew from 500 tests (C1) to 528 (`338f47e`) to 647 (this candidate). No tests were removed, skipped or marked `xfail`.

## Timing (C4 repair run, `stage3-c4r-*`)

Baseline and candidate were re-measured back to back with the same harness, so both sides record lowering time.

Re-measuring did not loosen any limit:
- **Original budgets.** For the 8 workloads whose call boundary is unchanged, the new candidates also pass the original frozen budgets from the C1 and first-C4 baselines on warm, compile and cold: 24 of 24 checks (Astra). The tightest is CPU batch at 4.328 ms against an original budget of 4.396 ms.
- **New baselines.** Two needed a baseline for the first time: order 4, because its call boundary was corrected, and lowering, which was not recorded before.
- **Residual noise.** The re-measured CPU batch baseline still has a 9.9% process-median CV. The pass against the original budget makes the decision independent of this baseline. The limits are the frozen ones: warm `max(1.10·B, B+0.05 ms)`; lowering, compile and cold `max(1.20·B, B+50 ms)`. The comparator checks four timing metrics (warm, lowering, compile, cold); the plan's separate no-recompilation rule is the trace-count test below. Values are medians; warm is the median of 3 process medians of 30 repetitions; lowering, compile and cold come from 5 fresh processes with the persistent cache disabled.

| Backend | Workload | Call | Warm ms (B → N) | Lowering ms | Compile ms | Cold ms |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | batch 256×4096, mixed | `unproj_offset` | 4.04 → 4.33 | 31.9 → 31.6 | 69.4 → 66.6 | 102 → 108 |
| CPU | SIP order 4, 1201², evaluation only | `unproj_offset` | 6.68 → 6.44 | 26.7 → 26.6 | 95.2 → 86.7 | 122 → 126 |
| CPU | SPHEREx order 3, 2040², steps 2 | `unproj_offset` + `proj_offset` | 72.7 → 67.2 | 80.4 → 78.6 | 290 → 260 | 441 → 402 |
| CPU | same, steps 3 | same | 86.5 → 80.3 | 92.1 → 88.9 | 338 → 305 | 516 → 465 |
| CPU | same, steps 5 | same | 114.5 → 103.7 | 162 → 110 | 418 → 377 | 708 → 587 |
| ROCm | batch 256×4096, mixed | `unproj_offset` | 0.66 → 0.64 | 32.6 → 31.9 | 97.4 → 99.0 | 133 → 134 |
| ROCm | SIP order 4, 1201², evaluation only | `unproj_offset` | 0.85 → 0.87 | 27.3 → 26.7 | 164 → 158 | 207 → 203 |
| ROCm | SPHEREx order 3, 2040², steps 2 | `unproj_offset` + `proj_offset` | 16.1 → 6.28 | 80.9 → 80.2 | 1017 → 917 | 1095 → 996 |
| ROCm | same, steps 3 | same | 18.9 → 6.35 | 95.3 → 90.8 | 1163 → 1095 | 1286 → 1215 |
| ROCm | same, steps 5 | same | 24.5 → 6.50 | 163 → 114 | 1520 → 1399 | 1699 → 1533 |

- **Order 4 is now evaluation only.** Its synthetic coefficients have no convergent inverse over the grid: the old workload's inverse was finite at about 5 of 1089 samples. The order-3 rows time evaluation plus inverse on the real SPHEREx header.
- **Rule 4 (no recompilation on changed numerical leaves)** is measured by `test_numeric_leaf_changes_reuse_the_compiled_function`. It counts traces for both projections in all three modes, with a changed-`inverse_steps` positive control. This replaces the old jaxpr-string comparison.
- **Pair workload.** A new-class pair candidate does not exist until C5 moves `Reprojector`. The legacy pair files are controls for C5.

### Reading the SPHEREx speedup

The large ROCm warm speedup on the order-3 rows is **not** evidence about the restructure. It comes from a precision change that the plan's casting table requires.

Legacy `TanJaxMixed.proj_offset` multiplies by the fp64 reference cosine, which promotes the whole native pipeline to fp64, SIP inverse included (`base.py:429`). The new `mixed` mode casts the native vector to f32 before the pixel side (`wcs.py:258`), so the SIP inverse runs in f32. Measured on the timing path (65×65 grid over the SPHEREx detector, CPU):

| Implementation | jaxpr equations producing f64 | Offsets dtype | Pixel output dtype | Finite |
| --- | --- | --- | --- | --- |
| legacy `TanJaxMixed` | 1714 | float64 | float64 | 4225/4225 |
| new `Tan(precision="mixed")` | 143 | float64 | float32 | 4225/4225 |

The counts include nested jaxprs and count equations with an f64 output. Astra reproduced 1714 vs 143; Fable got 1714 vs 137, and 3709 vs 234 with an earlier method. The errors below are the maximum per-axis |Δx| or |Δy|:

- **Legacy vs new:** the pixel outputs differ by at most 2.31e-4 px.
- **Round trip:** the error is 2.44e-4 px new and 1.94e-4 px legacy. The new figure is two f32 spacings at 2040, where one spacing is 1.22e-4 px.
- **Radial (Fable):** 3.45e-4 px new, 2.22e-4 px legacy, 3.2e-4 px legacy to new. The rule passes, but B here is a heavier, fp64 computation. At C5 the new f32 `Reprojector` is compared against the legacy f32 pair control. That is a complete `Reprojector` comparison, not an isolated SIP-kernel benchmark.

## C4 repairs in this candidate

1. **`crval` normalization** (`wcs.py`, `_normalize_crval`). A sequence is assembled, then rejected if assembly changed any component's own floating dtype. So `[np.float32(x), 30]`, `[f32, 30.0]`, `[f32, f64]` and `[30.0, f32]` are rejected. Integer-only sequences are promoted to float64. `[f32, f32]`, Python floats and JAX/tracer sequences (assembled with `jnp.stack`) are kept. A traced f32 value next to a Python float is rejected too, the same rule as on the host. Normalization runs only in the constructor.
2. **New-class tests** (`tests/test_wcs_new.py`), covering:
   - accepted and rejected `crval` cases, host and traced;
   - header-list vs array treedef, including stacking;
   - CD rejection after `with_precision`;
   - reconstruction with `__init__` patched to raise (`tree_map`, `eval_shape`, `jit`);
   - both mixed rotation-boundary casts at 0.05″/px;
   - masked-loss gradients for Tan and Sin in every mode and applicable method, with and without SIP (80 cases);
   - valid-sample gradients unchanged by invalid neighbours, per mode;
   - the rule-4 trace count;
   - JSON round trips with no SIP, A/B-only and AP/BP-seeded, checking outputs;
   - a fresh-process x64-off test for `f32`, covering offsets, absolute guards, `mixed`/`f64` refusal, the NumPy float64 reference on JSON reload, and `crval_dtype` float32 after an x64-off `jit`;
   - fresh-process post-toggle guards for `mixed` and `f64`.
3. **Harness.** Order 4 is evaluation only; lowering time is recorded and held to the compile limit; each worker's temporary compilation-cache directory is removed; a `call` field is part of the matched metadata.
4. **Comparator.** It accepts several pairs and `--output`, exits 1 on any failed rule and 2 on an unmatched pair.

## Downstream (C2)

reprojax is now present at `~/projects/reprojax`, clean at `40221bf` ("transform: add matrix_precision="f32" using mapprojax's Reprojector"). That is the proposed C2 starting pin. C2 has not started. By the user's decision it is not required before C5; it is required before C7.

## Canonical artifacts

- **C1 baseline:**
  - `stage3-c1-provenance.json`
  - `stage3-c1-accuracy.{json,md}`
  - `stage3-c1-noise-*.json` and `stage3-c1-noise-investigation.json`: the noise investigation, which did not change thresholds
  - `stage3-legacy-full-*-pair-f32.json` and `stage3-legacy-full-*-pair-f64.json`: the legacy pair controls for C5
- **C4 checkpoint:**
  - `stage3-c4r-{legacy,new}-{cpu,rocm}-*.json`
  - `stage3-c4r-comparison.json`
  - `stage3-c4r-accuracy-{cpu,rocm}.{json,md}`
  - `stage3-c4-accuracy-rocm.{json,md}`
  - this manifest
- **Superseded:** the first C4 packet. Its legacy baselines remain authoritative for the original-budget check above. Its timing pairs lack lowering time, and its order-4 pair timed a non-convergent inverse. Its QA report's "no blockers" is withdrawn. These files are kept as provenance only:
  - `stage3-c4-comparison.json`, `stage3-c4-qa.json`, `stage3-c4-cleaner.txt`
  - `stage3-legacy-full-*` and `stage3-new-full-*`, except the pair controls above
  - `stage3-c4-full-*`, `stage3-c4-*-smoke.json`, `stage3-c4-timing-*.json`, `stage3-c4-batch-cpu.json`, `stage3-c1-full-*`, `stage3-c1-baseline-smoke.json`, `stage3-c1-timing-*.json`
  - Pruning them is carried into C5.

## Open items

- **Carried into C5:**
  - forward and reverse `check_grads` on the new classes, including the inverse SIP and AP/BP;
  - f32 `Reprojector` timing against the legacy f32 pair control;
  - dead code at `wcs.py` (the unreachable f32 branch in `r_matrix`) and the duplicated `inverse_steps` aux entry;
  - stale docstrings and `_CLASS_NAMES` in `tests/accuracy_reference.py`;
  - pruning and indexing `benchmarks/results`.
- **Not committed.** The C4 repairs are uncommitted; committing them needs the operator's go-ahead.

## Committee re-review (Nov 4 2026)

Fable and Astra each verified the repaired candidate independently, and both say **GO for C5**.
- **Fable** reproduced the CPU and ROCm suites (647 each), the 3-test failure on `338f47e`, the accuracy identity, and the four metrics from raw samples.
- **Astra** reproduced the CPU suite, the accuracy identity, the comparator exit codes, the hashes, and the original-budget check. ROCm is unreachable from its sandbox, so the ROCm suite rests on the operator's run and Fable's.

The corrections above are theirs.
