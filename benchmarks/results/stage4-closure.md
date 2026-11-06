# Stage 4 closure evidence

## Exact revisions and scope

- C1 mapprojax API/tests/docs: `839ab64`.
- C2 mapprojax benchmark harness and clean-baseline timing set: `6969519a5da08d64517d4d5ae2b576a237b85cbb`.
- R1 reprojax adapter/tests/docs: `14990d3639877a027bbe195091f05f9f9e1572e1`.
- Feature-bearing verified pair: mapprojax C2 `6969519a5da08d64517d4d5ae2b576a237b85cbb` + reprojax R1 `14990d3639877a027bbe195091f05f9f9e1572e1`.
- C3 is this closure/roadmap documentation commit; it does not alter the feature-bearing pair.
- No version bump, push, tag, release, or publication.

## Exact-pair retest

Clean archives were regenerated with `git archive` into `/tmp/mapprojax-stage4-c2` and `/tmp/reprojax-stage4-r1`. Tests imported only those archive source trees, verified package paths, and asserted the actual backend.

- mapprojax CPU: `PYTHONPATH=/tmp/mapprojax-stage4-c2/src:/tmp/mapprojax-stage4-c2/tests JAX_PLATFORMS=cpu /home/jjlee/projects/mapprojax/.venv/bin/pytest -q` — **788 passed, 1 warning, 0 skipped**.
- mapprojax ROCm: same archive paths with `JAX_PLATFORMS=rocm XLA_PYTHON_CLIENT_PREALLOCATE=false` — **788 passed, 1 warning, 0 skipped**.
- reprojax CPU: `PYTHONPATH=/tmp/mapprojax-stage4-c2/src:/tmp/reprojax-stage4-r1/src JAX_PLATFORMS=cpu /home/jjlee/projects/reprojax/.venv/bin/python -m pytest -q` — **88 passed, 2 pre-existing GPU-only skips**.
- reprojax ROCm: same archive paths with `/tmp/.../rocmvenv/bin/python`, `JAX_PLATFORMS=rocm` — **90 passed, 0 skipped**.
- Import audit: `mapprojax.__file__` resolved under `/tmp/mapprojax-stage4-c2/src`, `reprojax.__file__` under `/tmp/reprojax-stage4-r1/src`; devices were `CpuDevice(id=0)` and `RocmDevice(id=0)`.

Environment: mapprojax CPU/ROCm Python 3.12.13, JAX/jaxlib 0.11.1; ROCm plugin `jax-rocm7-plugin`/`jax-rocm7-pjrt` 0.11.1. reprojax CPU Python 3.14.4, JAX/jaxlib 0.12.0.1. The reprojax ROCm gate used the Python 3.12 ROCm environment above.

## Gradient tolerances and sweeps

Chosen test settings remain: f32 kernel `eps=1`, `atol=1e-2`, `rtol=1e-2`; f64 kernel `eps=1e-3`, `atol=1e-6`, `rtol=1e-5`. A short scalar central-difference sweep over pixels, CD and live-crval losses used f32 eps `[0.1, 0.3, 1, 3]` and f64 eps `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2]` on CPU and ROCm. Maximum absolute errors over each sweep were:

| backend | kernel | pixels | CD | live crval |
| --- | --- | ---: | ---: | ---: |
| CPU | f32 | 1.54e-5 | 1.22e-4 | 4.92e-2 |
| ROCm | f32 | 1.53e-5 | 1.22e-4 | 4.92e-2 |
| CPU | f64 | 3.32e-11 | 9.55e-12 | 5.47e-7 |
| ROCm | f64 | 3.32e-11 | 9.55e-12 | 5.47e-7 |

The f32 live-crval floor is recorded separately as approximately `4.93e-2` absolute in this scalar sweep; it is a precision floor, not a tolerance relaxation. The AP/BP counterexample is f32 with `A[2,0]=0.01`, `U=1`, zero B/AP/BP: one-step validity is true and `|du/dAP[2,0]|≈1.92e-4`; two-step sensitivity is strictly smaller. Well-converged sensitivity and source-leg AP/BP control are covered by tests.

## Accuracy and protected rows

`uv run --no-sync python benchmarks/benchmark_accuracy.py` exited 0 on the clean C2 archive. Protected files remained byte-identical and unchanged: `tests/data/accuracy_baselines.json` SHA-256 `aadd23e50b454395a58d34738120b8c366ab984651704aee24dba2c578511de6` (7417 bytes) and `docs/benchmarks/accuracy_results.md` SHA-256 `f168da26278a8fb453cd5ed9d181f81bddc7cf16b9707908767dfb19ab6b6e61` (7183 bytes). Overlap remains **1009/1090 forward** and **320/1090 reverse**; caps are unchanged.

## Reprojax capability equivalence

The exact-pair real SPHEREx 2040²→1201² command was `/home/jjlee/projects/mapprojax/.venv/bin/python /tmp/real_equivalence.py` with `PYTHONPATH=/tmp/mapprojax-stage4-c2/src:/tmp/reprojax-stage4-r1/src`, source NaNs injected at `(0,0)`, `(100,100)`, and `(2039,2039)`, and an independently computed finite/half-open fallback mask. Across order 0/1 and coordinate dtypes f32/f64, image-value, NaN-mask and footprint mismatch counts were all zero.

- Order 0 (both dtypes): canonical values SHA-256 `ba12ea2319bfb72cf69ee43cb32a67db374c9513854d839e6c25f8a63865a13a`; NaN-mask `347e6f0cfb9b8c0cc49a7a57c9de71e112832b6fbf75fa47f64b1da2c88f1d5e`; footprint `c0aaa1cf1c98b5d2138e255fe81585f0e3adee86cdc6ca8d716ee82d1bdd7144`.
- Order 1 (both dtypes): canonical values SHA-256 `576006dc06baa438d1c3c16a9d72ededfa977ccc1d8650c515c11c8376752de5`; NaN-mask `f665220ab2fbbd9a646d938e171b7c9b16e416586af4da8f7fc9452237ff3609`; footprint `b418bcd37acc298b24825866b8a760ef92bcf9cb8b2ea733f2676951c707ddd4`.

## Timing provenance

The committed timing set is the clean `c82e76b` baseline versus C1/C2 candidate, measured in the same full-mode session with unchanged rules. Files are `benchmarks/results/stage4-{baseline,new,comparison}-{cpu,rocm}-{pair,pixel_map}-{f32,f64}-r2.json`; all eight comparison records and all four warm/lowering/compile/cold rules pass. Baseline package identity is `5413074f147ede139594ad6f51bfdfa508f3fa95c52853ba6119c850f60428f7`; candidate package identity is `74ae95d290d017d8eb6ff77d7d9ac6ec727fdc60835a57ef507f8f792e879017`; benchmark-script identity is `032de873e55f6f73773d5c5066c8450e2b15f78c0da7672595d1181e29092de9` on both sides. Per-side provenance is recorded but intentionally not a matching field. The earlier same-tree timing set was superseded and is not committed; the 26 old `stage4-*.json` files remain untracked, including the size-8 smoke files.

## Closure

Stage 4 is complete on the exact feature-bearing C2/R1 pair. This record hashes no self-referential frozen receipt: runtime source-hash receipts previously used by review covered the full working-tree change set and were superseded by the exact committed archives above. Stage 5, release, packaging policy and publication remain separate work.

## Correction and re-pin — Nov 6 2026

The committee rejected the preceding evidence sections, not the feature-bearing
source. This append-only correction supersedes the preceding gradient-sweep,
real-pair-equivalence and pinned-pair claims without rewriting the frozen
record above.

- C4 mapprojax evidence scripts and edge test: `74bc5f3fb9d53235154eb4d7977c76db66a6350b`.
- R2 reprojax equivalence tests: `0483edba197317357cd1856c84141403a2d93400`.
- Corrected feature-bearing pair: C4 `74bc5f3fb9d53235154eb4d7977c76db66a6350b` + R2 `0483edba197317357cd1856c84141403a2d93400`.
- C5 is documentation-only and cites this C4/R2 pair; it does not cite itself.

### Scaled gradient sweep

`benchmarks/sweep_reprojector_gradients.py` sweeps the committed
`_scaled_reprojection`/`_live_crval_reprojection` equivalents for pixels,
source/target CRPIX and CD, source/target A/B, target AP/BP, `m_rel`, and both
live CRVALs. The selected settings are f32 `eps=1`, `atol=1e-2`, `rtol=1e-2`
and f64 `eps=1e-3`, `atol=1e-6`, `rtol=1e-5`. Plateau values below are
`min-max` absolute FD/JVP error over the three-step neighborhood of the
selected dimensionless step; `sel` is the error at the selected step. The
exact JSON outputs were generated by:

```text
PYTHONPATH=/tmp/mapprojax-stage4-c4/src JAX_PLATFORMS=cpu \
  /home/jjlee/projects/mapprojax/.venv/bin/python \
  /tmp/mapprojax-stage4-c4/benchmarks/sweep_reprojector_gradients.py
PYTHONPATH=/tmp/mapprojax-stage4-c4/src JAX_PLATFORMS=rocm \
  /tmp/.../rocmvenv/bin/python \
  /tmp/mapprojax-stage4-c4/benchmarks/sweep_reprojector_gradients.py
```

| family | CPU f32 sel / plateau | ROCm f32 sel / plateau | f64 sel / plateau (both backends) |
| --- | ---: | ---: | ---: |
| pixels | 3.07e-6 / 1.79e-6–3.07e-6 | 3.07e-6 / 1.79e-6–3.07e-6 | 8.84e-12 / 1.74e-12–8.84e-12 |
| src_crpix | 3.58e-6 / 2.38e-7–3.58e-6 | 3.58e-6 / 2.38e-7–3.58e-6 | 1.72e-12 / 1.72e-12–1.72e-12 |
| tgt_crpix | 0 / 0–0 | 0 / 0–0 | 4.77e-12 / 3.80e-14–4.77e-12 |
| src_cd | 2.03e-6 / 5.07e-7–2.03e-6 | 2.03e-6 / 5.07e-7–2.03e-6 | 1.00e-11 / 2.93e-12–1.00e-11 |
| tgt_cd | 4.67e-5 / 4.67e-5–4.32e-4 | 4.67e-5 / 4.67e-5–4.32e-4 | 5.28e-11 / 5.28e-11–4.34e-10 |
| src_a | 1.76e-6 / 4.92e-7–1.76e-6 | 1.76e-6 / 4.92e-7–1.76e-6 | 5.45e-12 / 1.66e-12–5.45e-12 |
| src_b | 1.35e-6 / 1.35e-6–2.47e-6 | 1.36e-6 / 1.36e-6–2.46e-6 | 7.19e-12 / 8.12e-14–7.19e-12 |
| tgt_a | 6.11e-5 / 6.11e-5–5.52e-4 | 6.11e-5 / 6.11e-5–5.52e-4 | 6.07e-11 / 6.07e-11–5.49e-10 |
| tgt_b | 5.80e-6 / 5.80e-6–4.90e-5 | 5.80e-6 / 5.80e-6–4.90e-5 | 1.44e-11 / 1.44e-11–4.99e-11 |
| tgt_ap | 2.04e-21 / 2.04e-21–2.04e-21 | 1.84e-21 / 1.84e-21–1.84e-21 | 6.01e-34 / 6.01e-34–6.01e-34 |
| tgt_bp | 2.92e-22 / 2.92e-22–2.92e-22 | 1.82e-22 / 1.82e-22–1.82e-22 | 5.21e-35 / 5.21e-35–5.21e-35 |
| m_rel | 2.55e-6 / 1.26e-6–2.55e-6 | 2.55e-6 / 1.26e-6–2.55e-6 | 3.53e-12 / 1.21e-12–3.53e-12 |
| src_crval | 2.86e-6 / 2.86e-6–1.18e-5 | 2.86e-6 / 2.86e-6–1.18e-5 | 1.68e-10 / 1.68e-10–9.38e-10 |
| tgt_crval | 2.86e-6 / 2.86e-6–1.18e-5 | 2.86e-6 / 2.86e-6–1.18e-5 | 2.61e-10 / 2.61e-10–2.87e-10 |

The f32 live-CRVAL result is recorded as a scaled-loss FD/JVP error
(approximately `2.86e-6` at the selected step), not as a precision floor.
The f32 AP/BP convergence counterexample and source-leg control remain covered
by the committed gradient tests.

### Corrected reprojax equivalence

`benchmarks/compare_reprojax_pixel_map.py` is now the regeneration command.
It uses the real 2040² SPHEREx input and 1201² template, injects source NaNs at
`(0,0)`, `(100,100)` and `(2039,2039)`, runs both matrix kernels (`f32`, `f64`),
both input-grid coordinate dtypes (`float32`, `float64`) and both interpolation
orders, and compares the mapprojax capability against a plain delegate that
returns `transform(x, y)` unmodified. Bounds are checked independently from
the returned validity mask. The exact command was:

```text
PYTHONPATH=/tmp/mapprojax-stage4-c4/src:/tmp/reprojax-stage4-r2/src \
  JAX_PLATFORMS=cpu \
  /home/jjlee/projects/mapprojax/.venv/bin/python \
  /tmp/mapprojax-stage4-c4/benchmarks/compare_reprojax_pixel_map.py \
  --data-dir /tmp/reprojax-stage4-r2/tests/data \
  --output /tmp/stage4-real-equivalence-c4-r2.json
```

All 8 cases had zero value mismatches, zero NaN-mask mismatches and zero
footprint mismatches. The corrected report records the per-case canonical
value, NaN-mask and footprint digests; the run used `backend=cpu` and
`RocmDevice(id=0)` was independently verified by the exact-pair ROCm suite.

### Corrected exact-pair retest

Clean archives `/tmp/mapprojax-stage4-c4` and `/tmp/reprojax-stage4-r2`
imported only C4/R2 sources and passed package-path/backend assertions.
mapprojax passed **788 CPU** and **788 ROCm**. Reprojax passed **88 plus 2
pre-existing CPU skips** and **90 ROCm**. No timing rerun was needed.

## Evidence durability correction — Nov 6 2026

The plateau window implementation and generated reports are now durable. The
previous C4/C5 evidence remains frozen; this append supersedes only the
plateau-window wording and report-location claims. The sweep now selects the
three records centered on the selected step by index, so f32 includes
`0.3, 1, 3` and f64 includes `3e-4, 1e-3, 3e-3`.

- CPU sweep report: `benchmarks/results/stage4-gradient-sweep-cpu-c6.json`.
- ROCm sweep report: `benchmarks/results/stage4-gradient-sweep-rocm-c6.json`.
- Real-pair report: `benchmarks/results/stage4-real-equivalence-c6.json`.
- Regeneration scripts: `benchmarks/sweep_reprojector_gradients.py` and
  `benchmarks/compare_reprojax_pixel_map.py`.
- The corrected reports contain all 14 parameter families × 2 kernels × 2
  backends and all 8 real-pair kernel/coordinate-dtype/order cases. Every
  real-pair mismatch count remains zero.

## Coverage correction and re-pin — Nov 6 2026, round 2

The committee identified two remaining coverage gaps; production code and
timing were unchanged. The C5 plateau table is superseded by the committed C7
JSON reports. In particular, the C5 table's earlier two-point window was
described as a three-step neighborhood, and its combined f64 column obscured a
ROCm/source-CD maximum difference.

- C7 mapprojax script and reports: `300f11b53264c6e64e393a03b8292fffdb7b9b13`.
- R3 reprojax tests: `ef1cb31e2325acc6e23085ceafc02285a7a71cd4`.
- Corrected feature-bearing pair: C7 + R3.
- C6 reports remain frozen; C7 reports are
  `benchmarks/results/stage4-gradient-sweep-cpu-c7.json` and
  `benchmarks/results/stage4-gradient-sweep-rocm-c7.json`.

The C7 script now covers 18 families, including `tgt_ap_only`,
`tgt_bp_only`, `src_crval_nosip` and `tgt_crval_nosip`, across f32/f64 on CPU
and ROCm. The AP/BP-only f32 selected-step derivatives are approximately
`0.2851813` and `0.0855544`, respectively; the SIP-free live-CRVAL f32
selected-step FD/JVP error is approximately `1.07e-6`. C7's plateau is the
three records centered by index on the selected step (`[0.3,1,3]` for f32 and
`[3e-4,1e-3,3e-3]` for f64), and the full per-family values are durable in
the two JSON reports.

R3 changes the synthetic equivalence fixture to assert geometric invalidity
from a target header 150 degrees away, while retaining independent bounds,
plain delegates, source NaNs, both y-axis neighboring floats, both kernels,
both coordinate dtypes and both interpolation orders. The C7/R3 exact archive
retest imported only `/tmp/mapprojax-stage4-c7` and
`/tmp/reprojax-stage4-r3`: mapprojax passed **788 CPU** and **788 ROCm**;
reprojax passed **88 plus 2 pre-existing CPU skips** and **90 ROCm**; package
paths and actual devices were asserted.
