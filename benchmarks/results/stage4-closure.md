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
