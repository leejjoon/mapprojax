# Avoiding fp64 in mapprojax: Report and Implementation Plan

**Date:** 2026-08-04
**Status:** Proposal — research and numerical validation complete, implementation not started.

This document has three parts:

1. **Problem analysis** — why the current pipeline needs fp64 at all (summary of the
   analysis in `DEVELOPER.md` §4).
2. **Research report** — a survey of techniques for avoiding native fp64 while keeping
   sub-pixel accuracy, with numerical validation performed against this codebase.
3. **Implementation plan** — concrete, phased changes to mapprojax.

---

## Part 1 — Problem Analysis (recap)

The WCS pipeline mixes two regimes with very different precision demands:

| Regime | Quantities | Magnitude | Required absolute accuracy | fp32 OK? |
|---|---|---|---|---|
| **Absolute** | RA/Dec, celestial unit vectors, `crval`, rotation matrix | O(1) (unit vectors), 0–2π (angles) | ~1e-8–1e-9 rad (0.001–0.01″) | **No** — needs relative precision below fp32's ε ≈ 1.2e-7 |
| **Relative** | native-frame offsets, plane coords, pixel offsets from `crpix`, SIP terms | field-of-view-sized (≤1e-2 rad), ≤1e4 px | sub-pixel | **Yes** — small values carry proportionally small absolute error |

The damage is injected in exactly one step: the celestial↔native rotation
(`apply_rotation` / `apply_rotation_transpose` in `src/mapprojax/utils.py`), where O(1)
products catastrophically cancel into tiny field offsets. Measured on a TAN round-trip
(200×200 grid, 3.6″/px):

| Configuration | Max round-trip error |
|---|---|
| all fp64 | 2.7e-11 px |
| rotation fp64, everything else fp32 (current `Mixed` classes) | 7.6e-6 px |
| rotation fp32, everything else fp64 | 1.076e-2 px |
| all fp32 | 1.077e-2 px |
| all fp64 but RA/Dec quantized to fp32 | 1.7e-3 px |

Casting *only* the rotation to fp32 reproduces essentially 100% of the all-fp32 error.
At a modern 0.1″/px plate scale the all-fp32 error scales to ~0.4 px — unusable.
The question this document answers: **can we remove that last fp64 dependency?**

---

## Part 2 — Research Report

### Method and confidence labels

Findings come from a fan-out web-research run (5 search angles, 21 sources, 102
extracted claims, top 25 adversarially verified by 3-vote panels) plus numerical
experiments run directly against this repository. Claims are labeled:

- **✓** survived adversarial verification (3-0 or 2-1),
- **✗** refuted (≥2/3 refute votes),
- **~** plausible but unverified (the verification/synthesis phases were partially cut
  short by a usage limit; unverified claims are flagged, not silently kept),
- **[measured]** validated numerically in this repo.

### Verdict

**fp64 can be removed from the per-pixel path, and the best route is algorithmic, not
arithmetic.** The fp64 requirement exists only because the pipeline routes through
absolute celestial coordinates between the source and target frames. Three families of
techniques were evaluated:

### 2.1 Algorithmic reformulation — compute offsets, never absolutes (RECOMMENDED)

The cancellation happens only when a rotation converts O(1) absolute unit vectors into
tiny field offsets. Astrometry, geodesy, and game rendering all converged on the same
fix: **handle the reference in high precision once, at setup; do all bulk work on small
offsets in low precision.**

**Prior art:**

- **dfreproject (GPU WCS reprojection, PyTorch)** ✓ — computes
  `ra = ra₀ + atan2(small, small)`: the RA *offset* from the projection center comes
  directly from spherical trig on field-sized quantities and is added to the reference;
  absolute RA is never recovered from a rotated 3-vector.
  <https://dfreproject.readthedocs.io/en/latest/methodology.html>
  Caveat ✓: the source claims fidelity to wcslib but gives no quantitative fp32
  accuracy figure — it is prior art for the *formulation*, not proof of accuracy.
- **SLALIB** ✓ — ships a single-precision tangent-plane routine (`sla_S2TP`) alongside
  the double-precision one: fp32 was considered legitimate for tangent-plane work once
  coordinates are relative to the tangent point.
  <https://irtfweb.ifa.hawaii.edu/~tcs3/tcs3/Misc/slalib_html/node226.html>
  (The stronger claim that SLALIB works "entirely in offsets" was **refuted ✗** — its
  inputs are still absolute RA/Dec.)
- **wcslib** ~ — works in angle space (Euler angles, `phi/theta ↔ lng/lat`) rather than
  3-vectors and provides offset primitives (`sphdpa`/`sphpad`), but every spherical
  routine is fp64-only. Formulation ideas, not an fp32 precedent.
  <https://www.atnf.csiro.au/computing/software/wcs/wcslib/sph_8h.html>
- **Gaia AGIS / SMOK** ~ (votes lost to quota; mathematically sound) — attaches a local
  tangent basis (normal triad `[p q r]`) per source, works in dimensionless offsets of
  order ≤1e-4, and replaces near-identity rotations with the differential form
  `S₁ ≈ S₀ + φ × S₀`. Note Gaia itself runs plain fp64 — even μas astrometry needed no
  exotic arithmetic *given* fp64.
  <https://www.aanda.org/articles/aa/pdf/2012/02/aa17905-11.pdf>,
  <https://arxiv.org/pdf/1407.4025>
- **Graphics / geodesy** ✓ — the identical large-world problem (Earth-sized fp32
  coordinates jitter at meter scale) is solved by relative-to-center / camera-relative
  rendering: subtract a reference on the host in high precision, render offsets in fp32
  (Cesium RTC, Unity HDRP camera-relative rendering, GPS ECEF→local-tangent
  `x_t = R·(x_e − x₀)`).
- **Trig identities (haversine-style)** — `sin²(θ/2)` avoids the `1 − cos θ`
  cancellation for small separations ✓, but the identity *relocates* ill-conditioning
  (unstable near antipodes) rather than removing it ✓ — any adopted identity must be
  validated over the full domain.

**Numerical validation in this repo [measured]:**

For src→tgt reprojection, compose the relative rotation
`M_rel = M_tgt · M_srcᵀ` **in fp64 once at setup** (host-side, 9 scalars), then run the
entire per-pixel path in fp32:
`pixel → native (fp32) → v_tgt = M_rel·v_src (fp32) → native → pixel (fp32)`.
Measured against the all-fp64 pipeline (TAN→SIN, 200×200 grid, 3.6″/px):

| Field-center separation | fp32 relative-rotation error | current fp32 absolute path |
|---|---|---|
| 0.36° | 8.5e-5 px | ~1.1e-2 px |
| 1° | 2.8e-4 px | — |
| 5° | 1.4e-3 px | — |
| 20° | 5.9e-3 px | — |

Why no cancellation: when fields overlap, `M_rel` is close to the identity — its
off-diagonal entries are themselves field-separation-sized, so the small outputs are
sums of *small* products rather than differences of O(1) products. The differential
split `v_tgt = v_src + E·v_src` with `E = M_rel − I` gives identical accuracy in fp32;
the plain composed matrix already suffices.

Error growth is ∝ separation (fp32 quantization of the now-larger matrix entries).
Scaled to a 0.1″/px instrument, keep separations under ~1–2°, or store `M_rel` as a
hi/lo fp32 pair (`M_hi·v + M_lo·v`) to eliminate the entry-quantization term.

### 2.2 Double-float (df64) / compensated arithmetic — viable fallback

For APIs that must *output absolute* RA/Dec without global fp64:

- df64 (unevaluated pair of fp32) provides **~48-bit mantissa, ~3.6e-15 relative
  precision** — ample headroom over the ~1e-8 requirement ✓
  (<https://andrewthall.org/papers/df64_qf128.pdf>).
- Basic ops cost a few ulps; **transcendentals degrade badly** (cos up to 241 ulp) ✓ —
  but the rotation step needs only add/multiply (the matrix is precomputed in fp64 on
  the host and stored as hi/lo fp32 pairs), so the trig caveat does not bite.
- **JAX-specific risk:** `jax.experimental.doubledouble` was removed because XLA
  fusion/reassociation can break the error-free-transform assumptions double-float
  depends on (<https://github.com/jax-ml/jax/discussions/17870> — forum source ~).
  However, the claim that compensated arithmetic is broken in JAX today was
  **refuted ✗**: TFP's `reduce_kahan_sum` works on the JAX substrate ✓
  (<https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/math/reduce_kahan_sum>).
  Compensated code is practical, but must be guarded by a round-trip accuracy test in
  CI to catch compiler regressions.
- Cost estimate: ~10–20 fp32 ops per df64 op → ~100–200 fp32 flops for the 3×3 rotation
  per pixel. Wins on consumer GPUs (native fp64 at 1/32–1/64 rate); on CPU or
  A100/H100 (fp64 at 1/2 rate), native fp64 is simpler and likely faster.

### 2.3 Tensor-core emulation (bf16x9, Ozaki) — NOT applicable

- bf16x9 targets **fp32-level accuracy, not fp64** ✓ — insufficient alone
  (<https://arxiv.org/html/2605.16617>).
- cuBLAS fp64 emulation (Ozaki fixed-point on INT8 tensor cores; dynamic mantissa
  control matching-or-beating native fp64, falling back to native above 79 bits ✓) is
  exposed **only as GEMM math modes** ✓ and is **significantly slower than native DGEMM
  at small problem sizes** ✓ (<https://docs.nvidia.com/cuda/cublas/>,
  <https://arxiv.org/pdf/2508.00441>). A per-pixel 3×3 rotation is a memory-bound k=3
  operation that cannot amortize slicing overhead, and no JAX/XLA exposure exists.

### 2.4 Fixed-point / integer angles — not recommended

int32 spanning 2π gives ~1.5e-9 rad (~0.3 mas) resolution (marginal); int64 is trivially
sufficient. But no GPU/JAX prior art survived verification, and every transcendental
would force conversion back to float. Low confidence; dropped.

---

## Part 3 — Implementation Plan

### Phase 1 — Fused relative-rotation reprojection (the big win)

Removes fp64 from the per-pixel path for the reprojection use case (what
`reproject_sip.py` and `adaptive.py` actually do).

1. **`src/mapprojax/utils.py`**: add `compose_relative_rotation(wcs_src, wcs_tgt)`
   returning `M_rel = M_tgt @ M_src.T` computed in fp64 (NumPy, host-side — this runs
   once at setup, not per pixel).
2. **New module `src/mapprojax/reproject.py`**: a `Reprojector` pytree class holding
   `(wcs_src_f32_params, wcs_tgt_f32_params, M_rel)` with a single method
   `pix_to_pix(x, y)`:
   `pix_to_native (fp32) → apply_rotation(M_rel) (fp32) → native_to_pix (fp32)`.
   Reuses the existing `pix_to_native` / `native_to_pix` (which already never touch
   absolute coordinates) and the existing projection mixins; SIP composes unchanged.
3. **Precision option**: `Reprojector(..., matrix_precision="f32" | "f32x2" | "f64")`.
   - `f32`: plain fp32 `M_rel` (default; validated above).
   - `f32x2`: hi/lo split `M_hi + M_lo`, apply as `M_hi·v + M_lo·v` — removes matrix
     entry-quantization error for large separations / fine plate scales at ~2× rotation
     cost (still pure fp32 hardware ops).
   - `f64`: current behavior, for A100/H100/CPU where native fp64 is cheap.
4. **Tests (`tests/test_reproject_relative.py`)**: parametrize over
   TAN→TAN, TAN→SIN, SIN→TAN, with/without SIP, separations {0.3°, 1°, 5°, 20°},
   plate scales {3.6″, 0.1″}. Assert error vs. the fp64 absolute-path reference:
   ≤1e-3 px for overlapping-field cases at 3.6″/px, documented looser bounds for the
   extreme combinations; assert `f32x2` beats `f32` at 20°/0.1″.
5. **Wire in**: switch `reproject_sip.py`, `adaptive.py`, and the benchmark scripts to
   the `Reprojector` path; keep the old two-step path available.

**Acceptance:** reprojection accuracy ≤1e-3 px vs fp64 reference for overlapping fields
without `jax_enable_x64`, and no fp64 ops in the jaxpr of `pix_to_pix` (assert via
`jax.make_jaxpr` dtype inspection in a test).

### Phase 2 — Offset-form `unproj` (dfreproject-style absolute output)

For callers that need absolute sky coordinates without global fp64:

1. Add `WCSBase.unproj_offset(x, y)` returning small quantities
   `(Δα·cos δ₀, Δδ)` (native-frame spherical offsets from `crval`), computed entirely
   in fp32 via spherical trig of field-sized values (dfreproject formulas).
2. Add `unproj_absolute(x, y)` = fp64 epilogue `(ra₀ + Δα, dec₀ + Δδ)` — two adds and a
   `% 2π` in fp64, negligible cost, no fp64 in the bulk math.
3. Mirror for `proj`: accept `(Δα·cos δ₀, Δδ)` offsets relative to a stated reference
   so a catalog can be recentered once in fp64 and projected many times in fp32.
4. Tests: round-trip `unproj_offset → proj_offset` in pure fp32 vs fp64 reference,
   including RA near 0/2π wraparound and poles (where the offset parameterization must
   be validated — the haversine lesson: reformulations relocate ill-conditioning).

### Phase 3 (optional) — df64 rotation variant

Only if absolute-coordinate output is needed on fp64-crippled GPUs and Phase 2's
formulation proves insufficient somewhere:

1. `src/mapprojax/df64.py`: minimal two-float type (TwoSum, TwoProd via FMA, add, mul)
   — no transcendentals needed; the rotation matrix is precomputed in fp64 and split
   into hi/lo fp32 pairs at setup.
2. `MixedPrecisionDF64Mixin`: identical to today's `Mixed` classes but the rotation
   runs in df64 instead of native fp64.
3. **CI guard**: a round-trip precision test that fails if XLA optimizations break the
   error-free transforms (this is why `jax.experimental.doubledouble` was removed —
   treat any tolerance regression as a compiler-interaction bug, and check whether
   `jax.default_matmul_precision` / donation / fusion flags changed).

### Phase 4 — Benchmarks and documentation

1. Extend `benchmarks/benchmark_precision.py`: add `Reprojector` (f32, f32x2) and the
   df64 variant vs NumPy f64 / JAX f64 / current Mixed, on CPU and GPU.
2. Document the decision tree in `DEVELOPER.md` (new subsection under §4):
   - reprojection → `Reprojector` (no fp64 anywhere per-pixel);
   - absolute output needed → offset-form + fp64 epilogue;
   - datacenter GPU / CPU → current Mixed classes are fine;
   - never → tensor-core emulation for this workload.
3. Record measured accuracy tables (Parts 1–2 of this document) alongside the
   benchmark results in `docs/benchmarks/`.

### Non-goals

- bf16x9 / Ozaki / cuBLAS emulation paths (wrong accuracy target or GEMM-only; see §2.3).
- Fixed-point angle representations (see §2.4).
- Changing the default behavior of existing classes — all new paths are additive.

### Risks

| Risk | Mitigation |
|---|---|
| Relative-rotation error grows with field separation / fine plate scales | `f32x2` matrix option; tests pin the envelope per plate scale |
| XLA reassociation silently breaks df64/compensated code | CI round-trip tolerance test (Phase 3); prefer Phases 1–2 which use no compensated tricks |
| Offset parameterization degenerates at poles / RA wrap | explicit pole and wraparound tests in Phase 2 |
| SIP interaction: distortion polynomials assume pixel offsets | unchanged — SIP already operates in the relative regime; covered by Phase 1 test matrix |

### Suggested order and effort

1. Phase 1 (core value, ~1 day incl. tests) →
2. Phase 4 benchmarks for Phase 1 (half day) →
3. Phase 2 (1 day, careful edge-case tests) →
4. Phase 3 only if a concrete need appears.

---

## Appendix — experiment scripts

Both experiments were run against this repo (JAX CPU backend, `uv run`):

- Stage-isolation round-trip (Part 1 table): varies which pipeline stage runs in fp32;
  shows the rotation step accounts for ~100% of the fp32 error.
- Relative-rotation reprojection (Part 2.1 table): composes `M_rel` in fp64 at setup,
  runs the per-pixel path in pure fp32, sweeps field-center separation.

Scripts live in the session scratchpad (`stage_analysis.py`, `relrot_experiment.py`);
they will be turned into `tests/test_reproject_relative.py` in Phase 1.
