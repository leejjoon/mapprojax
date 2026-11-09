# Changelog

## Unreleased

- Post-release documentation: PyPI install instructions, dated 0.3.0 entry, local release-gate input definitions.

## 0.3.0 — 2026-11-09

- Breaking: the public `Tan` and `Sin` classes are JAX PyTrees with required `precision="f32"`, `"mixed"` or `"f64"`. Retired NumPy, `*Array`, `*Jax` and `*Mixed` class families have no compatibility aliases.
- Replace array classes with stacked compatible PyTrees and `jax.vmap`. Absolute `proj`/`unproj` require mixed or f64 precision and JAX x64; f32 uses offset methods and producer-typed host composition helpers.
- JSON `to_dict`/`from_dict` is the serialization contract; ASDF and undocumented byte methods are removed. See the [migration table](docs/user_manual.md#migration-from-the-retired-api).
- `Reprojector.pixel_map` returns finite invalid-coordinate stand-ins and a validity mask, with optional half-open target bounds. It maps coordinates, not image data.
- Document gradients through pixels, CD and SIP coefficients, and the live-matrix `from_matrix` path; host reference composition is not differentiable. See the [differentiability contract](docs/user_manual.md#differentiability-contract).
- Precision claims distinguish WCS modes, f32/f64 reprojection, regression caps and sampled agreement with Astropy. See the [accuracy evidence](docs/benchmarks/accuracy_results.md) and [benchmark index](benchmarks/benchmarks.md) for workload-specific limitations, including mixed-mode native-pipeline errors.
- Add MIT project licensing and upstream projection attribution, evidence-based dependency floors, installed-distribution verification and CPU CI. Published to PyPI on 2026-11-09 through the protected publish workflow.
