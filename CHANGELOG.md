# Changelog

## Unreleased

- Development version `0.4.1.dev0`; records the verified PyPI 0.4.0 publication.

## 0.4.0 — 2026-11-13

- New `mapprojax.from_astropy` converts an `astropy.wcs.WCS` into a `Tan` or `Sin`, reading `crval`, `crpix`, the pixel scale matrix and the SIP coefficients from the WCS rather than from FITS cards. It handles what the raw header loader cannot: CROTA normalisation, non-degree axis units, and astropy's own resolution of competing linear representations. Anything it cannot reproduce exactly is refused with `from_astropy: <property> is unsupported (<detail>)` rather than approximated; see the [refusal table](docs/user_manual.md#refusals) and the [divergence table](docs/user_manual.md#where-it-differs-from-load_wcs_params_from_header). The conversion preserves the pixel-to-coordinate mapping but not the frame identity or observation metadata, and a cube needs an explicit `wcs.celestial`.
- New `[astropy]` extra (`astropy>=7.2.0`). astropy stays optional: `import mapprojax` still neither imports nor resolves it, so the kernel's runtime dependencies stay numpy and JAX. Install with `pip install 'mapprojax[astropy]'`.
- Fix: `wcs_utils.load_wcs_params_from_header` now reads only SIP terms with `i + j <= order`, as the SIP convention defines and as astropy evaluates them. It previously read every card in the square `(order+1) x (order+1)` grid, so a coefficient outside the triangle entered the matrix. This changes public loader behaviour only for headers carrying such a coefficient: matrices are bit-identical otherwise, the shipped fixtures are unchanged, and order-1 SIP handling is untouched.
- Release engineering: PyPI install instructions, a dated 0.3.0 entry, and defined local release-gate inputs.

## 0.3.0 — 2026-11-09

- Breaking: the public `Tan` and `Sin` classes are JAX PyTrees with required `precision="f32"`, `"mixed"` or `"f64"`. Retired NumPy, `*Array`, `*Jax` and `*Mixed` class families have no compatibility aliases.
- Replace array classes with stacked compatible PyTrees and `jax.vmap`. Absolute `proj`/`unproj` require mixed or f64 precision and JAX x64; f32 uses offset methods and producer-typed host composition helpers.
- JSON `to_dict`/`from_dict` is the serialization contract; ASDF and undocumented byte methods are removed. See the [migration table](docs/user_manual.md#migration-from-the-retired-api).
- `Reprojector.pixel_map` returns finite invalid-coordinate stand-ins and a validity mask, with optional half-open target bounds. It maps coordinates, not image data.
- Document gradients through pixels, CD and SIP coefficients, and the live-matrix `from_matrix` path; host reference composition is not differentiable. See the [differentiability contract](docs/user_manual.md#differentiability-contract).
- Precision claims distinguish WCS modes, f32/f64 reprojection, regression caps and sampled agreement with Astropy. See the [accuracy evidence](docs/benchmarks/accuracy_results.md) and [benchmark index](benchmarks/benchmarks.md) for workload-specific limitations, including mixed-mode native-pipeline errors.
- Add MIT project licensing and upstream projection attribution, evidence-based dependency floors, installed-distribution verification and CPU CI. Published to PyPI on 2026-11-09 through the protected publish workflow.
