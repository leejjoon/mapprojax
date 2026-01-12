# migration to jax (dual-backend) - COMPLETED

   *. **Update Dependencies**: Added `jax` to `pyproject.toml`.
   *. **Refactor Core Logic for Dual Backend Support**:
       * Modified `src/mapprojax/utils.py` to be backend-agnostic (accepts `xp` argument).
       * Modified `src/mapprojax/base.py` to use a class-level `xp` attribute (defaults to `numpy`).
       * Created JAX-specific subclasses in `src/mapprojax/jax_base.py` and `src/mapprojax/jax_projections.py` that override `xp` with `jax.numpy`.
       * Updated all methods in `base.py` and `projections.py` to use `self.xp` for math and array creation.
   *. **JAX Performance & PyTree Registration**:
       * Registered `WCSJax`, `WCSJaxArray`, and all projection subclasses (e.g., `TanJax`, `SinJax`) as **JAX PyTree nodes**.
       * **Benefit**: Allows WCS instances to be passed as standard arguments to `@jax.jit` functions. JAX will treat them as structured data (leaves), enabling efficient recompilation-free execution when only numeric parameters change.
   *. **Handle Serialization**:
       * *Note*: JAX arrays must be cast to NumPy (using `np.asarray()`) before passing to ASDF to ensure compatibility with the WCS schema.
   *. **Verification**:
       * Created `tests/verify_jax.py` to confirm result consistency (matches NumPy to `1e-12`) and benchmark JIT performance.
       * Existing NumPy-based tests remain fully compatible and pass.