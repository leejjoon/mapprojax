"""Small precision-explicit WCS batching benchmark."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from mapprojax.wcs import Tan

jax.config.update("jax_enable_x64", True)


def run_benchmark(n_centres=3, n_pixels=10000):
    centres = jnp.asarray([[45.0, 30.0], [50.0, 30.0], [55.0, 30.0]])[:n_centres]
    pixels = jnp.asarray(np.linspace(0.0, 200.0, n_pixels), dtype=jnp.float64)

    def project(crval):
        wcs = Tan([100.0, 100.0], [[-1e-3, 0.0], [0.0, 1e-3]], crval, precision="f64")
        return wcs.unproj(pixels, pixels)

    batched = jax.jit(jax.vmap(project))
    batched(centres)[0].block_until_ready()
    start = time.perf_counter()
    ra, dec = batched(centres)
    ra.block_until_ready()
    elapsed = time.perf_counter() - start
    print(f"{n_centres} WCS centres x {n_pixels} pixels: {elapsed * 1e3:.2f} ms")
    return ra, dec


if __name__ == "__main__":
    run_benchmark()
