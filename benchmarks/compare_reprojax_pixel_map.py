"""Compare reprojax pixel-map capability and plain-callable fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from reprojax.interp import reproject_interp_core
from reprojax.transform import build_transform


def _digest(values, *, boolean=False):
    array = np.asarray(values)
    if boolean:
        array = array.astype(np.uint8)
    else:
        array = np.nan_to_num(array, nan=0.0)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def compare_pair(data_dir: Path) -> list[dict]:
    with fits.open(data_dir / "spherex_l2b_v20.fits") as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32).copy()
        input_header = hdul[0].header.copy()
    output_header = fits.Header.fromtextfile(
        data_dir / "galactic_template_header.fits"
    )
    data[0, 0] = np.nan
    data[100, 100] = np.nan
    data[2039, 2039] = np.nan
    ny = int(output_header["NAXIS2"])
    nx = int(output_header["NAXIS1"])
    records = []

    for matrix_precision in ("f32", "f64"):
        transform = build_transform(
            WCS(input_header), WCS(output_header),
            matrix_precision=matrix_precision,
        )

        def plain_delegate(x, y):
            return transform(x, y)

        for coordinate_dtype in (jnp.float32, jnp.float64):
            axis_x = jnp.arange(nx, dtype=coordinate_dtype)
            axis_y = jnp.arange(ny, dtype=coordinate_dtype)
            x_out, y_out = jnp.meshgrid(axis_x, axis_y)
            mapped_x, mapped_y = transform(x_out, y_out)
            expected_valid = (
                jnp.isfinite(mapped_x) & jnp.isfinite(mapped_y)
                & (mapped_x >= -0.5) & (mapped_x < data.shape[1] - 0.5)
                & (mapped_y >= -0.5) & (mapped_y < data.shape[0] - 0.5)
            )
            _, _, returned_valid = transform.pixel_map(
                x_out, y_out, bounds_shape=data.shape
            )
            np.testing.assert_array_equal(
                np.asarray(returned_valid), np.asarray(expected_valid)
            )
            for order in (0, 1):
                capability = reproject_interp_core(
                    data, x_out, y_out, transform, order=order
                )
                fallback = reproject_interp_core(
                    data, x_out, y_out, plain_delegate, order=order
                )
                cap_values, cap_footprint = map(np.asarray, capability)
                fallback_values, fallback_footprint = map(np.asarray, fallback)
                cap_nan = np.isnan(cap_values)
                fallback_nan = np.isnan(fallback_values)
                records.append({
                    "matrix_precision": matrix_precision,
                    "coordinate_dtype": str(coordinate_dtype),
                    "order": order,
                    "value_mismatch": int(np.count_nonzero(
                        ~cap_nan & ~fallback_nan
                        & (cap_values != fallback_values)
                    )),
                    "nan_mask_mismatch": int(np.count_nonzero(
                        cap_nan != fallback_nan
                    )),
                    "footprint_mismatch": int(np.count_nonzero(
                        cap_footprint != fallback_footprint
                    )),
                    "cap_value_sha256": _digest(cap_values),
                    "fallback_value_sha256": _digest(fallback_values),
                    "cap_nan_mask_sha256": _digest(cap_nan, boolean=True),
                    "fallback_nan_mask_sha256": _digest(
                        fallback_nan, boolean=True
                    ),
                    "cap_footprint_sha256": _digest(cap_footprint),
                    "fallback_footprint_sha256": _digest(fallback_footprint),
                })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "data",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    records = compare_pair(args.data_dir)
    result = {
        "backend": jax.default_backend(),
        "devices": [repr(device) for device in jax.devices()],
        "records": records,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
