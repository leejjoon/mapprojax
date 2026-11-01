"""Shared astropy-reference evaluator for mapprojax accuracy.

Used by ``tests/test_astropy_reference.py``, ``tests/test_reproject_astropy.py``
and ``benchmarks/benchmark_accuracy.py`` so that the tests and the published
table measure the same cases with the same sampling and metrics. Not
production API.

Conventions
-----------
- Pixels are 0-based (astropy ``origin=0``); mapprojax ``crpix`` is 0-based.
- Each case is sampled on a deterministic ``GRID_N x GRID_N`` grid spanning
  pixel centres ``0 .. NAXIS-1`` inclusive, plus CRPIX.
- Forward error: great-circle separation between mapprojax ``unproj`` and
  astropy ``all_pix2world``, in milliarcseconds.
- Inverse error: mapprojax ``proj`` of the astropy sky position against the
  originating pixel, in pixels (radial ``hypot(dx, dy)``). The oracle's own
  inverse (``all_world2pix``, tol 1e-10, maxiter 100) is checked against the
  same pixels and reported; non-convergence is a reported failure, never
  masked.
- Non-finite mapprojax output on a valid pixel is counted and reported,
  never dropped from the statistics.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

DATA_DIR = Path(__file__).resolve().parent / "data"
BASELINES_PATH = DATA_DIR / "accuracy_baselines.json"
GRID_N = 33
ORACLE_TOL = 1e-10
ORACLE_MAXITER = 100

# fp64 budgets. SIP inverses are held to TARGET_PX as the Stage 2 goal and
# to the measured caps in accuracy_baselines.json as a regression guard.
FORWARD_MAS = 1e-4
INVERSE_PX = 1e-6
TARGET_PX = 1e-6

# Known Stage 2 defects, shared by the tests (strict xfail) and the report.
# Each one also has a regression cap in accuracy_baselines.json, asserted by an
# ordinary test, so it cannot get worse while its target is xfailed.
NEAR_POLE = (
    "Stage 2 near-pole declination: utils.xyz_to_radec's dec = arcsin(z) loses "
    "precision within ~0.1 deg of the pole (atan2(z, hypot(x, y)) is exact)"
)
SIP_INVERSE = "Stage 2 SIP inverse: single-pass AP/BP is not the converged inverse"

METRIC_KEYS = {"forward": ("forward_mas", "mas"), "inverse": ("inverse_px", "px")}

# Real-pair runs. The fp64 reverse direction has no SIP inverse in its path and
# is held to TARGET_PX; every other run is held to a measured regression cap.
PAIR_RUNS = (("forward", "f64"), ("forward", "f32"), ("reverse", "f64"), ("reverse", "f32"))
PAIR_EXACT = (("reverse", "f64"),)


def known_failure(case: "Case", impl: str, metric: str) -> str | None:
    """Reason string if (case, impl, metric) is a known Stage 2 defect."""
    if metric == "forward" and "near_pole" in case.name and impl in ("numpy", "jax"):
        return NEAR_POLE
    if metric == "inverse" and case.has_sip:
        return SIP_INVERSE
    return None


# --- headers and cases -------------------------------------------------------

def header_sep(name: str) -> str:
    """Card separator recorded for a fixture in ``provenance.json``: astropy's
    map headers are a single 80-character card stream, the SPHEREx extracts are
    newline-separated."""
    representation = json.loads((DATA_DIR / "provenance.json").read_text())[name]["representation"]
    return "\n" if "newline" in representation else ""


def load_header(name: str) -> fits.Header:
    """Load a ``tests/data`` header fixture."""
    return fits.Header.fromstring((DATA_DIR / name).read_text(), sep=header_sep(name))


def synthetic_header(proj: str, crval, naxis: int = 256, cdelt: float = 2e-3) -> fits.Header:
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, naxis, naxis
    h["CTYPE1"], h["CTYPE2"] = f"RA---{proj}", f"DEC--{proj}"
    h["CRPIX1"] = h["CRPIX2"] = (naxis + 1) / 2.0
    h["CRVAL1"], h["CRVAL2"] = crval
    h["CDELT1"], h["CDELT2"] = -cdelt, cdelt
    h["PC1_1"], h["PC1_2"], h["PC2_1"], h["PC2_2"] = 0.96, 0.28, -0.28, 0.96
    return h


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    header: fits.Header

    @property
    def proj(self) -> str:
        return "SIN" if "SIN" in self.header["CTYPE1"] else "TAN"

    @property
    def has_sip(self) -> bool:
        return "A_ORDER" in self.header


def cases() -> list[Case]:
    out = [
        Case("1904-66_TAN", load_header("1904-66_TAN.hdr")),
        Case("1904-66_SIN", load_header("1904-66_SIN.hdr")),
        Case("spherex_l2", load_header("spherex_l2.hdr")),
        Case("spherex_template", load_header("spherex_template.hdr")),
    ]
    centres = {
        "ordinary": (150.0, 30.0),
        "ra_wrap_hi": (359.99, 5.0),
        "ra_wrap_lo": (0.01, -5.0),
        "near_pole_n": (45.0, 89.9),
        "near_pole_s": (200.0, -89.9),
    }
    for proj in ("TAN", "SIN"):
        for label, crval in centres.items():
            out.append(Case(f"{proj.lower()}_{label}", synthetic_header(proj, crval)))
    return out


def pixel_grid(header: fits.Header, n: int = GRID_N) -> np.ndarray:
    """(N, 2) 0-based pixel samples: inclusive n x n grid plus CRPIX."""
    xs = np.linspace(0.0, header["NAXIS1"] - 1.0, n)
    ys = np.linspace(0.0, header["NAXIS2"] - 1.0, n)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    crpix = [[header["CRPIX1"] - 1.0, header["CRPIX2"] - 1.0]]
    return np.vstack([grid, crpix])


# --- astropy oracle ------------------------------------------------------------

def astropy_wcs(header: fits.Header) -> WCS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(header)


def oracle_world(header: fits.Header, pix: np.ndarray) -> np.ndarray:
    """(N, 2) world degrees (lon, lat) for 0-based pixels."""
    return astropy_wcs(header).all_pix2world(pix, 0)


def oracle_pixel(header: fits.Header, world: np.ndarray) -> np.ndarray:
    """Converged astropy inverse; raises ``NoConvergence`` on failure."""
    return astropy_wcs(header).all_world2pix(
        world, 0, tolerance=ORACLE_TOL, maxiter=ORACLE_MAXITER, quiet=False
    )


# --- metrics -----------------------------------------------------------------

def _unit(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def separation_mas(lon1, lat1, lon2, lat2) -> np.ndarray:
    chord = np.linalg.norm(_unit(lon1, lat1) - _unit(lon2, lat2), axis=-1)
    return np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))) * 3.6e6


@dataclasses.dataclass
class Stats:
    max: float
    rms: float
    n: int
    n_nonfinite: int

    @classmethod
    def of(cls, err) -> "Stats":
        err = np.asarray(err, dtype=float)
        finite = np.isfinite(err)
        good = err[finite]
        return cls(
            max=float(good.max()) if good.size else float("nan"),
            rms=float(np.sqrt(np.mean(good**2))) if good.size else float("nan"),
            n=int(err.size),
            n_nonfinite=int((~finite).sum()),
        )


# --- mapprojax implementations ----------------------------------------------

def _params(header):
    from mapprojax.wcs_utils import load_wcs_params_from_header

    return load_wcs_params_from_header(header)


def build(case: Case, backend: str):
    """``backend``: ``numpy`` (Tan/Sin) or ``jax`` (TanJax/SinJax, needs x64)."""
    import mapprojax as mp

    names = {("numpy", "TAN"): "Tan", ("numpy", "SIN"): "Sin",
             ("jax", "TAN"): "TanJax", ("jax", "SIN"): "SinJax"}
    crpix, cd, crval, sip = _params(case.header)
    return getattr(mp, names[backend, case.proj])(crpix, cd, crval, sip=sip)


IMPLEMENTATIONS = ("numpy", "jax", "numpy_absolute")


def implementations_for(case: Case) -> tuple[str, ...]:
    # Every implementation runs on every case, including the CRVAL2 = -90
    # 1904-66 fixtures: the offset/absolute APIs' cos(CRVAL2) scaling cancels
    # in the absolute round trip, and they agree with astropy there.
    return IMPLEMENTATIONS


def run_impl(case: Case, impl: str, pix: np.ndarray, world_deg: np.ndarray):
    """Return (world_deg, pix) from mapprojax for the given impl."""
    wcs = build(case, "jax" if impl == "jax" else "numpy")
    ra_in, dec_in = np.radians(world_deg[:, 0]), np.radians(world_deg[:, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if impl == "numpy_absolute":
            ra, dec = wcs.unproj_absolute(pix[:, 0], pix[:, 1])
            x, y = wcs.proj_absolute(ra_in, dec_in)
        else:
            ra, dec = wcs.unproj(pix[:, 0], pix[:, 1])
            x, y = wcs.proj(ra_in, dec_in)
    world = np.column_stack([np.degrees(np.asarray(ra)), np.degrees(np.asarray(dec))])
    return world, np.column_stack([np.asarray(x), np.asarray(y)])


def evaluate(case: Case, impl: str, n: int = GRID_N) -> dict:
    pix = pixel_grid(case.header, n)
    world_ref = oracle_world(case.header, pix)
    world, pix_back = run_impl(case, impl, pix, world_ref)
    fwd = separation_mas(world[:, 0], world[:, 1], world_ref[:, 0], world_ref[:, 1])
    inv = np.hypot(*(pix_back - pix).T)
    return {
        "case": case.name,
        "impl": impl,
        "proj": case.proj,
        "sip": case.has_sip,
        "n_samples": int(len(pix)),
        "forward_mas": dataclasses.asdict(Stats.of(fwd)),
        "inverse_px": dataclasses.asdict(Stats.of(inv)),
    }


def oracle_self_check(case: Case, n: int = GRID_N) -> dict:
    """Astropy round trip on the same grid; raises on non-convergence."""
    pix = pixel_grid(case.header, n)
    back = oracle_pixel(case.header, oracle_world(case.header, pix))
    return {"case": case.name, "roundtrip_px": dataclasses.asdict(Stats.of(np.hypot(*(back - pix).T)))}


# --- real SPHEREx pair -------------------------------------------------------

def galactic_to_icrs_matrix() -> np.ndarray:
    """Rotation taking a Galactic Cartesian unit vector to ICRS, columns =
    Galactic basis vectors expressed in ICRS (astropy-derived)."""
    from astropy.coordinates import ICRS, CartesianRepresentation, Galactic

    basis = Galactic(CartesianRepresentation(np.eye(3)))  # columns = basis vectors
    icrs = basis.transform_to(ICRS()).cartesian.xyz.value  # row = component, col = basis
    return np.asarray(icrs, dtype=float)


def galactic_to_icrs_deg(lon, lat):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame="galactic").icrs
    return c.ra.deg, c.dec.deg


def icrs_to_galactic_deg(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    return c.l.deg, c.b.deg


# Reprojector direction: its first WCS supplies the input pixels and its
# second receives them. "forward" samples the Galactic template grid and
# lands in the ICRS L2 exposure (frame rotation Galactic -> ICRS);
# "reverse" samples the L2 grid and lands in the template (ICRS -> Galactic).
PAIR_DIRECTIONS = {
    "forward": ("spherex_template", "spherex_l2", galactic_to_icrs_deg, False),
    "reverse": ("spherex_l2", "spherex_template", icrs_to_galactic_deg, True),
}


def pair_reference(direction: str = "forward", n: int = GRID_N) -> dict:
    """Astropy chain src pixel -> src frame -> dst frame -> dst pixel, with
    the converged inverse. Overlap is decided from the astropy dst pixel,
    never from mapprojax output."""
    src, dst, convert, _ = PAIR_DIRECTIONS[direction]
    h_src, h_dst = load_header(f"{src}.hdr"), load_header(f"{dst}.hdr")
    pix_src = pixel_grid(h_src, n)
    lon, lat = oracle_world(h_src, pix_src).T
    lon_d, lat_d = convert(lon, lat)
    pix_dst = oracle_pixel(h_dst, np.column_stack([lon_d, lat_d]))
    inside = ((pix_dst[:, 0] >= -0.5) & (pix_dst[:, 0] <= h_dst["NAXIS1"] - 0.5)
              & (pix_dst[:, 1] >= -0.5) & (pix_dst[:, 1] <= h_dst["NAXIS2"] - 0.5))
    return {"src_pix": pix_src, "dst_pix": pix_dst, "overlap": inside}


def reprojector(precision: str, direction: str = "forward", frame_rotation=None):
    """``Reprojector`` for ``direction`` built from NumPy WCS legs. The
    default rotation is the astropy-derived one; pass ``frame_rotation`` to
    override (e.g. the transposed matrix as a negative control)."""
    from mapprojax import Reprojector

    src, dst, _, transpose = PAIR_DIRECTIONS[direction]
    if frame_rotation is None:
        m = galactic_to_icrs_matrix()
        frame_rotation = m.T if transpose else m
    w_src = build(Case(src, load_header(f"{src}.hdr")), "numpy")
    w_dst = build(Case(dst, load_header(f"{dst}.hdr")), "numpy")
    return Reprojector(w_src, w_dst, frame_rotation=frame_rotation, matrix_precision=precision)


def pair_errors(precision: str, direction: str = "forward", frame_rotation=None,
                n: int = GRID_N) -> tuple[np.ndarray, dict]:
    """Per-sample pixel error on the overlap (non-finite where mapprojax
    returned an invalid position), plus the astropy reference."""
    ref = pair_reference(direction, n)
    rp = reprojector(precision, direction, frame_rotation)
    s = ref["src_pix"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x, y = rp.pix_to_pix(s[:, 0], s[:, 1])
    got = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    return np.hypot(*(got - ref["dst_pix"]).T)[ref["overlap"]], ref


CONTROL_MIN_PX = 1.0  # a wrong rotation must miss by at least this, and 100x the good max


def transposed_rotation_control(direction: str) -> dict:
    """Negative control: the transposed frame rotation must be caught on
    every overlap sample. A sample counts as caught if mapprojax rejects it
    (non-finite output, e.g. once TAN/SIN mask the back hemisphere -- the
    wrong rotation puts the whole overlap behind TAN's horizon) or if it
    misses astropy by more than ``max(CONTROL_MIN_PX, 100 x good max)``."""
    good = Stats.of(pair_errors("f64", direction)[0]).max
    m = galactic_to_icrs_matrix()
    wrong = m if PAIR_DIRECTIONS[direction][3] else m.T
    err, _ = pair_errors("f64", direction, frame_rotation=wrong)
    threshold = max(CONTROL_MIN_PX, 100.0 * good)
    caught = ~np.isfinite(err) | (err > threshold)
    finite = err[np.isfinite(err)]
    return {
        "direction": direction,
        "n": int(err.size),
        "n_caught": int(caught.sum()),
        "n_rejected": int((~np.isfinite(err)).sum()),
        "min_finite_px": float(finite.min()) if finite.size else float("nan"),
        "threshold_px": threshold,
        "detected": bool(err.size and caught.all()),
    }


def evaluate_pair(precision: str, direction: str = "forward", frame_rotation=None,
                  n: int = GRID_N) -> dict:
    err, ref = pair_errors(precision, direction, frame_rotation, n)
    s = ref["src_pix"]
    return {
        "direction": direction,
        "precision": precision,
        "n_samples": int(len(s)),
        "n_overlap": int(ref["overlap"].sum()),
        "pixel_error_px": dataclasses.asdict(Stats.of(err)),
    }


def load_baselines() -> dict:
    return json.loads(BASELINES_PATH.read_text())


def known_failure_cap(case: Case, impl: str, metric: str) -> float:
    """Regression cap for a known failure, in the metric's unit."""
    return load_baselines()["known_failures"][metric][f"{case.name}/{impl}"]["cap"]


def pair_cap(direction: str, precision: str) -> float:
    return load_baselines()["pair"][f"{direction}/{precision}"]["cap"]
