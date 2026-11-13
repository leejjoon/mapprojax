"""Build a mapprojax kernel directly from an :class:`astropy.wcs.WCS`.

``from_astropy`` is the preferred entry point over
:func:`mapprojax.wcs_utils.load_wcs_params_from_header`, because astropy
normalises CROTA, units and alternate-version cards that the raw header
loader refuses.

astropy is an optional dependency. Importing this module never imports it;
the imports happen inside the functions that need them, so
``import mapprojax`` and ``import mapprojax.astropy`` both stay astropy-free.

Two properties are worth stating up front, because they are easy to get wrong
when reading this module:

* **Numbers come from attributes, never from header text.**
  ``wcs.wcs.crval``, ``wcs.wcs.crpix``, ``wcs.pixel_scale_matrix`` and
  ``wcs.sip.*`` keep the header's floats bit for bit, while
  ``WCS.to_header()`` rounds to 14 significant digits (and trims a long
  value to the FITS card field). Serialisation is used only for *presence and
  classification* -- which distortion or projection cards are in play.
* **Astropy semantics are authoritative.** ``from_astropy`` reproduces
  astropy's own interpretation of the CD / PC+CDELT / CROTA precedence and
  of unit normalisation. It is therefore *wider* than the raw loader, which
  accepts only one linear representation and only degrees. Numerical
  agreement with ``load_wcs_params_from_header`` is claimed only for headers
  where both read the same linear representation.

Inside this module, ``from astropy.wcs import WCS`` resolves to the
third-party package because Python 3 imports are absolute, so the sibling
module name does not shadow it.
"""

import warnings

from .sip import DEFAULT_INVERSE_STEPS, Sip
from .wcs import Sin, Tan
from .wcs_utils import _check_lonpole, _check_pv, _parse_ctype

__all__ = ["from_astropy"]

_INSTALL_HINT = (
    "mapprojax.from_astropy requires astropy; pip install 'mapprojax[astropy]'"
)

#: Properties named in refusal messages. Tests and the manual's refusal table
#: assert these exact tokens, so keep the set and the template in one place.
_DISTORTION_ATTRS = ("cpdis1", "cpdis2", "det2im1", "det2im2")
_DISTORTION_PREFIXES = ("CPDIS", "CQDIS", "DP", "DQ", "PS")

#: Celestial longitude/latitude codes the kernel understands.
_LON_CODES = {"RA--", "GLON", "ELON"}
_LAT_CODES = {"DEC-", "GLAT", "ELAT"}


def _require_astropy():
    """Import and return ``astropy.wcs.WCS``, or raise the install hint."""
    try:
        from astropy.wcs import WCS
    except ImportError as exc:  # pragma: no cover - exercised in a subprocess
        raise ImportError(_INSTALL_HINT) from exc
    return WCS


def _deg_scale(unit_text, axis):
    """Return the factor converting ``unit_text`` to degrees.

    ``path`` is one of ``'cunit1'`` / ``'cunit2'`` and only appears in the
    refusal, so the message names the offending property.
    """
    import astropy.units as u

    token = f"cunit{axis}"
    if unit_text is None:
        return 1.0
    text = str(unit_text).strip()
    if not text:
        return 1.0
    try:
        # `parse_strict="silent"` keeps spellings astropy would normalise only
        # through `wcsset` (for example 'DEG') out of the parse failure and
        # into the angular check below, which is the accurate complaint.
        unit = u.Unit(text, parse_strict="silent")
    except Exception as exc:
        raise ValueError(
            f"from_astropy: {token} is unsupported ({text!r} is not a unit)"
        ) from exc
    if not unit.is_equivalent(u.deg):
        raise ValueError(
            f"from_astropy: {token} is unsupported "
            f"({text!r} is not a recognised angular unit)"
        )
    factor = float(unit.to(u.deg))
    if not factor > 0.0 or not factor < float("inf"):
        raise ValueError(
            f"from_astropy: {token} is unsupported (scale factor {factor!r} deg per unit)"
        )
    return factor


def _finite(name, value):
    """Refuse a non-finite numeric leaf, naming it."""
    import numpy as np

    arr = np.asarray(value, dtype=np.float64)
    if arr.size and not np.isfinite(arr).all():
        raise ValueError(f"from_astropy: non-finite {name} is unsupported")


def _check_sip_triangle(name, matrix, order):
    """Refuse a nonzero coefficient outside ``i + j <= order``.

    A programmatically built :class:`astropy.wcs.Sip` keeps the full square
    matrix, unlike a header-parsed one, so the converter applies the SIP
    convention itself rather than trusting the object.
    """
    import numpy as np

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"from_astropy: sip triangle is unsupported ({name} is not 2-D)")
    rows, cols = arr.shape
    limit = min(rows, cols)
    for i in range(limit):
        for j in range(limit):
            if i + j > order and arr[i, j] != 0.0:
                raise ValueError(
                    f"from_astropy: sip triangle is unsupported "
                    f"({name}[{i}, {j}] = {float(arr[i, j])!r} is outside order {order})"
                )


def _raw_finiteness(wcs):
    """Validate the caller's read-only attributes before any copy is made.

    This has to run *before* ``copy.deepcopy`` for two verified reasons.
    First, ``deepcopy`` itself runs ``wcsset`` on the copy, so a non-angular
    ``cunit`` or an infinite ``crval`` raises astropy's ``InvalidTransformError``
    from the copy before any later check could fire. Second, WCSLIB rewrites a
    ``NaN`` in ``wcs.wcs.cd`` to the finite sentinel ``9.87654321e+107``, which
    ``wcsset`` copies into ``pc`` and ``pixel_scale_matrix`` preserves -- so
    ``np.isfinite(w.pixel_scale_matrix)`` *passes* on a NaN CD. The read-only
    getters used here (``crpix``, ``crval``, ``cd``, ``pc``, ``cdelt``,
    ``has_cd``, ``has_pc``, ``cunit``) do not run ``wcsset``.
    """

    prm = wcs.wcs
    _finite("crpix", prm.crpix)
    _finite("crval", prm.crval)
    if prm.has_cd():
        _finite("cd", prm.cd)
    # `pc`/`cdelt` may coexist with `cd`, and astropy prefers them when they
    # do, so a NaN there would still reach `pixel_scale_matrix`. Reading
    # `cdelt` next to `cd` warns, which is not the caller's business here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _finite("cdelt", prm.cdelt)
        if prm.has_pc():
            _finite("pc", prm.pc)
    if wcs.sip is not None:
        for name in ("a", "b", "ap", "bp"):
            matrix = getattr(wcs.sip, name, None)
            if matrix is not None:
                _finite(f"sip.{name}", matrix)
        _finite("sip.crpix", wcs.sip.crpix)
    # An angular unit is required on every celestial axis. `_deg_scale`
    # performs the conversion; here we only fail early and precisely.
    for axis in (1, 2):
        if str(prm.ctype[axis - 1])[:4] in (_LON_CODES | _LAT_CODES):
            _deg_scale(prm.cunit[axis - 1], axis)


def _refuse_distortions(wcs, header):
    """Refuse any distortion convention the kernel does not reproduce."""
    for name in _DISTORTION_ATTRS:
        if getattr(wcs, name, None) is not None:
            raise ValueError(f"from_astropy: {name} is unsupported (lookup table)")
    for key in header:
        for prefix in _DISTORTION_PREFIXES:
            if key.startswith(prefix):
                raise ValueError(
                    f"from_astropy: {prefix} is unsupported ({key} present)"
                )
    if wcs.wcs.get_ps():
        raise ValueError("from_astropy: PS is unsupported (PS parameters present)")


def _refuse_ctype(header):
    """Refuse unsupported projection conventions named by either CTYPE."""
    for axis in (1, 2):
        card = f"CTYPE{axis}"
        ctype = str(header.get(card, ""))
        for token in ("TPV", "TPD", "-TAB"):
            if token in ctype:
                raise ValueError(
                    f"from_astropy: {token} is unsupported ({card} = {ctype!r})"
                )


def _ctype_cards(header):
    """Return ``{'CTYPE1': ..., 'CTYPE2': ...}`` as the relaxed header wrote them.

    Reading ``wcs.wcs.ctype`` is not enough: a TPV, TPD or ``-TAB`` WCS
    reports plain ``RA---TAN`` there, so the hidden convention would be lost.
    The relaxed serialisation preserves it.
    """
    return {
        "CTYPE1": str(header.get("CTYPE1", "")),
        "CTYPE2": str(header.get("CTYPE2", "")),
    }


def from_astropy(wcs, *, precision, inverse_steps=DEFAULT_INVERSE_STEPS):
    """Build a :class:`~mapprojax.wcs.Tan` or :class:`~mapprojax.wcs.Sin`.

    Parameters
    ----------
    wcs:
        An :class:`astropy.wcs.WCS` with exactly two pixel and two world
        axes, longitude first. For a cube, pass ``wcs.celestial`` yourself.
    precision:
        ``"f32"``, ``"mixed"`` or ``"f64"``. It never enables JAX x64. In
        ``"f32"`` the absolute :meth:`~mapprojax.wcs.Tan.proj` /
        :meth:`~mapprojax.wcs.Tan.unproj` are unavailable; use the offset
        methods or :class:`~mapprojax.reproject.Reprojector`.
    inverse_steps:
        Newton steps for the SIP inverse; the default is
        :data:`~mapprojax.sip.DEFAULT_INVERSE_STEPS`.

    Returns
    -------
    Tan | Sin
        An ordinary kernel in the input WCS's *own* celestial frame. RADESYS,
        EQUINOX, MJD and observation keywords are ignored, not applied and not
        kept: the conversion preserves the numerical pixel-to-coordinate
        mapping but not the frame identity or observation metadata. Keep the
        original astropy WCS when composing different frames.

    Raises
    ------
    ImportError
        astropy is not installed (the message names the extra).
    TypeError
        ``wcs`` is not an :class:`astropy.wcs.WCS`.
    ValueError
        A property astropy accepts but this kernel cannot reproduce.
    """
    import copy

    import numpy as np

    WCS = _require_astropy()

    # 2. Type first: `deepcopy`/`wcs` below raise AttributeError for a Header,
    #    str or bare object, which would hide the intended TypeError.
    if not isinstance(wcs, WCS):
        raise TypeError(
            f"from_astropy: type is unsupported (expected astropy.wcs.WCS, "
            f"got {type(wcs).__name__})"
        )

    # 3. Shape. A cube is the caller's problem: dropping an axis silently
    #    changes SIP and pixel_scale_matrix slicing.
    if wcs.naxis != 2:
        raise ValueError(
            f"from_astropy: axis order is unsupported (naxis = {wcs.naxis}); "
            "pass `wcs.celestial` for a two-axis celestial WCS"
        )

    # 4. Raw finiteness and unit type, before the copy (see _raw_finiteness).
    _raw_finiteness(wcs)

    # 5. Copy and initialise together: `deepcopy` is what runs `wcsset`.
    try:
        w = copy.deepcopy(wcs)
        w.wcs.set()
    except Exception as exc:
        raise ValueError(f"from_astropy: init is unsupported ({exc})") from exc

    # 6. Axis role, on the initialised copy: the lng/lat/has_celestial
    #    getters run `wcsset` in place, which must not touch the caller.
    try:
        celestial = bool(w.has_celestial)
        lng, lat = w.wcs.lng, w.wcs.lat
    except Exception as exc:
        raise ValueError(f"from_astropy: axis order is unsupported ({exc})") from exc
    if not celestial:
        raise ValueError("from_astropy: non-celestial axes is unsupported (no celestial pair)")
    if lng == 1 and lat == 0:
        raise ValueError(
            "from_astropy: axis order is unsupported (axes are swapped: "
            "CTYPE1 is the latitude axis)"
        )
    if lng < 0 or lat < 0:
        raise ValueError(
            "from_astropy: non-celestial axes is unsupported "
            f"(CTYPE1 = {w.wcs.ctype[0]!r}, CTYPE2 = {w.wcs.ctype[1]!r})"
        )

    # 7. Post-init finiteness: belt and braces for anything `set()` produced.
    _finite("crpix", w.wcs.crpix)
    _finite("crval", w.wcs.crval)
    _finite("cd", w.pixel_scale_matrix)
    if w.sip is not None:
        for name in ("a", "b", "ap", "bp"):
            matrix = getattr(w.sip, name, None)
            if matrix is not None:
                _finite(f"sip.{name}", matrix)
        _finite("sip.crpix", w.sip.crpix)

    # 8. Classification. `key=" "` matters: a WCS selected with key='A'
    #    otherwise emits CTYPE1A, and `header['CTYPE1']` raises KeyError.
    try:
        header = w.to_header(relax=True, key=" ")
    except Exception as exc:
        raise ValueError(f"from_astropy: header is unsupported ({exc})") from exc

    _refuse_distortions(w, header)

    cards = _ctype_cards(header)
    if w.sip is None:
        for card in ("CTYPE1", "CTYPE2"):
            if cards[card].endswith("-SIP"):
                raise ValueError(
                    "from_astropy: -SIP without SIP is unsupported "
                    f"({card} = {cards[card]!r} but the WCS carries no SIP coefficients)"
                )
    _refuse_ctype(cards)

    # 9. Units. `_deg_scale` is the unit rule; the loader's own
    #    `_check_cunit` is not reused because it can only see synthetic
    #    post-conversion 'deg' cards and so checks nothing.
    scales = [
        _deg_scale(w.wcs.cunit[axis], axis + 1)
        for axis in (0, 1)
    ]
    crval = np.asarray(w.wcs.crval, dtype=np.float64) * np.asarray(scales)
    scale_matrix = np.asarray(w.pixel_scale_matrix, dtype=np.float64)

    # 10. Numbers: `crpix - 1` exactly once, and the matching matrix row
    #     scaled per axis.
    crpix = np.asarray(w.wcs.crpix, dtype=np.float64) - 1.0
    cd = scale_matrix * np.asarray(scales)[:, None]

    # 11. Policy, wrapped so the D5 template holds for the helpers' own
    #     wording (the loader's messages are kept as the chained cause).
    try:
        proj = _parse_ctype(cards)
    except ValueError as exc:
        raise ValueError(f"from_astropy: projection is unsupported ({exc})") from exc
    try:
        # `cards` holds only the two CTYPE strings, so applying the PV rule
        # to it was a no-op; the relaxed header carries the actual PV cards.
        _check_pv(header, proj)
        for axis, index, _value in w.wcs.get_pv():
            card = f"PV{axis}_{index}"
            if proj == "SIN" and card in ("PV2_1", "PV2_2"):
                continue
            raise ValueError(
                f"from_astropy: PV is unsupported ({card} is present on {proj})"
            )
    except ValueError as exc:
        if str(exc).startswith("from_astropy:"):
            raise
        raise ValueError(f"from_astropy: PV is unsupported ({exc})") from exc
    try:
        _check_lonpole({"LONPOLE": float(w.wcs.lonpole)}, float(crval[1]))
    except ValueError as exc:
        raise ValueError(f"from_astropy: LONPOLE is unsupported ({exc})") from exc

    # 12. SIP. The reference pixel is initialised from the same CRPIX cards,
    #     so compare the raw values bit-exactly *before* subtracting 1.
    sip = None
    if w.sip is not None:
        raw_sip_crpix = np.asarray(w.sip.crpix, dtype=np.float64)
        if not np.array_equal(raw_sip_crpix, np.asarray(w.wcs.crpix, dtype=np.float64)):
            raise ValueError(
                "from_astropy: sip.crpix is unsupported "
                f"(SIP {raw_sip_crpix.tolist()} != WCS {np.asarray(w.wcs.crpix).tolist()})"
            )
        matrices = {}
        for name in ("a", "b", "ap", "bp"):
            matrix = getattr(w.sip, name, None)
            matrix = None if matrix is None else np.asarray(matrix, dtype=np.float64)
            if matrix is not None:
                order = matrix.shape[0] - 1
                _check_sip_triangle(name.upper(), matrix, order)
            matrices[name] = matrix
        sip = Sip(
            a=matrices["a"],
            b=matrices["b"],
            ap=matrices["ap"],
            bp=matrices["bp"],
            inverse_steps=inverse_steps,
        )

    # 13. Construct; the constructor validates `precision` and the CD.
    cls = {"TAN": Tan, "SIN": Sin}[proj]
    return cls(crpix, cd, crval, sip=sip, precision=precision)
