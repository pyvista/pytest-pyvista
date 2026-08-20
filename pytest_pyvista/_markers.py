"""Reusable platform and VTK version conditional skip markers."""

from __future__ import annotations

import os
import platform

import pytest
import pyvista

_MARKER_DEFINITIONS = (
    ("skip_egl(reason=...)", "skip the test when running with a headless OSMesa/EGL VTK build."),
    (
        "skip_linux(machine=None, reason=...)",
        "skip the test on Linux; if machine is given (e.g. 'aarch64') only skip when platform.machine() matches.",
    ),
    (
        "skip_mac(machine=None, reason=...)",
        "skip the test on macOS; if machine is given (e.g. 'arm64') only skip when platform.machine() matches.",
    ),
    ("skip_windows(reason=...)", "skip the test on Windows."),
    (
        "needs_vtk_version(*version, at_least=None, less_than=None, reason=...)",
        "skip the test unless the running VTK version satisfies the given bound.",
    ),
)


def register_markers(config: pytest.Config) -> None:
    """Register the ``skip_egl``, ``skip_linux``, ``skip_mac``, ``skip_windows`` and ``needs_vtk_version`` markers."""
    for signature, description in _MARKER_DEFINITIONS:
        config.addinivalue_line("markers", f"{signature}: {description}")


def register_ini_options(parser: pytest.Parser) -> None:
    """Register the ``raise_obsolete_vtk`` ini option."""
    parser.addini(
        "raise_obsolete_vtk",
        type="bool",
        default=False,
        help=(
            "Error when a `needs_vtk_version` bound is at or below pyvista's own "
            "supported VTK floor, since such a check is guaranteed to always pass and "
            "can be removed (default: False, opt-in)."
        ),
    )


def _pad_version(version: tuple[int, ...]) -> tuple[int, int, int]:
    """
    Validate and pad a version tuple with trailing zeros to a length of three.

    This makes shorter tuples such as ``(9, 3)`` compare correctly against the
    three-element :data:`pyvista.vtk_version_info` named tuple (e.g.
    ``(9, 3, 0)``).

    Parameters
    ----------
    version : tuple[int, ...]
        Version tuple of one to three integers.

    Returns
    -------
    tuple[int, int, int]
        The version tuple right-padded with zeros to a length of three.

    """
    if not all(isinstance(item, int) for item in version):
        msg = f"Version must be a tuple of integers, got {version!r}."
        raise TypeError(msg)

    expected = 3
    if (length := len(version)) > expected:
        msg = f"Version tuple incorrect length (needs <= {expected}), got {version!r}."
        raise ValueError(msg)
    return (*version, *(0,) * (expected - length))  # type: ignore[return-value]


def _marker_skip_reason(mark: pytest.Mark, default: str) -> str:
    """Return the marker's positional or ``reason=`` skip message, falling back to ``default``."""
    return mark.args[0] if mark.args else mark.kwargs.get("reason", default)


def _platform_marker_skip_reason(item_mark: pytest.Mark, system_name: str, default: str) -> str | None:
    """
    Return a skip reason for a ``skip_linux``/``skip_mac``-style marker, or ``None``.

    Skips when :func:`platform.system` matches ``system_name``, optionally narrowed
    further by the marker's ``machine=`` kwarg matching :func:`platform.machine`.

    Parameters
    ----------
    item_mark : pytest.Mark
        The marker collected from the test item.
    system_name : str
        The :func:`platform.system` value that should trigger a skip (e.g. ``"Linux"``).
    default : str
        Fallback skip reason if the marker gives no ``reason``.

    Returns
    -------
    str | None
        The skip reason if the marker's condition holds, otherwise ``None``.

    """
    machine = item_mark.kwargs.get("machine")
    should_skip = platform.system() == system_name
    if machine is not None:
        should_skip = should_skip and machine == platform.machine()
    return _marker_skip_reason(item_mark, default) if should_skip else None


def _parse_vtk_version_constraint(
    item_mark: pytest.Mark,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    """
    Normalize a ``needs_vtk_version`` marker as a pair of minimum and maximum versions.

    Supports the positional form (``needs_vtk_version(9, 3)`` means
    ``at_least=(9, 3)``) and the explicit ``at_least=``/``less_than=`` tuple
    forms.

    Parameters
    ----------
    item_mark : pytest.Mark
        The ``needs_vtk_version`` marker collected from the test item.

    Returns
    -------
    tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]
        The padded ``(minimum, maximum)`` version bound. Either element may be
        ``None`` if that bound was not specified.

    """
    versions = item_mark.args
    at_least = item_mark.kwargs.get("at_least")
    less_than = item_mark.kwargs.get("less_than")

    if versions and at_least is not None:
        msg = "Cannot specify both positional versions and the `at_least` keyword argument to the `needs_vtk_version` marker."
        raise ValueError(msg)

    minimum_: tuple[int, ...] | None
    if versions:
        first = versions[0]
        minimum_ = first if len(versions) == 1 and isinstance(first, tuple) else versions
    else:
        minimum_ = at_least
        if minimum_ is None and less_than is None:
            msg = "Need to specify either `at_least` or `less_than` keyword arguments to the `needs_vtk_version` marker."
            raise ValueError(msg)

    minimum = _pad_version(tuple(minimum_)) if minimum_ is not None else None
    maximum = _pad_version(tuple(less_than)) if less_than is not None else None

    if minimum is not None and maximum is not None and minimum > maximum:
        msg = "Cannot specify a minimum version greater than the maximum one."
        raise ValueError(msg)

    return minimum, maximum


def _default_reason(
    minimum: tuple[int, int, int] | None,
    maximum: tuple[int, int, int] | None,
    current: tuple[int, ...],
) -> str:
    """Generate a message describing an unsatisfied ``needs_vtk_version`` constraint."""
    if maximum is None:
        requirement = f"VTK version {'.'.join(map(str, minimum))} or greater"  # type: ignore[arg-type]
    elif minimum is None:
        requirement = f"a VTK version less than {'.'.join(map(str, maximum))}"
    else:
        requirement = f"a VTK version of at least {'.'.join(map(str, minimum))} and less than {'.'.join(map(str, maximum))}"
    return f"Test needs {requirement}. The installed version is {'.'.join(map(str, current))}."


def _obsolete_vtk_version_reason(keyword: str, bound: tuple[int, int, int]) -> str | None:
    """
    Return a message if ``bound`` is guaranteed to be satisfied by any supported VTK.

    ``None`` when pyvista does not expose ``_MIN_SUPPORTED_VTK_VERSION`` (older
    pyvista, degrades gracefully) or ``bound`` is above that floor.

    Parameters
    ----------
    keyword : str
        Either ``"at_least"`` or ``"less_than"``, for the message.
    bound : tuple[int, int, int]
        The padded version bound to check.

    Returns
    -------
    str | None
        An explanatory, actionable message if ``bound`` is obsolete, otherwise ``None``.

    """
    min_supported = getattr(pyvista, "_MIN_SUPPORTED_VTK_VERSION", None)
    if min_supported is None or bound > tuple(min_supported):
        return None

    formatted_bound = ".".join(map(str, bound))
    formatted_floor = ".".join(map(str, min_supported))
    outcome = (
        "always passes and can be safely removed"
        if keyword == "at_least"
        else "can now never pass (the test would be permanently skipped) and its guarded code can be safely removed"
    )
    return (
        f"`needs_vtk_version` constraint `{keyword}={formatted_bound}` is obsolete: pyvista "
        f"{pyvista.__version__} already requires VTK >= {formatted_floor}, so this check {outcome}. "
        f"To keep this check anyway (e.g. while still supporting older pyvista), set "
        f"`raise_obsolete_vtk = false` in your pytest configuration."
    )


def _uses_egl() -> bool:
    """
    Return whether the running VTK is a headless OSMesa/EGL build.

    The ``uses_egl`` helper lives at a private path that is not guaranteed on
    older supported pyvista (>=0.37); when it cannot be imported the safe
    fallback is ``False`` so the test runs instead of being skipped.
    """
    try:
        from pyvista.plotting.utilities.gl_checks import uses_egl  # noqa: PLC0415
    except ImportError:
        return False
    return uses_egl()


def _needs_vtk_version_skip_reason(item_mark: pytest.Mark, config: pytest.Config) -> str | None:
    """
    Evaluate a ``needs_vtk_version`` marker against the running VTK version.

    The version comparison itself is made against a plain tuple copy of
    :data:`pyvista.vtk_version_info`, not the object itself -- comparing it
    directly would raise whenever the *constraint* is older than pyvista's own
    supported VTK floor, regardless of whether the installed VTK actually
    satisfies the marker. Instead, this plugin runs its own obsolete-constraint
    check first (see :func:`_obsolete_vtk_version_reason`), gated on the
    ``raise_obsolete_vtk`` ini option (default: ``False``, opt-in)
    so a project can enable it deliberately, with a message that names the
    exact ini setting to flip -- unlike pyvista's own side effect, which
    cannot be turned off independently of the real comparison.

    Parameters
    ----------
    item_mark : pytest.Mark
        The ``needs_vtk_version`` marker collected from the test item.
    config : pytest.Config
        The pytest config, used to read ``raise_obsolete_vtk``.

    Returns
    -------
    str | None
        A skip reason when the running VTK version does not satisfy the
        bound, otherwise ``None``.

    Raises
    ------
    pyvista.VTKVersionError
        If ``raise_obsolete_vtk`` is enabled and the
        constraint is at or below pyvista's own supported VTK floor.

    """
    minimum, maximum = _parse_vtk_version_constraint(item_mark)
    reason = item_mark.kwargs.get("reason")

    if config.getini("raise_obsolete_vtk"):
        for keyword, bound in (("at_least", minimum), ("less_than", maximum)):
            if bound is not None and (obsolete_reason := _obsolete_vtk_version_reason(keyword, bound)) is not None:
                error_cls = getattr(pyvista, "VTKVersionError", RuntimeError)
                raise error_cls(obsolete_reason)

    current = tuple(pyvista.vtk_version_info)

    if (minimum is not None and current < minimum) or (maximum is not None and current >= maximum):
        return reason if reason is not None else _default_reason(minimum, maximum, current)

    return None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """
    Apply the reusable platform and VTK conditional skip markers.

    Reads the ``skip_egl``, ``skip_linux``, ``skip_mac``, ``skip_windows`` and
    ``needs_vtk_version`` markers off ``item`` and calls :func:`pytest.skip`
    when the corresponding condition holds.
    """
    for item_mark in item.iter_markers("needs_vtk_version"):
        if (skip_reason := _needs_vtk_version_skip_reason(item_mark, item.config)) is not None:
            pytest.skip(skip_reason)

    if item_mark := item.get_closest_marker("skip_egl"):
        reason = _marker_skip_reason(item_mark, "Test fails when using OSMesa/EGL VTK build")
        if _uses_egl():
            pytest.skip(reason)

    if item_mark := item.get_closest_marker("skip_windows"):
        reason = _marker_skip_reason(item_mark, "Test fails on Windows")
        if os.name == "nt":
            pytest.skip(reason)

    if (item_mark := item.get_closest_marker("skip_mac")) and (
        platform_reason := _platform_marker_skip_reason(item_mark, "Darwin", "Test fails on MacOS")
    ) is not None:
        pytest.skip(platform_reason)

    if (item_mark := item.get_closest_marker("skip_linux")) and (
        platform_reason := _platform_marker_skip_reason(item_mark, "Linux", "Test fails on Linux")
    ) is not None:
        pytest.skip(platform_reason)
