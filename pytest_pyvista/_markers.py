"""Reusable platform and VTK version conditional skip markers."""

from __future__ import annotations

import os
import platform

import pytest
import pyvista

_MARKER_DEFINITIONS = (
    ("skip_egl(reason=...)", "skip the test when running with a headless OSMesa/EGL VTK build."),
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
    """Register the ``skip_egl``, ``skip_mac``, ``skip_windows`` and ``needs_vtk_version`` markers."""
    for signature, description in _MARKER_DEFINITIONS:
        config.addinivalue_line("markers", f"{signature}: {description}")


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


def _needs_vtk_version_skip_reason(item_mark: pytest.Mark) -> str | None:
    """
    Evaluate a ``needs_vtk_version`` marker against the running VTK version.

    The comparison is made directly against :data:`pyvista.vtk_version_info`
    (not a plain tuple derived from it) so that, on pyvista versions where it
    is a version-aware type, comparing against a constraint older than
    pyvista's own supported VTK floor raises for free — the plugin does not
    reimplement that check.

    Parameters
    ----------
    item_mark : pytest.Mark
        The ``needs_vtk_version`` marker collected from the test item.

    Returns
    -------
    str | None
        A skip reason when the running VTK version does not satisfy the
        bound, otherwise ``None``.

    """
    minimum, maximum = _parse_vtk_version_constraint(item_mark)
    reason = item_mark.kwargs.get("reason")
    current = pyvista.vtk_version_info

    if (minimum is not None and current < minimum) or (maximum is not None and current >= maximum):
        return reason if reason is not None else _default_reason(minimum, maximum, tuple(current))

    return None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """
    Apply the reusable platform and VTK conditional skip markers.

    Reads the ``skip_egl``, ``skip_mac``, ``skip_windows`` and
    ``needs_vtk_version`` markers off ``item`` and calls :func:`pytest.skip`
    when the corresponding condition holds.
    """
    for item_mark in item.iter_markers("needs_vtk_version"):
        if (skip_reason := _needs_vtk_version_skip_reason(item_mark)) is not None:
            pytest.skip(skip_reason)

    if item_mark := item.get_closest_marker("skip_egl"):
        reason = _marker_skip_reason(item_mark, "Test fails when using OSMesa/EGL VTK build")
        if _uses_egl():
            pytest.skip(reason)

    if item_mark := item.get_closest_marker("skip_windows"):
        reason = _marker_skip_reason(item_mark, "Test fails on Windows")
        if os.name == "nt":
            pytest.skip(reason)

    if item_mark := item.get_closest_marker("skip_mac"):
        reason = _marker_skip_reason(item_mark, "Test fails on MacOS")
        machine = item_mark.kwargs.get("machine")
        should_skip = platform.system() == "Darwin"
        if machine is not None:
            should_skip = should_skip and machine == platform.machine()
        if should_skip:
            pytest.skip(reason)
