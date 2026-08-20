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


def _pad_version(version: tuple[int, ...]) -> tuple[int, ...]:
    """
    Pad a version tuple with trailing zeros to a length of three.

    This makes shorter tuples such as ``(9, 3)`` compare correctly against the
    three-element :data:`pyvista.vtk_version_info` named tuple (e.g.
    ``(9, 3, 0)``).

    Parameters
    ----------
    version : tuple[int, ...]
        Version tuple of one to three integers.

    Returns
    -------
    tuple[int, ...]
        The version tuple right-padded with zeros to a length of three.

    """
    expected = 3
    if len(version) > expected:
        msg = f"Version tuple incorrect length (needs <= {expected})"
        raise ValueError(msg)
    return version + (0,) * (expected - len(version))


def _marker_skip_reason(mark: pytest.Mark, default: str) -> str:
    """Return the marker's positional or ``reason=`` skip message, falling back to ``default``."""
    return mark.args[0] if mark.args else mark.kwargs.get("reason", default)


def _validate_version_components(version: tuple[int, ...], origin: str) -> None:
    """Raise a clear ``pytest.UsageError`` if any version component is not an int."""
    if not all(isinstance(part, int) for part in version):
        msg = f"`needs_vtk_version` {origin} must be integers, got {version!r}."
        raise pytest.UsageError(msg)


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

    Supports the positional form (``needs_vtk_version(9, 3)`` means
    ``at_least=(9, 3)``) and the explicit ``at_least=``/``less_than=`` tuple
    forms. Version tuples are padded to length three so ``(9, 3)`` compares
    correctly against ``(9, 3, 0)``.

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
    args = tuple(item_mark.args)
    at_least = item_mark.kwargs.get("at_least")
    less_than = item_mark.kwargs.get("less_than")
    reason = item_mark.kwargs.get("reason")

    if args and at_least is not None:
        msg = "Cannot specify both *args and the `at_least` keyword argument to the `needs_vtk_version` marker."
        raise ValueError(msg)

    if args:
        _min = args[0] if len(args) == 1 and isinstance(args[0], tuple) else args
        _max = less_than
    else:
        _min = at_least
        _max = less_than
        if _min is None and _max is None:
            msg = "Need to specify either `at_least` or `less_than` keyword arguments to the `needs_vtk_version` marker."
            raise ValueError(msg)

    if _min is not None:
        _min = tuple(_min)
        _validate_version_components(_min, "version components")
        _min = _pad_version(_min)
    if _max is not None:
        _max = tuple(_max)
        _validate_version_components(_max, "`less_than` version components")
        _max = _pad_version(_max)

    if _min is not None and _max is not None and _min > _max:
        msg = "Cannot specify a minimum version greater than the maximum one."
        raise ValueError(msg)

    curr_version = tuple(pyvista.vtk_version_info)

    if _max is None and _min is not None and curr_version < _min:
        return reason or f"Test needs VTK version >= {_min}, current is {curr_version}."

    if _min is None and _max is not None and curr_version >= _max:
        return reason or f"Test needs VTK version < {_max}, current is {curr_version}."

    if _min is not None and _max is not None and (curr_version < _min or curr_version >= _max):
        return reason or f"Test needs {_min} <= VTK version < {_max}, current is {curr_version}."

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
