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


#: Name of the ``needs_vtk_version_floor`` ini option (registered in pytest_pyvista.py).
FLOOR_INI_OPTION = "needs_vtk_version_floor"
#: Name of the private ``config`` attribute holding its resolved value.
FLOOR_CONFIG_ATTR = "_needs_vtk_version_floor"


def resolve_needs_vtk_version_floor(raw: str) -> tuple[int, int, int] | None:
    """
    Resolve ``needs_vtk_version_floor`` to a floor tuple, or ``None`` to disable the check.

    Parameters
    ----------
    raw : str
        The raw ini value: ``""``/``"true"`` for pyvista's own
        ``_MIN_SUPPORTED_VTK_VERSION`` (degrading to disabled if pyvista lacks it),
        ``"false"`` to disable, or a dotted VTK version (e.g. ``"9.3"``) to use instead.

    Returns
    -------
    tuple[int, int, int] | None
        The floor to compare ``needs_vtk_version`` bounds against, or ``None`` if disabled.

    Raises
    ------
    pytest.UsageError
        If ``raw`` is neither ``"true"``/``"false"`` nor a valid dotted VTK version.

    """
    value = raw.strip().lower()

    if value in ("", "true"):
        min_supported = getattr(pyvista, "_MIN_SUPPORTED_VTK_VERSION", None)
        return tuple(min_supported) if min_supported is not None else None  # type: ignore[return-value]

    if value == "false":
        return None

    try:
        parts = tuple(int(part) for part in raw.strip().split("."))
    except ValueError:
        msg = f"Invalid `{FLOOR_INI_OPTION}` value {raw!r}: must be `true`, `false`, or a dotted VTK version like '9.3' or '9.3.1'."
        raise pytest.UsageError(msg) from None

    try:
        return _pad_version(parts)
    except (TypeError, ValueError) as error:
        msg = f"Invalid `{FLOOR_INI_OPTION}` value {raw!r}: {error}"
        raise pytest.UsageError(msg) from None


def _pad_version(version: tuple[int, ...]) -> tuple[int, int, int]:
    """
    Validate and pad a version tuple to length three, e.g. ``(9, 3)`` -> ``(9, 3, 0)``.

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

    The positional form (``needs_vtk_version(9, 3)``) means ``at_least=(9, 3)``.

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


def _obsolete_vtk_version_reason(keyword: str, bound: tuple[int, int, int], floor: tuple[int, int, int]) -> str | None:
    """
    Return a message if ``bound`` is at or below ``floor``, otherwise ``None``.

    Parameters
    ----------
    keyword : str
        Either ``"at_least"`` or ``"less_than"``, for the message.
    bound : tuple[int, int, int]
        The padded version bound to check.
    floor : tuple[int, int, int]
        The resolved ``needs_vtk_version_floor`` (see :func:`_resolve_needs_vtk_version_floor`).

    Returns
    -------
    str | None
        An explanatory, actionable message if ``bound`` is obsolete, otherwise ``None``.

    """
    if bound > floor:
        return None

    formatted_bound = ".".join(map(str, bound))
    formatted_floor = ".".join(map(str, floor))
    outcome = (
        "always passes and can be safely removed"
        if keyword == "at_least"
        else "can now never pass (the test would be permanently skipped) and its guarded code can be safely removed"
    )
    return (
        f"`needs_vtk_version` constraint `{keyword}={formatted_bound}` is at or below the "
        f"configured `{FLOOR_INI_OPTION}` ({formatted_floor}), so this check {outcome}. Lower "
        f"`{FLOOR_INI_OPTION}` if you need to keep checking an older VTK version, or set it to "
        f"`false` to disable this check entirely."
    )


def _uses_egl() -> bool:
    """Return whether the running VTK is a headless OSMesa/EGL build, defaulting to ``False``."""
    try:
        from pyvista.plotting.utilities.gl_checks import uses_egl  # noqa: PLC0415
    except ImportError:
        return False
    return uses_egl()


def _needs_vtk_version_skip_reason(item_mark: pytest.Mark, config: pytest.Config) -> str | None:
    """
    Evaluate a ``needs_vtk_version`` marker against the running VTK version.

    Parameters
    ----------
    item_mark : pytest.Mark
        The ``needs_vtk_version`` marker collected from the test item.
    config : pytest.Config
        The pytest config, used to read the floor resolved by :func:`validate_ini_options`.

    Returns
    -------
    str | None
        A skip reason when the running VTK version does not satisfy the
        bound, otherwise ``None``.

    Raises
    ------
    pyvista.VTKVersionError
        If the constraint is at or below the resolved ``needs_vtk_version_floor``.

    """
    minimum, maximum = _parse_vtk_version_constraint(item_mark)
    reason = item_mark.kwargs.get("reason")

    floor = getattr(config, FLOOR_CONFIG_ATTR, None)
    if floor is not None:
        for keyword, bound in (("at_least", minimum), ("less_than", maximum)):
            if bound is not None and (obsolete_reason := _obsolete_vtk_version_reason(keyword, bound, floor)) is not None:
                error_cls = getattr(pyvista, "VTKVersionError", RuntimeError)
                raise error_cls(obsolete_reason)

    current = tuple(pyvista.vtk_version_info)

    if (minimum is not None and current < minimum) or (maximum is not None and current >= maximum):
        return reason if reason is not None else _default_reason(minimum, maximum, current)

    return None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Apply the ``skip_egl``, ``skip_linux``, ``skip_mac``, ``skip_windows`` and ``needs_vtk_version`` markers."""
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
