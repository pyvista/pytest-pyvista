"""Validating exported approvals manifests before they are applied to the image cache."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from pytest_pyvista.summary.record import ALL_STATUSES
from pytest_pyvista.summary.record import SCHEMA_VERSION

_REQUIRED_KEYS = ("test_name", "image_name", "call_index", "status", "source", "destination")


class ManifestError(Exception):
    """Raised when an approvals manifest is malformed or unsafe to apply."""


@dataclass
class ApprovedImage:
    """One approved image, with both ends of the copy resolved and validated."""

    test_name: str
    image_name: str
    call_index: int
    status: str
    source: Path
    destination: Path


def _within(path: Path, root: Path) -> bool:
    """Return True if ``path`` resolves to somewhere inside ``root``, following symlinks."""
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, RuntimeError, OSError):
        # ValueError: outside root. RuntimeError: symlink loop. OSError: e.g. an
        # embedded NUL byte makes the underlying lstat() fail. All three mean the
        # same thing here -- this path cannot be trusted to be inside `root`.
        return False
    return True


def _entry_label(entry: dict, index: int) -> str:
    """Describe one approval entry for an error message, naming it when test/image names are present."""
    test_name = entry.get("test_name")
    image_name = entry.get("image_name")
    if test_name or image_name:
        return f"entry {index} ({test_name or '?'}/{image_name or '?'})"
    return f"entry {index}"


def _check_entry_shape(entry: object, index: int) -> dict:
    """
    Check that ``entry`` is a JSON object with every required key present and non-null.

    A value of ``None`` counts as missing: the exporter can legitimately emit a null source
    or destination (see record.py), and that must be rejected here rather than crash Path()
    further down. Returns ``entry`` narrowed to ``dict`` for the caller's field checks.
    """
    if not isinstance(entry, dict):
        msg = f"Approval entry {index} ({entry!r}) is not a JSON object."
        raise ManifestError(msg)

    missing = [key for key in _REQUIRED_KEYS if entry.get(key) is None]
    if missing:
        label = _entry_label(entry, index)
        msg = f"Approval {label} is missing required key(s): {', '.join(missing)}"
        raise ManifestError(msg)

    return entry


def _check_entry_field_types(entry: dict, label: str) -> None:
    """Check that ``entry``'s typed fields hold the JSON type the manifest contract requires."""
    if not isinstance(entry["source"], str):
        msg = f"Approval {label} has a non-string 'source': {entry['source']!r}."
        raise ManifestError(msg)
    if not isinstance(entry["destination"], str):
        msg = f"Approval {label} has a non-string 'destination': {entry['destination']!r}."
        raise ManifestError(msg)

    call_index_value = entry["call_index"]
    # bool is a subclass of int in Python, so it must be excluded explicitly or True/False
    # would silently pass as call_index 1/0.
    if not isinstance(call_index_value, int) or isinstance(call_index_value, bool):
        msg = f"Approval {label} has a non-integer 'call_index': {call_index_value!r}."
        raise ManifestError(msg)

    status_value = entry["status"]
    if status_value not in ALL_STATUSES:
        msg = f"Approval {label} has an unknown status {status_value!r}. Choose from: {', '.join(ALL_STATUSES)}."
        raise ManifestError(msg)


def _validate_entry(entry: object, index: int, *, source_root: Path, target_root: Path) -> ApprovedImage:
    """Validate one ``approved`` entry and return it as a resolved, contained ``ApprovedImage``."""
    entry = _check_entry_shape(entry, index)
    label = _entry_label(entry, index)
    _check_entry_field_types(entry, label)

    source = Path(entry["source"])
    destination = Path(entry["destination"])

    if not _within(source, source_root):
        msg = f"Approval {label}: source {source} resolves outside the generated image directory {source_root}."
        raise ManifestError(msg)
    if not _within(destination, target_root):
        msg = f"Approval {label}: destination {destination} resolves outside the target directory {target_root}."
        raise ManifestError(msg)
    if destination.resolve() == target_root.resolve():
        msg = f"Approval {label}: destination {destination} is the target directory itself, not a file within it."
        raise ManifestError(msg)
    if not source.is_file():
        msg = f"Approval {label}: source image {source} does not exist."
        raise ManifestError(msg)

    return ApprovedImage(
        test_name=str(entry["test_name"]),
        image_name=str(entry["image_name"]),
        call_index=entry["call_index"],
        status=entry["status"],
        # Resolved, not the raw string: this is the value that was actually checked for
        # containment, so it must also be the value the caller copies to and from --
        # otherwise a later symlink or an unresolved ".." could widen the gap between
        # what was approved and what gets touched on disk.
        source=source.resolve(),
        destination=destination.resolve(),
    )


def _check_cache_dir(payload: dict, cache_dir: Path, *, force: bool) -> None:
    """
    Check the manifest's top-level ``cache_dir`` against the run's actual cache directory.

    Required unconditionally, like ``schema_version``: ``force`` overrides a *mismatched*
    cache_dir, it is not a license to omit the key entirely.
    """
    if "cache_dir" not in payload:
        msg = "Manifest is missing required key: cache_dir"
        raise ManifestError(msg)
    manifest_cache = payload["cache_dir"]
    if not isinstance(manifest_cache, str):
        msg = f"Manifest 'cache_dir' must be a string, got {manifest_cache!r}."
        raise ManifestError(msg)

    if force:
        return

    try:
        manifest_root = Path(manifest_cache).resolve()
        run_root = Path(cache_dir).resolve()
    except (ValueError, RuntimeError, OSError) as error:
        msg = f"Manifest 'cache_dir' {manifest_cache!r} is not a usable path: {error}"
        raise ManifestError(msg) from error

    if manifest_root != run_root:
        msg = (
            f"Manifest was exported against cache directory {manifest_cache!r}, but this run resolves "
            f"{cache_dir!s}. Re-run from the right project, or pass --force to override."
        )
        raise ManifestError(msg)


def load_manifest(
    path: Path,
    *,
    cache_dir: Path,
    target_root: Path,
    source_root: Path,
    force: bool = False,
) -> list[ApprovedImage]:
    """
    Read and validate an exported approvals manifest.

    The manifest round-trips through the user's download directory, so it is treated as
    untrusted: the schema version must match, the cache directory must agree unless
    ``force`` is set, and every source and destination must resolve inside its root.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        msg = f"{path} could not be read: {error}"
        raise ManifestError(msg) from error

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        msg = f"{path} is not valid JSON: {error}"
        raise ManifestError(msg) from error

    if not isinstance(payload, dict):
        msg = f"{path} does not contain a JSON object."
        raise ManifestError(msg)

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        msg = f"Unsupported manifest schema version {version!r}; this pytest-pyvista expects {SCHEMA_VERSION}."
        raise ManifestError(msg)

    _check_cache_dir(payload, cache_dir, force=force)

    approved = payload.get("approved", [])
    if not isinstance(approved, list):
        msg = "Manifest 'approved' field must be a list."
        raise ManifestError(msg)

    return [_validate_entry(entry, index, source_root=source_root, target_root=target_root) for index, entry in enumerate(approved)]
