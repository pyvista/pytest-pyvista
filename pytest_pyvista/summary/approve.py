"""Validating exported approvals manifests before they are applied to the image cache."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

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
    except ValueError:
        return False
    return True


def _validate_entry(entry: object, *, source_root: Path, target_root: Path) -> ApprovedImage:
    """Validate one ``approved`` entry and return it as a resolved, contained ``ApprovedImage``."""
    if not isinstance(entry, dict):
        msg = f"Approval entry {entry!r} is not a JSON object."
        raise ManifestError(msg)

    # A value of None counts as missing: the exporter can legitimately emit a null source
    # or destination (see record.py), and that must be rejected rather than crash Path().
    missing = [key for key in _REQUIRED_KEYS if entry.get(key) is None]
    if missing:
        msg = f"Approval entry is missing required key(s): {', '.join(missing)}"
        raise ManifestError(msg)

    source = Path(entry["source"])
    destination = Path(entry["destination"])

    if not _within(source, source_root):
        msg = f"Source {source} resolves outside the generated image directory {source_root}."
        raise ManifestError(msg)
    if not _within(destination, target_root):
        msg = f"Destination {destination} resolves outside the target directory {target_root}."
        raise ManifestError(msg)
    if not source.is_file():
        msg = f"Source image {source} does not exist."
        raise ManifestError(msg)

    return ApprovedImage(
        test_name=str(entry["test_name"]),
        image_name=str(entry["image_name"]),
        call_index=int(entry["call_index"]),
        status=str(entry["status"]),
        source=source,
        destination=destination,
    )


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
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
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

    manifest_cache = payload.get("cache_dir", "")
    if not force and Path(manifest_cache).resolve() != Path(cache_dir).resolve():
        msg = (
            f"Manifest was exported against cache directory {manifest_cache!r}, but this run resolves "
            f"{cache_dir!s}. Re-run from the right project, or pass --force to override."
        )
        raise ManifestError(msg)

    approved = payload.get("approved", [])
    if not isinstance(approved, list):
        msg = "Manifest 'approved' field must be a list."
        raise ManifestError(msg)

    return [_validate_entry(entry, source_root=source_root, target_root=target_root) for entry in approved]
