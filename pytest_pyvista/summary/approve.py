"""Validating exported approvals manifests, and applying them, before they touch the image cache."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys

from pytest_pyvista.summary.record import ALL_STATUSES
from pytest_pyvista.summary.record import SCHEMA_VERSION

_REQUIRED_KEYS = ("test_name", "image_name", "call_index", "status", "source", "destination")


class ManifestError(Exception):
    """Raised when an approvals manifest is malformed or unsafe to apply."""


class ApplyError(Exception):
    """
    Raised when a copy fails partway through a batch.

    Carries every ``(source, destination)`` pair already applied before the failure, so the
    caller can report exactly what changed on disk rather than just how many.
    """

    def __init__(self, message: str, completed: list[tuple[Path, Path]]) -> None:
        """Store ``completed`` alongside the usual exception message."""
        super().__init__(message)
        self.completed = completed


@dataclass
class ApprovedImage:
    """One approved image, with both ends of the copy resolved and validated."""

    test_name: str
    image_name: str
    call_index: int
    status: str
    source: Path
    destination: Path


def _resolve_within(path: Path, root: Path) -> Path | None:
    """
    Resolve ``path`` and return the result if it lies inside ``root``, following symlinks; else None.

    Resolves exactly once: the returned ``Path`` is the same object the caller then checks with
    ``is_file()``, compares for equality, and stores on ``ApprovedImage`` -- there is no later,
    separate ``resolve()`` call that could observe a different filesystem than the one this
    containment check saw.
    """
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (ValueError, RuntimeError, OSError):
        # ValueError: outside root. RuntimeError: symlink loop. OSError: e.g. an
        # embedded NUL byte makes the underlying lstat() fail. All three mean the
        # same thing here -- this path cannot be trusted to be inside `root`.
        return None
    return resolved


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

    # Resolved once, inside _resolve_within, and reused for every check below and for the
    # returned value -- never re-resolved -- so the path that is checked for containment is
    # provably the same object later checked for existence, returned to the caller, and (in
    # Task 11) copied.
    source = _resolve_within(Path(entry["source"]), source_root)
    if source is None:
        msg = f"Approval {label}: source {entry['source']} resolves outside the generated image directory {source_root}."
        raise ManifestError(msg)
    destination = _resolve_within(Path(entry["destination"]), target_root)
    if destination is None:
        msg = f"Approval {label}: destination {entry['destination']} resolves outside the target directory {target_root}."
        raise ManifestError(msg)
    if destination == target_root.resolve():
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
        source=source,
        destination=destination,
    )


def _check_cache_dir(payload: dict, cache_dir: Path, *, force: bool) -> None:
    """
    Check the manifest's top-level ``cache_dir`` against the run's actual cache directory.

    Required unconditionally, like ``schema_version``: ``force`` overrides a *mismatched*
    cache_dir, it is not a license to omit the key, leave it empty, or name the filesystem
    root -- an empty string or the root are rejected outright, regardless of ``force``, because
    both are a plausible result of a hand-edited or malformed manifest and neither is ever a
    legitimate image cache.
    """
    if "cache_dir" not in payload:
        msg = "Manifest is missing required key: cache_dir"
        raise ManifestError(msg)
    manifest_cache = payload["cache_dir"]
    if not isinstance(manifest_cache, str):
        msg = f"Manifest 'cache_dir' must be a string, got {manifest_cache!r}."
        raise ManifestError(msg)
    if not manifest_cache.strip():
        msg = "Manifest 'cache_dir' must not be empty."
        raise ManifestError(msg)

    try:
        manifest_root = Path(manifest_cache).resolve()
        run_root = Path(cache_dir).resolve()
    except (ValueError, RuntimeError, OSError) as error:
        msg = f"Manifest 'cache_dir' {manifest_cache!r} is not a usable path: {error}"
        raise ManifestError(msg) from error

    if manifest_root == Path(manifest_root.anchor):
        msg = f"Manifest 'cache_dir' {manifest_cache!r} resolves to the filesystem root ({manifest_root}); refusing to treat that as an image cache."
        raise ManifestError(msg)

    if force:
        return

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


def apply_approvals(approved: list[ApprovedImage], *, dry_run: bool = False) -> list[tuple[Path, Path]]:
    """
    Copy each approved image to its destination, returning the (source, destination) pairs applied.

    Stops at the first copy failure instead of continuing past it. Baseline images are
    load-bearing for the test suite and may be the only copy that exists; applying some
    approvals from a batch while silently skipping others -- with no way for the caller to
    see which succeeded -- would leave the cache in a worse, half-updated state than simply
    stopping and saying exactly how far it got. The raised ``ApplyError`` carries every copy
    already applied (``.completed``) so the caller can report precisely what changed on disk,
    not just how many.

    Before copying, any existing file at the destination is removed rather than overwritten in
    place: ``shutil.copy`` opens the destination for writing, which follows a symlink if one is
    there, so an existing destination that is a symlink pointing outside the approved target
    root could otherwise be written through even after validation confirmed the *checked* path
    was safe. Removing it first guarantees the write always creates a fresh regular file.
    """
    copies: list[tuple[Path, Path]] = []
    for image in approved:
        if not dry_run:
            try:
                image.destination.parent.mkdir(parents=True, exist_ok=True)
                image.destination.unlink(missing_ok=True)
                shutil.copy(image.source, image.destination)
            except OSError as error:
                msg = f"Failed to copy {image.source} -> {image.destination} after applying {len(copies)} of {len(approved)}: {error}"
                raise ApplyError(msg, completed=copies) from error
        copies.append((image.source, image.destination))
    return copies


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``pytest-pyvista-approve`` console script."""
    parser = argparse.ArgumentParser(
        prog="pytest-pyvista-approve",
        description="Apply images approved in a pytest-pyvista image summary report.",
    )
    parser.add_argument("manifest", help="Path to the approvals.json exported from the report.")
    parser.add_argument("--target", choices=["staging", "cache"], default="staging", help="Where to copy approved images.")
    parser.add_argument("--staging_dir", default="approved_images", help="Staging directory used when --target=staging.")
    parser.add_argument(
        "--generated_image_dir",
        default=None,
        help="Override the generated image directory the manifest's sources must resolve inside.",
    )
    parser.add_argument(
        "--image_cache_dir",
        default="image_cache_dir",
        help=(
            "The project's actual image cache directory -- independent ground truth used to check the "
            "manifest's claimed cache_dir against and to contain every destination path. Defaults to "
            "'image_cache_dir', matching pytest-pyvista's own --image_cache_dir default. The manifest's own "
            "'cache_dir' field is never trusted as this value: it is only ever compared against it."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Apply even if the manifest's cache directory does not match this one.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned copies without performing them.")
    return parser


def _restage(approved: list[ApprovedImage], *, cache_dir: Path, target_root: Path) -> list[ApprovedImage]:
    """
    Rewrite each entry's destination from its manifest cache path to the equivalent path under ``target_root``.

    The manifest never describes the staging directory -- it only knows about the cache -- so
    Task 10's containment check never ran against these rewritten paths. Re-resolving and
    re-checking each one here closes that gap: a pre-existing symlink sitting at the staging
    destination and pointing outside ``target_root`` is rejected rather than followed on copy.
    """
    restaged = []
    for image in approved:
        candidate = target_root / image.destination.relative_to(cache_dir)
        destination = _resolve_within(candidate, target_root)
        if destination is None:
            msg = f"Staging destination for {image.test_name}/{image.image_name} ({candidate}) resolves outside the staging directory {target_root}."
            raise ManifestError(msg)
        restaged.append(
            ApprovedImage(
                test_name=image.test_name,
                image_name=image.image_name,
                call_index=image.call_index,
                status=image.status,
                source=image.source,
                destination=destination,
            ),
        )
    return restaged


def _print_copies(copies: list[tuple[Path, Path]], *, dry_run: bool) -> None:
    """Print one line per copy performed (or, under ``--dry-run``, merely planned)."""
    prefix = "would copy" if dry_run else "copied"
    for source, destination in copies:
        print(f"{prefix}: {source} -> {destination}")  # noqa: T201


def _report(copies: list[tuple[Path, Path]], *, target_root: Path, dry_run: bool) -> None:
    """Print one line per copy performed (or planned), then a summary line, for a batch that ran to completion."""
    _print_copies(copies, dry_run=dry_run)
    verb = "planned" if dry_run else "applied"
    suffix = " (dry run)" if dry_run else ""
    print(f"{len(copies)} image(s) {verb} to {target_root}{suffix}")  # noqa: T201


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for the ``pytest-pyvista-approve`` console script.

    Reads an exported ``approvals.json``, validates it with ``load_manifest``, and copies each
    approved image to a staging directory (default) or straight into the image cache.
    """
    args = _build_parser().parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"error: {manifest_path} does not exist", file=sys.stderr)  # noqa: T201
        return 1

    try:
        # image_cache_dir is independent ground truth for this run's real cache directory --
        # it never comes from the manifest's own (untrusted) 'cache_dir' claim. The manifest's
        # claim is only ever *compared* against it (in load_manifest, via _check_cache_dir).
        image_cache_dir = Path(args.image_cache_dir).resolve()
        source_root = Path(args.generated_image_dir).resolve() if args.generated_image_dir else Path.cwd()
        target_root = image_cache_dir if args.target == "cache" else Path(args.staging_dir).resolve()
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: could not resolve --image_cache_dir/--generated_image_dir/--staging_dir: {error}", file=sys.stderr)  # noqa: T201
        return 1

    try:
        approved = load_manifest(
            manifest_path,
            cache_dir=image_cache_dir,
            target_root=image_cache_dir,
            source_root=source_root,
            force=args.force,
        )
        if args.target == "staging":
            approved = _restage(approved, cache_dir=image_cache_dir, target_root=target_root)
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)  # noqa: T201
        return 1

    if not approved:
        print("No approved images in manifest; nothing to do.")  # noqa: T201
        return 0

    try:
        copies = apply_approvals(approved, dry_run=args.dry_run)
    except ApplyError as error:
        _print_copies(error.completed, dry_run=False)
        print(f"error: {error}", file=sys.stderr)  # noqa: T201
        return 1

    _report(copies, target_root=target_root, dry_run=args.dry_run)
    return 0
