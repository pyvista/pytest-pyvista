"""Tests for pytest_pyvista.summary.approve."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pytest_pyvista.summary.approve import ManifestError
from pytest_pyvista.summary.approve import load_manifest

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_pyvista.summary.approve import ApprovedImage


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    """Build a source root with one generated image and an empty cache directory."""
    source_root = tmp_path / "generated"
    cache_dir = tmp_path / "cache"
    source_root.mkdir()
    cache_dir.mkdir()
    (source_root / "sphere.png").write_bytes(b"png")
    return {"root": tmp_path, "source_root": source_root, "cache_dir": cache_dir}


def _manifest(workspace: dict[str, Path], **overrides: object) -> Path:
    """Write an approvals manifest to disk, applying ``overrides`` to the default payload."""
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "exported_at": "2026-08-12T14:05:00Z",
        "cache_dir": str(workspace["cache_dir"]),
        "image_format": "png",
        "approved": [
            {
                "test_name": "test_sphere",
                "image_name": "sphere.png",
                "call_index": 0,
                "status": "failed",
                "source": str(workspace["source_root"] / "sphere.png"),
                "destination": str(workspace["cache_dir"] / "sphere.png"),
            },
        ],
    }
    payload.update(overrides)
    path = workspace["root"] / "approvals.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load(workspace: dict[str, Path], path: Path, **kwargs: object) -> list[ApprovedImage]:
    """Load ``path`` as a manifest, defaulting roots to the workspace's directories."""
    target_root = kwargs.pop("target_root", workspace["cache_dir"])
    return load_manifest(
        path,
        cache_dir=workspace["cache_dir"],
        target_root=target_root,
        source_root=workspace["source_root"],
        **kwargs,
    )


def test_valid_manifest_loads(workspace: dict[str, Path]) -> None:
    """A well-formed manifest whose paths and schema version match loads cleanly."""
    approved = _load(workspace, _manifest(workspace))

    assert len(approved) == 1
    assert approved[0].test_name == "test_sphere"


def test_unknown_schema_version_is_rejected(workspace: dict[str, Path]) -> None:
    """A manifest exported by a different schema version is rejected outright."""
    with pytest.raises(ManifestError, match="schema version"):
        _load(workspace, _manifest(workspace, schema_version=99))


def test_mismatched_cache_dir_is_rejected(workspace: dict[str, Path]) -> None:
    """A manifest exported against a different cache directory is rejected by default."""
    with pytest.raises(ManifestError, match="cache director"):
        _load(workspace, _manifest(workspace, cache_dir="/somewhere/else"))


def test_mismatched_cache_dir_is_allowed_with_force(workspace: dict[str, Path]) -> None:
    """Passing force=True overrides the cache directory mismatch rejection."""
    assert len(_load(workspace, _manifest(workspace, cache_dir="/somewhere/else"), force=True)) == 1


def test_source_outside_the_generated_root_is_rejected(workspace: dict[str, Path]) -> None:
    """A source path pointing outside the generated-image root is rejected."""
    entry = {
        "test_name": "test_evil",
        "image_name": "evil.png",
        "call_index": 0,
        "status": "new",
        "source": "/etc/passwd",
        "destination": str(workspace["cache_dir"] / "evil.png"),
    }

    with pytest.raises(ManifestError, match="outside"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_destination_escaping_the_target_root_is_rejected(workspace: dict[str, Path]) -> None:
    """A destination path that traverses out of the target root is rejected."""
    entry = {
        "test_name": "test_evil",
        "image_name": "evil.png",
        "call_index": 0,
        "status": "new",
        "source": str(workspace["source_root"] / "sphere.png"),
        "destination": str(workspace["cache_dir"] / ".." / "escaped.png"),
    }

    with pytest.raises(ManifestError, match="outside"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_missing_source_file_is_rejected(workspace: dict[str, Path]) -> None:
    """A source path that resolves inside the root but does not exist is rejected."""
    entry = {
        "test_name": "test_gone",
        "image_name": "gone.png",
        "call_index": 0,
        "status": "new",
        "source": str(workspace["source_root"] / "gone.png"),
        "destination": str(workspace["cache_dir"] / "gone.png"),
    }

    with pytest.raises(ManifestError, match="does not exist"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_malformed_json_is_rejected(workspace: dict[str, Path]) -> None:
    """A manifest file that is not valid JSON is rejected with a clear message."""
    path = workspace["root"] / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ManifestError, match="not valid JSON"):
        _load(workspace, path)


def test_missing_required_key_is_rejected(workspace: dict[str, Path]) -> None:
    """An approval entry missing a required key is rejected."""
    entry = {"test_name": "test_x", "image_name": "x.png", "call_index": 0, "status": "new"}

    with pytest.raises(ManifestError, match="missing"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_null_source_is_rejected(workspace: dict[str, Path]) -> None:
    """A null source, which the exporter can legitimately emit, is rejected rather than crashing."""
    entry = {
        "test_name": "test_x",
        "image_name": "x.png",
        "call_index": 0,
        "status": "new",
        "source": None,
        "destination": str(workspace["cache_dir"] / "x.png"),
    }

    with pytest.raises(ManifestError, match="missing"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_null_destination_is_rejected(workspace: dict[str, Path]) -> None:
    """A null destination, which the exporter can legitimately emit, is rejected rather than crashing."""
    entry = {
        "test_name": "test_x",
        "image_name": "x.png",
        "call_index": 0,
        "status": "new",
        "source": str(workspace["source_root"] / "sphere.png"),
        "destination": None,
    }

    with pytest.raises(ManifestError, match="missing"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_empty_approval_list_loads_as_empty(workspace: dict[str, Path]) -> None:
    """An empty ``approved`` list is valid input, not an error."""
    assert _load(workspace, _manifest(workspace, approved=[])) == []


def test_non_object_manifest_is_rejected(workspace: dict[str, Path]) -> None:
    """A syntactically valid JSON document that is not an object is rejected, not crashed on."""
    path = workspace["root"] / "approvals.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ManifestError, match="JSON object"):
        _load(workspace, path)


def test_non_list_approved_field_is_rejected(workspace: dict[str, Path]) -> None:
    """An ``approved`` field that is not a list is rejected, not crashed on."""
    with pytest.raises(ManifestError, match="must be a list"):
        _load(workspace, _manifest(workspace, approved="not-a-list"))


def test_non_object_approval_entry_is_rejected(workspace: dict[str, Path]) -> None:
    """An entry in ``approved`` that is not itself a JSON object is rejected, not crashed on."""
    with pytest.raises(ManifestError, match="not a JSON object"):
        _load(workspace, _manifest(workspace, approved=["not-an-object"]))
