"""Tests for pytest_pyvista.summary.approve."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from pytest_pyvista.summary.approve import ManifestError
from pytest_pyvista.summary.approve import load_manifest

if TYPE_CHECKING:
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
    """A well-formed manifest whose paths and schema version match loads cleanly, field for field."""
    approved = _load(workspace, _manifest(workspace))

    assert len(approved) == 1
    entry = approved[0]
    assert entry.test_name == "test_sphere"
    assert entry.image_name == "sphere.png"
    assert entry.call_index == 0
    assert isinstance(entry.call_index, int)
    assert entry.status == "failed"
    assert entry.source == (workspace["source_root"] / "sphere.png").resolve()
    assert isinstance(entry.source, Path)
    assert entry.destination == (workspace["cache_dir"] / "sphere.png").resolve()
    assert isinstance(entry.destination, Path)


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


def test_unknown_status_is_rejected(workspace: dict[str, Path]) -> None:
    """A status outside the six documented values is rejected rather than passed through."""
    entry = {
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": 0,
        "status": "totally-bogus",
        "source": str(workspace["source_root"] / "sphere.png"),
        "destination": str(workspace["cache_dir"] / "sphere.png"),
    }

    with pytest.raises(ManifestError, match="unknown status"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_non_string_source_is_rejected(workspace: dict[str, Path]) -> None:
    """A source that is not a JSON string (e.g. a hand-edited integer) is rejected, not crashed on."""
    entry = {
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": 0,
        "status": "new",
        "source": 12345,
        "destination": str(workspace["cache_dir"] / "sphere.png"),
    }

    with pytest.raises(ManifestError, match="non-string 'source'"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_non_string_destination_is_rejected(workspace: dict[str, Path]) -> None:
    """A destination that is not a JSON string (e.g. a hand-edited list) is rejected, not crashed on."""
    entry = {
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": 0,
        "status": "new",
        "source": str(workspace["source_root"] / "sphere.png"),
        "destination": ["not", "a", "string"],
    }

    with pytest.raises(ManifestError, match="non-string 'destination'"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_non_integer_call_index_is_rejected(workspace: dict[str, Path]) -> None:
    """A call_index that cannot be parsed as an integer is rejected rather than raising ValueError."""
    entry = {
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": "abc",
        "status": "new",
        "source": str(workspace["source_root"] / "sphere.png"),
        "destination": str(workspace["cache_dir"] / "sphere.png"),
    }

    with pytest.raises(ManifestError, match="non-integer 'call_index'"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_boolean_call_index_is_rejected(workspace: dict[str, Path]) -> None:
    """A boolean call_index is rejected rather than silently coerced to 0/1 (bool is a subclass of int)."""
    entry = {
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": True,
        "status": "new",
        "source": str(workspace["source_root"] / "sphere.png"),
        "destination": str(workspace["cache_dir"] / "sphere.png"),
    }

    with pytest.raises(ManifestError, match="non-integer 'call_index'"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_non_string_cache_dir_is_rejected(workspace: dict[str, Path]) -> None:
    """A top-level cache_dir that is not a JSON string is rejected, not crashed on."""
    with pytest.raises(ManifestError, match="'cache_dir' must be a string"):
        _load(workspace, _manifest(workspace, cache_dir=12345))


def test_missing_cache_dir_key_is_rejected(workspace: dict[str, Path]) -> None:
    """A manifest missing the top-level cache_dir key entirely is rejected, like a missing schema_version."""
    path = workspace["root"] / "approvals.json"
    payload = json.loads(_manifest(workspace).read_text(encoding="utf-8"))
    del payload["cache_dir"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="missing required key: cache_dir"):
        _load(workspace, path)


def test_missing_cache_dir_key_is_rejected_even_with_force(workspace: dict[str, Path]) -> None:
    """Force overrides a cache_dir *mismatch*, not the absence of the key entirely."""
    path = workspace["root"] / "approvals.json"
    payload = json.loads(_manifest(workspace).read_text(encoding="utf-8"))
    del payload["cache_dir"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="missing required key: cache_dir"):
        _load(workspace, path, force=True)


def test_manifest_file_does_not_exist_is_rejected(workspace: dict[str, Path]) -> None:
    """A nonexistent manifest path -- the most likely CLI typo -- is rejected, not a raw traceback."""
    with pytest.raises(ManifestError, match="could not be read"):
        _load(workspace, workspace["root"] / "nope.json")


def test_manifest_path_is_a_directory_is_rejected(workspace: dict[str, Path]) -> None:
    """A manifest path that names a directory is rejected, not a raw IsADirectoryError."""
    with pytest.raises(ManifestError, match="could not be read"):
        _load(workspace, workspace["root"])


def test_destination_equal_to_the_target_root_is_rejected(workspace: dict[str, Path]) -> None:
    """A destination that names the target directory itself, not a file within it, is rejected."""
    entry = {
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": 0,
        "status": "new",
        "source": str(workspace["source_root"] / "sphere.png"),
        "destination": str(workspace["cache_dir"]),
    }

    with pytest.raises(ManifestError, match="target directory itself"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_symlinked_source_escaping_root_is_rejected(workspace: dict[str, Path]) -> None:
    """A symlink inside source_root pointing outside it is rejected, not silently followed."""
    outside = workspace["root"] / "outside.png"
    outside.write_bytes(b"png")
    link = workspace["source_root"] / "escape.png"
    link.symlink_to(outside)
    entry = {
        "test_name": "test_evil",
        "image_name": "escape.png",
        "call_index": 0,
        "status": "new",
        "source": str(link),
        "destination": str(workspace["cache_dir"] / "escape.png"),
    }

    with pytest.raises(ManifestError, match="outside"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_symlinked_destination_directory_escaping_root_is_rejected(workspace: dict[str, Path]) -> None:
    """A symlinked directory inside target_root, used as a destination parent, is rejected."""
    outside_dir = workspace["root"] / "outside_dir"
    outside_dir.mkdir()
    link_dir = workspace["cache_dir"] / "escape_dir"
    link_dir.symlink_to(outside_dir, target_is_directory=True)
    entry = {
        "test_name": "test_evil",
        "image_name": "escape.png",
        "call_index": 0,
        "status": "new",
        "source": str(workspace["source_root"] / "sphere.png"),
        "destination": str(link_dir / "escape.png"),
    }

    with pytest.raises(ManifestError, match="outside"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_symlink_loop_in_source_is_rejected(workspace: dict[str, Path]) -> None:
    """A symlink loop reachable from source is rejected rather than crashing with RuntimeError."""
    loop_a = workspace["source_root"] / "loop_a.png"
    loop_b = workspace["source_root"] / "loop_b.png"
    loop_a.symlink_to(loop_b)
    loop_b.symlink_to(loop_a)
    entry = {
        "test_name": "test_evil",
        "image_name": "loop.png",
        "call_index": 0,
        "status": "new",
        "source": str(loop_a),
        "destination": str(workspace["cache_dir"] / "loop.png"),
    }

    with pytest.raises(ManifestError, match="outside"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_missing_required_key_error_names_the_entry(workspace: dict[str, Path]) -> None:
    """The missing-key error identifies which entry, by index and name, out of a multi-entry manifest."""
    good = {
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": 0,
        "status": "failed",
        "source": str(workspace["source_root"] / "sphere.png"),
        "destination": str(workspace["cache_dir"] / "sphere.png"),
    }
    bad = {"test_name": "test_broken", "image_name": "broken.png", "call_index": 0, "status": "new"}

    with pytest.raises(ManifestError, match=r"entry 1 \(test_broken/broken.png\)"):
        _load(workspace, _manifest(workspace, approved=[good, bad]))


def test_dot_dot_inside_the_roots_is_resolved_before_being_returned(workspace: dict[str, Path]) -> None:
    """A source/destination containing '..' that still resolves inside its root is stored resolved, not raw."""
    nested = workspace["source_root"] / "sub"
    nested.mkdir()
    entry = {
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": 0,
        "status": "new",
        "source": str(nested / ".." / "sphere.png"),
        "destination": str(workspace["cache_dir"] / "sphere.png"),
    }

    approved = _load(workspace, _manifest(workspace, approved=[entry]))

    assert approved[0].source == (workspace["source_root"] / "sphere.png").resolve()
    assert ".." not in approved[0].source.parts
