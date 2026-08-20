"""Tests for the pytest-pyvista-approve console script."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pytest_pyvista.summary.approve import main

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a project directory containing generated images, a cache dir, and an approvals.json."""
    generated = tmp_path / "generated_images"
    cache = tmp_path / "image_cache_dir"
    generated.mkdir()
    cache.mkdir()
    (generated / "sphere.png").write_bytes(b"generated-bytes")

    manifest = {
        "schema_version": 1,
        "run_id": "run-1",
        "exported_at": "2026-08-12T14:05:00Z",
        "cache_dir": str(cache),
        "image_format": "png",
        "approved": [
            {
                "test_name": "test_sphere",
                "image_name": "sphere.png",
                "call_index": 0,
                "status": "new",
                "source": str(generated / "sphere.png"),
                "destination": str(cache / "sphere.png"),
            },
        ],
    }
    (tmp_path / "approvals.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_default_target_is_staging(project: Path, capsys: pytest.CaptureFixture) -> None:
    """With no --target flag, approved images land in ./approved_images, not the cache."""
    assert main(["approvals.json"]) == 0

    assert (project / "approved_images" / "sphere.png").read_bytes() == b"generated-bytes"
    assert not (project / "image_cache_dir" / "sphere.png").exists()
    assert "approved_images" in capsys.readouterr().out


def test_cache_target_writes_into_the_cache(project: Path) -> None:
    """--target cache copies straight into the manifest's cache directory."""
    assert main(["approvals.json", "--target", "cache"]) == 0

    assert (project / "image_cache_dir" / "sphere.png").read_bytes() == b"generated-bytes"


def test_dry_run_copies_nothing(project: Path, capsys: pytest.CaptureFixture) -> None:
    """--dry-run reports the planned copy but writes nothing to disk."""
    assert main(["approvals.json", "--target", "cache", "--dry-run"]) == 0

    assert not (project / "image_cache_dir" / "sphere.png").exists()
    assert "dry run" in capsys.readouterr().out.lower()


def test_custom_staging_dir_is_honoured(project: Path) -> None:
    """--staging_dir overrides the default ./approved_images location."""
    assert main(["approvals.json", "--staging_dir", "review"]) == 0

    assert (project / "review" / "sphere.png").is_file()


def test_invalid_manifest_exits_non_zero(project: Path, capsys: pytest.CaptureFixture) -> None:
    """A manifest that is not valid JSON produces exit code 1 and a clear stderr message."""
    (project / "approvals.json").write_text("{not json", encoding="utf-8")

    assert main(["approvals.json"]) == 1
    assert "not valid JSON" in capsys.readouterr().err


@pytest.mark.usefixtures("project")
def test_missing_manifest_exits_non_zero() -> None:
    """A manifest path that does not exist produces exit code 1."""
    assert main(["nope.json"]) == 1


def test_empty_manifest_reports_nothing_to_do(project: Path, capsys: pytest.CaptureFixture) -> None:
    """An empty approved list is not an error; it exits 0 and says there was nothing to do."""
    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["approved"] = []
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json"]) == 0
    assert "no approved images" in capsys.readouterr().out.lower()
