"""Tests for the pytest-pyvista-approve console script."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pytest_pyvista.summary.approve import main

if TYPE_CHECKING:
    from pathlib import Path


# Where a default `pytest --summary_html` run leaves its renders, and therefore where the
# approve CLI looks for a manifest's sources unless --generated_image_dir says otherwise.
GENERATED_SUBPATH = "image_test_report/generated"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a project directory containing generated images, a cache dir, and an approvals.json."""
    generated = tmp_path / GENERATED_SUBPATH
    cache = tmp_path / "image_cache_dir"
    generated.mkdir(parents=True)
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


def test_root_cache_dir_manifest_is_rejected(project: Path) -> None:
    """A manifest claiming the filesystem root as its cache_dir is rejected, even with --force, and writes nothing."""
    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["cache_dir"] = "/"
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json", "--target", "cache"]) == 1
    assert main(["approvals.json", "--target", "cache", "--force"]) == 1
    assert not (project / "image_cache_dir" / "sphere.png").exists()


def test_empty_cache_dir_manifest_is_rejected(project: Path) -> None:
    """A manifest with an empty cache_dir is rejected outright, not silently resolved to the cwd."""
    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["cache_dir"] = ""
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json", "--target", "cache"]) == 1


def test_root_image_cache_dir_flag_is_rejected_even_with_force(project: Path) -> None:
    """--image_cache_dir / is rejected outright: the manifest-side root guard means nothing if the CLI-side flag can still name the root."""
    victim = project.parent / "victim.txt"
    victim.write_bytes(b"irreplaceable-victim-data")
    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["approved"][0]["destination"] = str(victim)
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json", "--target", "cache", "--force", "--image_cache_dir", "/"]) == 1
    assert victim.read_bytes() == b"irreplaceable-victim-data"


@pytest.mark.usefixtures("project")
def test_empty_image_cache_dir_flag_is_rejected() -> None:
    """--image_cache_dir '' is rejected outright, the same as an empty manifest cache_dir."""
    assert main(["approvals.json", "--target", "cache", "--image_cache_dir", ""]) == 1


def test_root_staging_dir_flag_is_rejected(project: Path) -> None:
    """--staging_dir / reproduces N2's exact outcome one flag over, on the default --target: rejected the same way, no --force needed."""
    victim = project.parent / "victim.txt"
    victim.write_bytes(b"irreplaceable-victim-data")
    cache = project / "image_cache_dir"
    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    # destination lives inside the real cache (so load_manifest's own containment check would
    # pass it) with a relative path chosen so that, once rejoined under a --staging_dir of "/",
    # it reconstructs the victim's real absolute path -- the exact escape this test guards.
    payload["approved"][0]["destination"] = str(cache / victim.relative_to(victim.anchor))
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json", "--staging_dir", "/"]) == 1
    assert victim.read_bytes() == b"irreplaceable-victim-data"


@pytest.mark.usefixtures("project")
def test_empty_staging_dir_flag_is_rejected() -> None:
    """--staging_dir '' is rejected outright, for consistency with --image_cache_dir '' -- not silently treated as the cwd."""
    assert main(["approvals.json", "--staging_dir", ""]) == 1


def test_symlinked_staging_destination_is_not_followed(project: Path) -> None:
    """A pre-existing symlink at the staging destination, pointing outside staging, is rejected rather than written through."""
    outside = project.parent / "outside.png"
    outside.write_bytes(b"do-not-touch")
    staging = project / "approved_images"
    staging.mkdir()
    (staging / "sphere.png").symlink_to(outside)

    assert main(["approvals.json"]) == 1
    assert outside.read_bytes() == b"do-not-touch"


def test_restaged_destination_equal_to_the_staging_root_is_rejected(project: Path, capsys: pytest.CaptureFixture) -> None:
    """
    A restaged destination that resolves to the staging root itself is rejected by name, not merely failing closed later for an unrelated reason.

    Constructed via a pre-existing symlink at the intermediate restaged path that points back at
    the staging root: _resolve_within follows it, sees the result is (trivially) contained in
    the root, and returns it -- without a destination-equals-root check, apply_approvals would
    then act on that path's *parent*, one level above the validated staging directory, and only
    happen to fail (with an unrelated "Is a directory" error from the final rename) rather than
    being rejected for the real reason. Asserting the specific message, not just the exit code,
    is what makes this a regression test for the missing check rather than for the fallback
    failure mode -- exit code 1 alone is unchanged whether the check is present or not.
    """
    cache = project / "image_cache_dir"
    staging = project / "approved_images"
    (staging / "sub").mkdir(parents=True)
    (staging / "sub" / "x.png").symlink_to(staging)

    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["approved"][0]["destination"] = str(cache / "sub" / "x.png")
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json"]) == 1
    assert "is the staging directory itself" in capsys.readouterr().err


def test_mismatched_cache_dir_requires_force_and_still_writes_to_the_real_cache(project: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """A manifest exported against a different cache_dir needs --force, which then still writes to the *real* cache, not the manifest's claim."""
    decoy_cache = tmp_path_factory.mktemp("decoy_cache")
    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["cache_dir"] = str(decoy_cache)
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json", "--target", "cache"]) == 1

    assert main(["approvals.json", "--target", "cache", "--force"]) == 0
    assert (project / "image_cache_dir" / "sphere.png").read_bytes() == b"generated-bytes"
    assert not (decoy_cache / "sphere.png").exists()


def test_generated_image_dir_override_is_honoured(project: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """--generated_image_dir lets sources live outside the current working directory."""
    external = tmp_path_factory.mktemp("external_generated")
    (external / "sphere.png").write_bytes(b"external-bytes")
    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["approved"][0]["source"] = str(external / "sphere.png")
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json", "--target", "cache", "--generated_image_dir", str(external)]) == 0

    assert (project / "image_cache_dir" / "sphere.png").read_bytes() == b"external-bytes"


def test_source_outside_default_generated_dir_is_rejected_without_the_override(
    project: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Without --generated_image_dir, a source outside the default generated dir is rejected -- confirms the override above is load-bearing."""
    external = tmp_path_factory.mktemp("external_generated")
    (external / "sphere.png").write_bytes(b"external-bytes")
    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["approved"][0]["source"] = str(external / "sphere.png")
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json", "--target", "cache"]) == 1


def test_a_source_elsewhere_in_the_working_tree_is_rejected(project: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Containment is the run's generated directory, not the whole working tree.

    A manifest round-trips through the reader's download directory, so it is untrusted: were
    the default source root the cwd, an edited manifest could name any file below it -- here a
    file that is not an image at all -- and have it copied into the cache under a baseline's
    name. The rejection names --generated_image_dir, so a project whose renders genuinely live
    elsewhere is told how to say so rather than being stuck.
    """
    secret = project / "secrets.env"
    secret.write_bytes(b"API_TOKEN=hunter2")
    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["approved"][0]["source"] = str(secret)
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json", "--target", "cache"]) == 1

    assert "--generated_image_dir" in capsys.readouterr().err
    assert not (project / "image_cache_dir" / "sphere.png").exists()


def test_overwriting_existing_baseline_replaces_its_content(project: Path) -> None:
    """Applying an approval for a baseline that already exists in the cache replaces its bytes."""
    (project / "image_cache_dir" / "sphere.png").write_bytes(b"old-baseline-bytes")

    assert main(["approvals.json", "--target", "cache"]) == 0

    assert (project / "image_cache_dir" / "sphere.png").read_bytes() == b"generated-bytes"


def test_unreadable_source_fails_the_copy_without_destroying_an_existing_baseline(project: Path, capsys: pytest.CaptureFixture) -> None:
    """A source that cannot be read fails the copy with exit code 1 -- and a pre-existing baseline at the destination survives untouched."""
    destination = project / "image_cache_dir" / "sphere.png"
    destination.write_bytes(b"IRREPLACEABLE-BASELINE")
    source = project / GENERATED_SUBPATH / "sphere.png"
    source.chmod(0o000)
    try:
        assert main(["approvals.json", "--target", "cache"]) == 1
    finally:
        source.chmod(0o644)

    assert "Failed to copy" in capsys.readouterr().err
    assert destination.read_bytes() == b"IRREPLACEABLE-BASELINE"


def test_long_destination_filename_does_not_prevent_the_copy(project: Path) -> None:
    """
    A destination basename long enough to overflow NAME_MAX once echoed into a temp-file prefix still copies cleanly.

    250 chars is comfortably under NAME_MAX (255) for the real destination -- a plain copy would
    succeed -- but an earlier version of apply_approvals built its temp filename as
    f".{destination.name}." plus mkstemp's own random suffix and ".tmp", which overflowed 255
    for exactly this length even though the destination name alone did not.
    """
    long_name = "a" * 246 + ".png"
    generated = project / GENERATED_SUBPATH
    (generated / long_name).write_bytes(b"long-name-bytes")
    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["approved"].append(
        {
            "test_name": "test_long",
            "image_name": long_name,
            "call_index": 0,
            "status": "new",
            "source": str(generated / long_name),
            "destination": str(project / "image_cache_dir" / long_name),
        },
    )
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json", "--target", "cache"]) == 0

    assert (project / "image_cache_dir" / long_name).read_bytes() == b"long-name-bytes"


def test_partial_failure_reports_every_completed_copy_before_stopping(project: Path, capsys: pytest.CaptureFixture) -> None:
    """When one copy in a batch fails, every copy that succeeded before it is still printed, and the batch stops there."""
    generated = project / GENERATED_SUBPATH
    cache = project / "image_cache_dir"
    (generated / "cube.png").write_bytes(b"cube-bytes")
    blocked = cache / "blocked"
    blocked.mkdir()

    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["approved"].append(
        {
            "test_name": "test_cube",
            "image_name": "cube.png",
            "call_index": 0,
            "status": "new",
            "source": str(generated / "cube.png"),
            "destination": str(blocked / "cube.png"),
        },
    )
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")
    blocked.chmod(0o555)

    try:
        assert main(["approvals.json", "--target", "cache"]) == 1
    finally:
        blocked.chmod(0o755)

    out, err = capsys.readouterr()
    assert (cache / "sphere.png").read_bytes() == b"generated-bytes"
    assert "copied:" in out
    assert "sphere.png" in out
    assert "cube.png" not in out
    assert "after applying 1 of 2" in err
    assert not (blocked / "cube.png").exists()
