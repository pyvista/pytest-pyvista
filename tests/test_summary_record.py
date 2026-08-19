"""Tests for pytest_pyvista.summary.record."""

from __future__ import annotations

from pathlib import Path

from pytest_pyvista.summary.record import ALL_STATUSES
from pytest_pyvista.summary.record import SCHEMA_VERSION
from pytest_pyvista.summary.record import ImageRecord
from pytest_pyvista.summary.record import read_records
from pytest_pyvista.summary.record import write_record


def _record(**overrides: object) -> ImageRecord:
    kwargs = {
        "run_id": "run-1",
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": 0,
        "status": "failed",
    }
    kwargs.update(overrides)
    return ImageRecord(**kwargs)


def test_all_statuses_are_the_six_documented_values() -> None:
    """Verify ALL_STATUSES contains exactly the six documented status values."""
    assert set(ALL_STATUSES) == {"passed", "warned", "failed", "skipped", "new", "reset"}


def test_record_defaults_to_current_schema_version() -> None:
    """Verify ImageRecord defaults schema_version to SCHEMA_VERSION."""
    assert _record().schema_version == SCHEMA_VERSION


def test_write_then_read_round_trips_a_record(tmp_path: Path) -> None:
    """Verify write_record and read_records round-trip a record faithfully."""
    record = _record(error=812.4, error_threshold=500.0, candidate_baselines=["a.png", "b.png"])
    write_record(tmp_path, "gw0", record)

    assert read_records(tmp_path) == [record]


def test_read_combines_records_from_several_workers(tmp_path: Path) -> None:
    """Verify read_records combines records from multiple worker files."""
    write_record(tmp_path, "gw0", _record(test_name="test_a", image_name="a.png"))
    write_record(tmp_path, "gw1", _record(test_name="test_b", image_name="b.png"))

    names = [record.test_name for record in read_records(tmp_path)]

    assert names == ["test_a", "test_b"]


def test_read_sorts_by_test_name_then_call_index(tmp_path: Path) -> None:
    """Verify read_records sorts results by test_name then call_index."""
    write_record(tmp_path, "gw0", _record(test_name="test_b", image_name="b.png"))
    write_record(tmp_path, "gw0", _record(test_name="test_a", image_name="a_1.png", call_index=1))
    write_record(tmp_path, "gw0", _record(test_name="test_a", image_name="a.png", call_index=0))

    keys = [(record.test_name, record.call_index) for record in read_records(tmp_path)]

    assert keys == [("test_a", 0), ("test_a", 1), ("test_b", 0)]


def test_read_skips_a_truncated_trailing_line(tmp_path: Path) -> None:
    """Verify read_records gracefully skips truncated/malformed JSON lines."""
    write_record(tmp_path, "gw0", _record())
    with Path(tmp_path, "records_gw0.jsonl").open("a") as file:
        file.write('{"run_id": "run-1", "test_na')

    assert len(read_records(tmp_path)) == 1


def test_read_skips_a_line_missing_a_required_field(tmp_path: Path) -> None:
    """Verify a valid-JSON line lacking required fields is skipped without losing the good records."""
    write_record(tmp_path, "gw0", _record(test_name="test_a", image_name="a.png"))
    with Path(tmp_path, "records_gw0.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"run_id": "run-1", "test_name": "test_b"}\n')
    write_record(tmp_path, "gw0", _record(test_name="test_c", image_name="c.png"))

    assert [record.test_name for record in read_records(tmp_path)] == ["test_a", "test_c"]


def test_read_skips_a_line_that_is_not_a_json_object(tmp_path: Path) -> None:
    """Verify a line holding valid JSON that is not an object is skipped."""
    write_record(tmp_path, "gw0", _record())
    with Path(tmp_path, "records_gw0.jsonl").open("a", encoding="utf-8") as file:
        file.write("[1, 2, 3]\n")

    assert len(read_records(tmp_path)) == 1


def test_read_returns_empty_list_for_missing_directory(tmp_path: Path) -> None:
    """Verify read_records returns empty list when directory does not exist."""
    assert read_records(tmp_path / "nope") == []
