"""Result records for the image summary report."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
import json
from typing import TYPE_CHECKING
from typing import Literal
from typing import get_args

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = 1

ImageStatus = Literal["passed", "warned", "failed", "skipped", "new", "reset"]
ALL_STATUSES: tuple[ImageStatus, ...] = get_args(ImageStatus)

CacheWriteReason = Literal["add_missing_images", "reset_image_cache", "reset_only_failed"]


@dataclass
class ImageRecord:
    """A single generated image and everything the report needs to describe it."""

    run_id: str
    test_name: str
    image_name: str
    call_index: int
    status: ImageStatus

    error: float | None = None
    error_threshold: float | None = None
    warning_threshold: float | None = None
    high_variance_test: bool = False
    size_mismatch: bool = False

    # Both image sizes, so that a size mismatch can be reported in numbers. Optional with
    # defaults: a record written before these existed still loads, hence no schema bump.
    baseline_width: int | None = None
    baseline_height: int | None = None
    generated_width: int | None = None
    generated_height: int | None = None

    baseline_image: str | None = None
    generated_image: str | None = None
    diff_image: str | None = None
    baseline_image_full: str | None = None
    generated_image_full: str | None = None
    diff_image_full: str | None = None

    matched_baseline: str | None = None
    candidate_baselines: list[str] = field(default_factory=list)

    cache_written: bool = False
    cache_write_reason: CacheWriteReason | None = None
    skip_reason: str | None = None

    generated_source: str | None = None
    cache_destination: str | None = None

    cache_dir: str = ""
    image_format: str = "png"
    env_info: str = ""

    schema_version: int = SCHEMA_VERSION


def write_record(records_dir: Path, worker_id: str, record: ImageRecord) -> None:
    """Append a record as one line of JSON to this worker's records file."""
    records_dir.mkdir(parents=True, exist_ok=True)
    path = records_dir / f"records_{worker_id}.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(record)) + "\n")


def read_records(records_dir: Path) -> list[ImageRecord]:
    """
    Combine every worker's records into one list, ordered by test name then call index.

    Malformed lines are skipped so that a worker killed mid-write still yields the
    records it completed. A line counts as malformed when it is not valid JSON, when
    it is not a JSON object, or when it lacks a field ``ImageRecord`` requires.
    """
    if not records_dir.is_dir():
        return []

    known = {f.name for f in fields(ImageRecord)}
    records: list[ImageRecord] = []
    for path in sorted(records_dir.glob("records_*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    record = ImageRecord(**{k: v for k, v in data.items() if k in known})
                except (AttributeError, TypeError, json.JSONDecodeError):
                    continue
                records.append(record)

    return sorted(records, key=lambda record: (record.test_name, record.call_index))
