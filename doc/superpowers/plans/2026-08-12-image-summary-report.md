# Image Summary Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in HTML report, generated at the end of a normal `pytest` run, that shows every image test as a card with baseline, generated render and pixel difference, and lets the reader selectively approve images for the cache.

**Architecture:** Records are captured *inside* `VerifyImageCache.__call__` at comparison time — never reconstructed by scanning directories afterwards, because nothing is persisted by default and the cache-writing policies destroy the prior baseline. Each worker appends `ImageRecord` JSONL to a temp directory; the master combines them in `pytest_terminal_summary` and renders a report directory. A separate console script applies exported approvals.

**Tech Stack:** Python ≥3.10, pytest, PIL (via pyvista), NumPy (via pyvista), stdlib `json`/`dataclasses`/`importlib.resources`. No new runtime dependencies. Vanilla JS and CSS in the report — no frameworks, no CDN.

**Spec:** [`doc/superpowers/specs/2026-08-12-image-summary-report-design.md`](../specs/2026-08-12-image-summary-report-design.md)

## Global Constraints

- **Python floor:** `requires-python = ">=3.10"`. No `match`-only-3.11 syntax, no `Self`, no `tomllib`-3.11 assumptions.
- **No new runtime dependencies.** PIL and NumPy arrive via `pyvista`; use them, add nothing to `[project].dependencies`.
- **Ruff `select = ["ALL"]`, `line-length = 150`.** Every new module needs a module docstring, every public function a docstring. Imports: `from __future__ import annotations` is required first, `force-single-line = true`, `force-sort-within-sections = true` — one import per line, sorted by name ignoring `from`/`import`.
- **Type annotations on everything**, including `-> None`. `mypy` runs with `ignore_missing_imports = true`.
- **Flag naming is snake_case** to match every existing plugin flag: `--summary_html`, not `--summary-html`.
- **Statuses are exactly these six lowercase strings:** `passed`, `warned`, `failed`, `skipped`, `new`, `reset`.
- **`SCHEMA_VERSION = 1`** for both the record JSONL and the exported manifest.
- **Tests use the `pytester` fixture** (`pytest_plugins = "pytester"` is already set in `tests/conftest.py`). Per-file ignores already exempt `tests/**` from `ANN001`, `D104`, `INP001`, `S101`.
- **Commit after every task.** Conventional-commit prefixes (`feat:`, `test:`, `docs:`).

## File Structure

The existing package is flat, but `pytest_pyvista/pytest_pyvista.py` is already ~1180 lines. Rather than grow it further, the report lives in a focused subpackage; only the wiring (options, capture call, terminal summary) is added to the existing module.

| File | Responsibility |
|---|---|
| `pytest_pyvista/summary/__init__.py` | Public re-exports for the subpackage |
| `pytest_pyvista/summary/record.py` | `ImageRecord` dataclass, status type, JSONL write/read/combine |
| `pytest_pyvista/summary/diff.py` | Pixel-difference image, size-mismatch detection |
| `pytest_pyvista/summary/store.py` | Report image store: slugging, downscaling, full-size retention |
| `pytest_pyvista/summary/collect.py` | Status determination, capture entry point called from `VerifyImageCache` |
| `pytest_pyvista/summary/render.py` | Records → `index.html` |
| `pytest_pyvista/summary/assets/report.css` | Report stylesheet |
| `pytest_pyvista/summary/assets/report.js` | Filters, approval state, export |
| `pytest_pyvista/summary/approve.py` | `pytest-pyvista-approve` console script |
| `pytest_pyvista/pytest_pyvista.py` | Modified: options, `pytest_configure`, capture calls, `pytest_terminal_summary` |
| `pyproject.toml` | Modified: `[project.scripts]` |
| `README.rst` | Modified: docs |
| `tests/test_summary_*.py` | One test module per source module |

---

### Task 1: Image record and JSONL transport

**Files:**
- Create: `pytest_pyvista/summary/__init__.py`
- Create: `pytest_pyvista/summary/record.py`
- Test: `tests/test_summary_record.py`

**Interfaces:**
- Consumes: nothing
- Produces: `SCHEMA_VERSION: int`, `ImageStatus` (Literal alias), `ALL_STATUSES: tuple[str, ...]`, `CacheWriteReason` (Literal alias), `ImageRecord` (dataclass), `write_record(records_dir: Path, worker_id: str, record: ImageRecord) -> None`, `read_records(records_dir: Path) -> list[ImageRecord]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_record.py`:

```python
"""Tests for pytest_pyvista.summary.record."""

from __future__ import annotations

from pathlib import Path

from pytest_pyvista.summary.record import ALL_STATUSES
from pytest_pyvista.summary.record import SCHEMA_VERSION
from pytest_pyvista.summary.record import ImageRecord
from pytest_pyvista.summary.record import read_records
from pytest_pyvista.summary.record import write_record


def _record(**overrides) -> ImageRecord:
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
    assert set(ALL_STATUSES) == {"passed", "warned", "failed", "skipped", "new", "reset"}


def test_record_defaults_to_current_schema_version() -> None:
    assert _record().schema_version == SCHEMA_VERSION


def test_write_then_read_round_trips_a_record(tmp_path: Path) -> None:
    record = _record(error=812.4, error_threshold=500.0, candidate_baselines=["a.png", "b.png"])
    write_record(tmp_path, "gw0", record)

    assert read_records(tmp_path) == [record]


def test_read_combines_records_from_several_workers(tmp_path: Path) -> None:
    write_record(tmp_path, "gw0", _record(test_name="test_a", image_name="a.png"))
    write_record(tmp_path, "gw1", _record(test_name="test_b", image_name="b.png"))

    names = [record.test_name for record in read_records(tmp_path)]

    assert names == ["test_a", "test_b"]


def test_read_sorts_by_test_name_then_call_index(tmp_path: Path) -> None:
    write_record(tmp_path, "gw0", _record(test_name="test_b", image_name="b.png"))
    write_record(tmp_path, "gw0", _record(test_name="test_a", image_name="a_1.png", call_index=1))
    write_record(tmp_path, "gw0", _record(test_name="test_a", image_name="a.png", call_index=0))

    keys = [(record.test_name, record.call_index) for record in read_records(tmp_path)]

    assert keys == [("test_a", 0), ("test_a", 1), ("test_b", 0)]


def test_read_skips_a_truncated_trailing_line(tmp_path: Path) -> None:
    write_record(tmp_path, "gw0", _record())
    with Path(tmp_path, "records_gw0.jsonl").open("a") as file:
        file.write('{"run_id": "run-1", "test_na')

    assert len(read_records(tmp_path)) == 1


def test_read_returns_empty_list_for_missing_directory(tmp_path: Path) -> None:
    assert read_records(tmp_path / "nope") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summary_record.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pytest_pyvista.summary'`

- [ ] **Step 3: Write minimal implementation**

Create `pytest_pyvista/summary/__init__.py`:

```python
"""Image summary report for pytest-pyvista."""

from __future__ import annotations
```

Create `pytest_pyvista/summary/record.py`:

```python
"""Result records for the image summary report."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
import json
from pathlib import Path
from typing import Literal
from typing import get_args

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
    records it completed.
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
                except json.JSONDecodeError:
                    continue
                records.append(ImageRecord(**{k: v for k, v in data.items() if k in known}))

    return sorted(records, key=lambda record: (record.test_name, record.call_index))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_summary_record.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add pytest_pyvista/summary/__init__.py pytest_pyvista/summary/record.py tests/test_summary_record.py
git commit -m "feat: add ImageRecord and JSONL transport for the summary report"
```

---

### Task 2: Difference image

**Files:**
- Create: `pytest_pyvista/summary/diff.py`
- Test: `tests/test_summary_diff.py`

**Interfaces:**
- Consumes: nothing
- Produces: `DIFF_PIXEL_THRESHOLD: int`, `DIFF_COLOR: tuple[int, int, int]`, `compute_diff_image(baseline_path: Path, generated_path: Path) -> tuple[Image.Image | None, bool]` returning `(diff, size_mismatch)` — `diff` is `None` exactly when `size_mismatch` is `True`

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_diff.py`:

```python
"""Tests for pytest_pyvista.summary.diff."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pytest_pyvista.summary.diff import DIFF_COLOR
from pytest_pyvista.summary.diff import compute_diff_image


def _write(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_identical_images_produce_no_highlighted_pixels(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "a.png", (10, 20, 30))
    generated = _write(tmp_path / "b.png", (10, 20, 30))

    diff, size_mismatch = compute_diff_image(baseline, generated)

    assert size_mismatch is False
    assert not np.any(np.all(np.asarray(diff) == DIFF_COLOR, axis=2))


def test_fully_different_images_highlight_every_pixel(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "a.png", (0, 0, 0))
    generated = _write(tmp_path / "b.png", (255, 255, 255))

    diff, size_mismatch = compute_diff_image(baseline, generated)

    assert size_mismatch is False
    assert np.all(np.all(np.asarray(diff) == DIFF_COLOR, axis=2))


def test_diff_keeps_the_baseline_dimensions(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "a.png", (0, 0, 0), size=(12, 5))
    generated = _write(tmp_path / "b.png", (0, 0, 255), size=(12, 5))

    diff, _ = compute_diff_image(baseline, generated)

    assert diff.size == (12, 5)


def test_mismatched_sizes_report_a_mismatch_and_no_image(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "a.png", (0, 0, 0), size=(8, 8))
    generated = _write(tmp_path / "b.png", (0, 0, 0), size=(9, 8))

    diff, size_mismatch = compute_diff_image(baseline, generated)

    assert size_mismatch is True
    assert diff is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summary_diff.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pytest_pyvista.summary.diff'`

- [ ] **Step 3: Write minimal implementation**

Create `pytest_pyvista/summary/diff.py`:

```python
"""Pixel-difference images for the summary report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

# Summed absolute channel difference above which a pixel counts as changed.
DIFF_PIXEL_THRESHOLD = 8

# Magenta stays distinguishable under the common colour-vision deficiencies.
DIFF_COLOR = (255, 0, 255)

# How much of the baseline's luminance survives behind the overlay.
_FADE = 0.35


def compute_diff_image(baseline_path: Path, generated_path: Path) -> tuple[Image.Image | None, bool]:
    """
    Build a difference image highlighting where two renders disagree.

    Changed pixels are painted in ``DIFF_COLOR`` over a faded greyscale copy of the
    baseline, so unchanged structure stays legible behind the overlay.

    Returns a ``(diff, size_mismatch)`` pair. When the two images differ in size a
    pixel difference is undefined, so ``diff`` is ``None`` and ``size_mismatch`` is
    ``True``.
    """
    baseline = Image.open(baseline_path).convert("RGB")
    generated = Image.open(generated_path).convert("RGB")

    if baseline.size != generated.size:
        return None, True

    delta = np.abs(np.asarray(baseline, dtype=np.int16) - np.asarray(generated, dtype=np.int16)).sum(axis=2)
    changed = delta > DIFF_PIXEL_THRESHOLD

    faded = np.asarray(baseline.convert("L").convert("RGB"), dtype=np.float32)
    faded = faded * _FADE + 255.0 * (1.0 - _FADE)
    canvas = faded.astype(np.uint8)
    canvas[changed] = DIFF_COLOR

    return Image.fromarray(canvas), False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_summary_diff.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add pytest_pyvista/summary/diff.py tests/test_summary_diff.py
git commit -m "feat: add difference image generation for the summary report"
```

---

### Task 3: Report image store

**Files:**
- Create: `pytest_pyvista/summary/store.py`
- Test: `tests/test_summary_store.py`

**Interfaces:**
- Consumes: nothing
- Produces: `FullSizeMode` (Literal alias `"none" | "failing" | "all"`), `slugify(name: str) -> str`, `ReportImageStore` with `__init__(self, report_dir: Path, *, max_image_size: int = 400, full_size: FullSizeMode = "failing")`, `save_file(self, source: Path, slug: str, role: str, *, status: str) -> tuple[str, str | None]`, `save_image(self, image: Image.Image, slug: str, role: str, *, status: str) -> tuple[str, str | None]`. Both `save_*` return `(thumbnail_relpath, full_relpath_or_None)`, POSIX-style relative to `report_dir`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_store.py`:

```python
"""Tests for pytest_pyvista.summary.store."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from pytest_pyvista.summary.store import ReportImageStore
from pytest_pyvista.summary.store import slugify


def _write(path: Path, size: tuple[int, int] = (800, 600)) -> Path:
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sphere", "sphere"),
        ("test_foo[1-2]", "test_foo_1-2"),
        ("a/b\\c", "a_b_c"),
        ("spaces here", "spaces_here"),
    ],
)
def test_slugify_makes_names_filesystem_safe(name: str, expected: str) -> None:
    assert slugify(name) == expected


def test_save_file_downscales_to_the_max_edge(tmp_path: Path) -> None:
    store = ReportImageStore(tmp_path / "report", max_image_size=100)
    source = _write(tmp_path / "src.png", size=(800, 600))

    thumb, _ = store.save_file(source, "sphere", "baseline", status="passed")

    assert max(Image.open(tmp_path / "report" / thumb).size) == 100


def test_save_file_returns_a_posix_relative_path(tmp_path: Path) -> None:
    store = ReportImageStore(tmp_path / "report")
    source = _write(tmp_path / "src.png")

    thumb, _ = store.save_file(source, "sphere", "baseline", status="passed")

    assert thumb == "images/sphere.baseline.png"
    assert (tmp_path / "report" / thumb).is_file()


def test_failing_mode_keeps_full_size_for_non_passed_only(tmp_path: Path) -> None:
    store = ReportImageStore(tmp_path / "report", max_image_size=100, full_size="failing")
    source = _write(tmp_path / "src.png")

    _, passed_full = store.save_file(source, "a", "generated", status="passed")
    _, failed_full = store.save_file(source, "b", "generated", status="failed")

    assert passed_full is None
    assert failed_full == "images/b.generated.full.png"
    assert Image.open(tmp_path / "report" / failed_full).size == (800, 600)


def test_none_mode_never_keeps_full_size(tmp_path: Path) -> None:
    store = ReportImageStore(tmp_path / "report", full_size="none")
    source = _write(tmp_path / "src.png")

    assert store.save_file(source, "b", "generated", status="failed")[1] is None


def test_all_mode_keeps_full_size_even_for_passed(tmp_path: Path) -> None:
    store = ReportImageStore(tmp_path / "report", full_size="all")
    source = _write(tmp_path / "src.png")

    assert store.save_file(source, "a", "generated", status="passed")[1] is not None


def test_smaller_images_are_not_enlarged(tmp_path: Path) -> None:
    store = ReportImageStore(tmp_path / "report", max_image_size=1000)
    source = _write(tmp_path / "src.png", size=(40, 30))

    thumb, _ = store.save_file(source, "sphere", "baseline", status="passed")

    assert Image.open(tmp_path / "report" / thumb).size == (40, 30)


def test_save_image_accepts_an_in_memory_image(tmp_path: Path) -> None:
    store = ReportImageStore(tmp_path / "report", max_image_size=100)

    thumb, _ = store.save_image(Image.new("RGB", (800, 600)), "sphere", "diff", status="failed")

    assert (tmp_path / "report" / thumb).is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summary_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pytest_pyvista.summary.store'`

- [ ] **Step 3: Write minimal implementation**

Create `pytest_pyvista/summary/store.py`:

```python
"""On-disk image store for the summary report."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Literal

from PIL import Image

FullSizeMode = Literal["none", "failing", "all"]

_IMAGES_SUBDIR = "images"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(name: str) -> str:
    """Reduce a test or image name to something safe to use as a filename."""
    return _UNSAFE.sub("_", name).strip("_")


class ReportImageStore:
    """Writes the report's copies of baseline, generated and difference images."""

    def __init__(self, report_dir: Path, *, max_image_size: int = 400, full_size: FullSizeMode = "failing") -> None:
        self.report_dir = Path(report_dir)
        self.max_image_size = max_image_size
        self.full_size = full_size
        self.images_dir = self.report_dir / _IMAGES_SUBDIR

    def save_file(self, source: Path, slug: str, role: str, *, status: str) -> tuple[str, str | None]:
        """Store an image read from ``source``. See :meth:`save_image`."""
        with Image.open(source) as image:
            return self.save_image(image, slug, role, status=status)

    def save_image(self, image: Image.Image, slug: str, role: str, *, status: str) -> tuple[str, str | None]:
        """
        Store a downscaled copy of ``image``, plus a full-resolution copy when retained.

        Returns ``(thumbnail, full)`` as POSIX paths relative to the report directory.
        ``full`` is ``None`` when this record does not retain a full-resolution copy.
        """
        self.images_dir.mkdir(parents=True, exist_ok=True)
        safe = slugify(slug)
        image = image.convert("RGB")

        full_relpath: str | None = None
        if self._keeps_full_size(status):
            full_name = f"{safe}.{role}.full.png"
            image.save(self.images_dir / full_name)
            full_relpath = f"{_IMAGES_SUBDIR}/{full_name}"

        thumbnail = image.copy()
        thumbnail.thumbnail((self.max_image_size, self.max_image_size))
        thumb_name = f"{safe}.{role}.png"
        thumbnail.save(self.images_dir / thumb_name)

        return f"{_IMAGES_SUBDIR}/{thumb_name}", full_relpath

    def _keeps_full_size(self, status: str) -> bool:
        if self.full_size == "all":
            return True
        if self.full_size == "none":
            return False
        return status != "passed"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_summary_store.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add pytest_pyvista/summary/store.py tests/test_summary_store.py
git commit -m "feat: add report image store with downscaling and full-size retention"
```

---

### Task 4: Status determination

**Files:**
- Create: `pytest_pyvista/summary/collect.py`
- Test: `tests/test_summary_collect.py`

**Interfaces:**
- Consumes: `ImageStatus`, `CacheWriteReason` from Task 1
- Produces: `determine_status(*, skipped: bool, baseline_existed: bool, cache_write_reason: CacheWriteReason | None, error: float | None, error_threshold: float, warning_threshold: float, matched_alternate: bool) -> ImageStatus`

This is pure logic, kept separate from the capture wiring in Task 6 so it can be exhaustively tested without a pytest session.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_collect.py`:

```python
"""Tests for pytest_pyvista.summary.collect."""

from __future__ import annotations

import pytest

from pytest_pyvista.summary.collect import determine_status


def _status(**overrides) -> str:
    kwargs = {
        "skipped": False,
        "baseline_existed": True,
        "cache_write_reason": None,
        "error": 0.0,
        "error_threshold": 500.0,
        "warning_threshold": 200.0,
        "matched_alternate": False,
    }
    kwargs.update(overrides)
    return determine_status(**kwargs)


def test_skipped_wins_over_everything() -> None:
    assert _status(skipped=True, baseline_existed=False, error=9999.0) == "skipped"


def test_absent_baseline_is_new() -> None:
    assert _status(baseline_existed=False) == "new"


def test_absent_baseline_written_by_add_missing_images_is_still_new() -> None:
    assert _status(baseline_existed=False, cache_write_reason="add_missing_images") == "new"


@pytest.mark.parametrize("reason", ["reset_image_cache", "reset_only_failed"])
def test_overwritten_existing_baseline_is_reset(reason: str) -> None:
    assert _status(cache_write_reason=reason, error=9999.0) == "reset"


def test_error_below_warning_threshold_passes() -> None:
    assert _status(error=199.0) == "passed"


def test_error_above_warning_threshold_warns() -> None:
    assert _status(error=201.0) == "warned"


def test_error_exactly_on_warning_threshold_passes() -> None:
    assert _status(error=200.0) == "passed"


def test_error_above_error_threshold_fails() -> None:
    assert _status(error=501.0) == "failed"


def test_error_exactly_on_error_threshold_warns() -> None:
    assert _status(error=500.0) == "warned"


def test_matching_an_alternate_baseline_warns_however_large_the_primary_error() -> None:
    assert _status(error=9999.0, matched_alternate=True) == "warned"


def test_uncomputable_error_fails() -> None:
    assert _status(error=None) == "failed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summary_collect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pytest_pyvista.summary.collect'`

- [ ] **Step 3: Write minimal implementation**

Create `pytest_pyvista/summary/collect.py`:

```python
"""Turning a single image comparison into a report record."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_pyvista.summary.record import CacheWriteReason
    from pytest_pyvista.summary.record import ImageStatus


def determine_status(  # noqa: PLR0911
    *,
    skipped: bool,
    baseline_existed: bool,
    cache_write_reason: CacheWriteReason | None,
    error: float | None,
    error_threshold: float,
    warning_threshold: float,
    matched_alternate: bool,
) -> ImageStatus:
    """
    Classify one generated image.

    ``error`` is the report's own comparison against the preserved prior baseline, and
    is ``None`` when no comparison was possible. ``matched_alternate`` marks a test that
    failed its primary baseline but matched another of its candidates — the plugin
    treats that as a warning rather than a failure.
    """
    if skipped:
        return "skipped"
    if not baseline_existed:
        return "new"
    if cache_write_reason in {"reset_image_cache", "reset_only_failed"}:
        return "reset"
    if matched_alternate:
        return "warned"
    if error is None or error > error_threshold:
        return "failed"
    if error > warning_threshold:
        return "warned"
    return "passed"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_summary_collect.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add pytest_pyvista/summary/collect.py tests/test_summary_collect.py
git commit -m "feat: add status determination for summary report records"
```

---

### Task 5: Plugin options

**Files:**
- Modify: `pytest_pyvista/pytest_pyvista.py` (option registration inside `pytest_addoption`; constant near line 45; validation in `pytest_configure` near line 956)
- Test: `tests/test_summary_options.py`

**Interfaces:**
- Consumes: `ALL_STATUSES` from Task 1, `FullSizeMode` from Task 3
- Produces: CLI flags `--summary_html`, `--summary_html_dir`, `--summary_html_include`, `--summary_html_max_image_size`, `--summary_html_full_size`, `--summary_html_embed`; matching ini options; module constant `PYVISTA_SUMMARY_RECORDS_DIRNAME = "pyvista_summary_records_dir"`; helper `_summary_html_enabled(config: pytest.Config) -> bool`

Note the existing helper names: `_add_unit_test_cli_option` registers a flag *and* adds it to `_UNIT_TEST_CLI_ARGS`, which `pytest_configure` uses to reject unit-test flags under `--doc_mode` ([`pytest_pyvista.py:942-954`](../../../pytest_pyvista/pytest_pyvista.py#L942-L954)). Registering the summary flags through it gives the doc-mode rejection for free.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_options.py`:

```python
"""Tests for the summary report's pytest options."""

from __future__ import annotations

import pytest

SIMPLE_TEST = """
    import pyvista as pv
    pv.OFF_SCREEN = True

    def test_sphere(verify_image_cache):
        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere())
        pl.show()
"""


def test_summary_html_is_rejected_under_doc_mode(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(SIMPLE_TEST)

    result = pytester.runpytest("--doc_mode", "--summary_html")

    result.stderr.fnmatch_lines(["*--summary_html cannot be used with --doc_mode enabled*"])


def test_summary_html_include_rejects_an_unknown_status(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(SIMPLE_TEST)

    result = pytester.runpytest("--summary_html", "--summary_html_include", "passed,bogus")

    result.stderr.fnmatch_lines(["*unknown status*bogus*"])


def test_summary_html_full_size_rejects_an_unknown_mode(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(SIMPLE_TEST)

    result = pytester.runpytest("--summary_html", "--summary_html_full_size", "bogus")

    assert result.ret != 0


def test_summary_html_dir_implies_enablement(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        from pytest_pyvista.pytest_pyvista import _summary_html_enabled

        def test_enabled(pytestconfig):
            assert _summary_html_enabled(pytestconfig)
        """
    )

    result = pytester.runpytest("--summary_html_dir", "somewhere")

    result.assert_outcomes(passed=1)


def test_summary_html_is_off_by_default(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        from pytest_pyvista.pytest_pyvista import _summary_html_enabled

        def test_disabled(pytestconfig):
            assert not _summary_html_enabled(pytestconfig)
        """
    )

    result = pytester.runpytest()

    result.assert_outcomes(passed=1)


def test_ini_can_enable_the_report(pytester: pytest.Pytester) -> None:
    pytester.makeini(
        """
        [pytest]
        summary_html = true
        """
    )
    pytester.makepyfile(
        """
        from pytest_pyvista.pytest_pyvista import _summary_html_enabled

        def test_enabled(pytestconfig):
            assert _summary_html_enabled(pytestconfig)
        """
    )

    result = pytester.runpytest()

    result.assert_outcomes(passed=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summary_options.py -v`
Expected: FAIL — `unrecognized arguments: --summary_html`

- [ ] **Step 3: Write minimal implementation**

In `pytest_pyvista/pytest_pyvista.py`, add the constant beside the other cache dirnames (near line 45):

```python
PYVISTA_SUMMARY_RECORDS_DIRNAME = "pyvista_summary_records_dir"
```

Add the import beside the other package imports (near line 33):

```python
from pytest_pyvista.summary.record import ALL_STATUSES
```

Inside `pytest_addoption`, at the end of `_add_unit_test_cli_and_ini_options`:

```python
        _add_unit_test_cli_option(
            "--summary_html",
            action="store_const",
            const=True,
            default=None,
            help="Generate an HTML image summary report at the end of the run.",
        )
        parser.addini("summary_html", type="bool", default=None, help="Generate an HTML image summary report.")

        _add_unit_test_cli_option(
            "--summary_html_dir",
            action="store",
            default=None,
            help="Directory for the HTML image summary report. Implies --summary_html.",
        )
        parser.addini("summary_html_dir", default=None, help="Directory for the HTML image summary report.")

        _add_unit_test_cli_option(
            "--summary_html_include",
            action="store",
            default=None,
            help="Comma-separated statuses to include in the summary report.",
        )
        parser.addini("summary_html_include", default=None, help="Comma-separated statuses to include in the summary report.")

        _add_unit_test_cli_option(
            "--summary_html_max_image_size",
            action="store",
            default=None,
            help="Longest-edge pixel limit for images shown inline in the summary report.",
        )
        parser.addini("summary_html_max_image_size", default=None, help="Longest-edge pixel limit for summary report images.")

        _add_unit_test_cli_option(
            "--summary_html_full_size",
            action="store",
            choices=["none", "failing", "all"],
            default=None,
            help="Which records retain a full-resolution copy in the summary report.",
        )
        parser.addini("summary_html_full_size", default=None, help="Which records retain a full-resolution image copy.")

        _add_unit_test_cli_option(
            "--summary_html_embed",
            action="store_const",
            const=True,
            default=None,
            help="Embed summary report images as data URIs in a single HTML file.",
        )
        parser.addini("summary_html_embed", type="bool", default=None, help="Embed summary report images as data URIs.")
```

Add the accessor helpers after `_get_option_from_config_or_ini` (near line 825):

```python
DEFAULT_SUMMARY_HTML_DIR = "image_test_report"
DEFAULT_SUMMARY_HTML_MAX_IMAGE_SIZE = 400
DEFAULT_SUMMARY_HTML_FULL_SIZE = "failing"


def _summary_html_enabled(pytestconfig: pytest.Config) -> bool:
    """Return True if the HTML summary report should be generated."""
    if _get_option_from_config_or_ini(pytestconfig, "summary_html"):
        return True
    return _get_option_from_config_or_ini(pytestconfig, "summary_html_dir") is not None


def _summary_html_statuses(pytestconfig: pytest.Config) -> tuple[str, ...]:
    """Return the statuses to write to the summary report."""
    value = _get_option_from_config_or_ini(pytestconfig, "summary_html_include")
    if not value:
        return ALL_STATUSES
    statuses = tuple(part.strip() for part in str(value).split(",") if part.strip())
    unknown = [status for status in statuses if status not in ALL_STATUSES]
    if unknown:
        msg = f"--summary_html_include: unknown status {unknown[0]!r}. Choose from: {', '.join(ALL_STATUSES)}"
        raise pytest.UsageError(msg)
    return statuses
```

In `pytest_configure`, after the existing CLI validation loop and before the `doc_mode` branch:

```python
    if _summary_html_enabled(config):
        # Validate eagerly so a typo fails the run rather than the report.
        _summary_html_statuses(config)
        if is_master:
            _make_config_cache_dir(config, PYVISTA_SUMMARY_RECORDS_DIRNAME, clean=True)
```

Add `PYVISTA_SUMMARY_RECORDS_DIRNAME` to the cleanup list in `pytest_unconfigure` (line 1154):

```python
        for dirname in [
            PYVISTA_FAILED_IMAGE_CACHE_DIRNAME,
            PYVISTA_GENERATED_IMAGE_CACHE_DIRNAME,
            PYVISTA_IMAGE_NAMES_CACHE_DIRNAME,
            PYVISTA_SUMMARY_RECORDS_DIRNAME,
        ]:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_summary_options.py -v`
Expected: PASS (6 passed)

Then confirm nothing regressed: `pytest tests/test_pyvista.py -q`

- [ ] **Step 5: Commit**

```bash
git add pytest_pyvista/pytest_pyvista.py tests/test_summary_options.py
git commit -m "feat: add summary report options and doc-mode rejection"
```

---

### Task 6: Capture records during comparison

**Files:**
- Create: `pytest_pyvista/summary/session.py`
- Modify: `pytest_pyvista/pytest_pyvista.py` (`VerifyImageCache.__init__` and `__call__`; `verify_image_cache` fixture near line 1016)
- Test: `tests/test_summary_capture.py`

**Interfaces:**
- Consumes: `ImageRecord`, `write_record` (Task 1), `compute_diff_image` (Task 2), `ReportImageStore` (Task 3), `determine_status` (Task 4), options (Task 5)
- Produces: `SummarySession` with `__init__(self, *, run_id: str, records_dir: Path, store: ReportImageStore, worker_id: str, statuses: tuple[str, ...], cache_dir: Path)` and `capture(self, *, test_name, image_name, call_index, baseline_source, generated_source, cache_destination, skipped, skip_reason, baseline_existed, cache_write_reason, error, error_threshold, warning_threshold, high_variance_test, matched_alternate, matched_baseline, candidate_baselines, image_format, env_info) -> ImageRecord | None`

The critical ordering constraint: the baseline must be copied into the store **before** the cache write at [`pytest_pyvista.py:520-521`](../../../pytest_pyvista/pytest_pyvista.py#L520-L521), and the report computes its own error against that preserved copy rather than trusting the plugin's comparison, which is image-against-itself under `--reset_image_cache`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_capture.py`:

```python
"""Tests for record capture during image comparison."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from pytest_pyvista.summary.record import read_records
from pytest_pyvista.summary.session import SummarySession
from pytest_pyvista.summary.store import ReportImageStore


def _write(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color).save(path)
    return path


def _session(tmp_path: Path, statuses: tuple[str, ...] = ()) -> SummarySession:
    from pytest_pyvista.summary.record import ALL_STATUSES

    return SummarySession(
        run_id="run-1",
        records_dir=tmp_path / "records",
        store=ReportImageStore(tmp_path / "report"),
        worker_id="master",
        statuses=statuses or ALL_STATUSES,
        cache_dir=tmp_path / "cache",
    )


def _capture(session: SummarySession, **overrides):
    kwargs = {
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": 0,
        "baseline_source": None,
        "generated_source": None,
        "cache_destination": None,
        "skipped": False,
        "skip_reason": None,
        "baseline_existed": True,
        "cache_write_reason": None,
        "error": None,
        "error_threshold": 500.0,
        "warning_threshold": 200.0,
        "high_variance_test": False,
        "matched_alternate": False,
        "matched_baseline": None,
        "candidate_baselines": [],
        "image_format": "png",
        "env_info": "env",
    }
    kwargs.update(overrides)
    return session.capture(**kwargs)


def test_capture_writes_a_record_and_all_three_images(tmp_path: Path) -> None:
    session = _session(tmp_path)
    baseline = _write(tmp_path / "cache" / "sphere.png", (0, 0, 0))
    generated = _write(tmp_path / "gen" / "sphere.png", (255, 255, 255))

    record = _capture(session, baseline_source=baseline, generated_source=generated)

    assert record.status == "failed"
    assert (tmp_path / "report" / record.baseline_image).is_file()
    assert (tmp_path / "report" / record.generated_image).is_file()
    assert (tmp_path / "report" / record.diff_image).is_file()
    assert read_records(tmp_path / "records") == [record]


def test_capture_computes_its_own_error_ignoring_the_caller(tmp_path: Path) -> None:
    session = _session(tmp_path)
    baseline = _write(tmp_path / "cache" / "sphere.png", (0, 0, 0))
    generated = _write(tmp_path / "gen" / "sphere.png", (255, 255, 255))

    record = _capture(session, baseline_source=baseline, generated_source=generated, error=0.0)

    assert record.error > 0.0


def test_identical_images_pass(tmp_path: Path) -> None:
    session = _session(tmp_path)
    baseline = _write(tmp_path / "cache" / "sphere.png", (10, 20, 30))
    generated = _write(tmp_path / "gen" / "sphere.png", (10, 20, 30))

    record = _capture(session, baseline_source=baseline, generated_source=generated)

    assert record.status == "passed"


def test_new_image_records_no_baseline_and_no_diff(tmp_path: Path) -> None:
    session = _session(tmp_path)
    generated = _write(tmp_path / "gen" / "sphere.png", (255, 0, 0))

    record = _capture(session, baseline_existed=False, generated_source=generated, cache_write_reason="add_missing_images")

    assert record.status == "new"
    assert record.baseline_image is None
    assert record.diff_image is None
    assert record.generated_image is not None
    assert record.cache_written is True


def test_skipped_image_records_the_baseline_only(tmp_path: Path) -> None:
    session = _session(tmp_path)
    baseline = _write(tmp_path / "cache" / "sphere.png", (0, 0, 0))

    record = _capture(session, skipped=True, skip_reason="windows_skip_image_cache", baseline_source=baseline)

    assert record.status == "skipped"
    assert record.baseline_image is not None
    assert record.generated_image is None
    assert record.skip_reason == "windows_skip_image_cache"


def test_size_mismatch_is_flagged_without_a_diff(tmp_path: Path) -> None:
    session = _session(tmp_path)
    baseline = _write(tmp_path / "cache" / "sphere.png", (0, 0, 0))
    generated = tmp_path / "gen" / "sphere.png"
    generated.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (41, 30), (0, 0, 0)).save(generated)

    record = _capture(session, baseline_source=baseline, generated_source=generated)

    assert record.size_mismatch is True
    assert record.diff_image is None
    assert record.error is None


def test_excluded_statuses_are_not_recorded(tmp_path: Path) -> None:
    session = _session(tmp_path, statuses=("failed",))
    baseline = _write(tmp_path / "cache" / "sphere.png", (10, 20, 30))
    generated = _write(tmp_path / "gen" / "sphere.png", (10, 20, 30))

    record = _capture(session, baseline_source=baseline, generated_source=generated)

    assert record is None
    assert read_records(tmp_path / "records") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summary_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pytest_pyvista.summary.session'`

- [ ] **Step 3: Write minimal implementation**

Create `pytest_pyvista/summary/session.py`:

```python
"""Per-session capture of summary report records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pytest_pyvista.summary.collect import determine_status
from pytest_pyvista.summary.diff import compute_diff_image
from pytest_pyvista.summary.record import ImageRecord
from pytest_pyvista.summary.record import write_record
from pytest_pyvista.summary.store import slugify

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_pyvista.summary.record import CacheWriteReason
    from pytest_pyvista.summary.store import ReportImageStore


class SummarySession:
    """Collects one record per generated image for the duration of a test session."""

    def __init__(
        self,
        *,
        run_id: str,
        records_dir: Path,
        store: ReportImageStore,
        worker_id: str,
        statuses: tuple[str, ...],
        cache_dir: Path,
    ) -> None:
        self.run_id = run_id
        self.records_dir = records_dir
        self.store = store
        self.worker_id = worker_id
        self.statuses = statuses
        self.cache_dir = cache_dir

    def capture(  # noqa: PLR0913
        self,
        *,
        test_name: str,
        image_name: str,
        call_index: int,
        baseline_source: Path | None,
        generated_source: Path | None,
        cache_destination: Path | None,
        skipped: bool,
        skip_reason: str | None,
        baseline_existed: bool,
        cache_write_reason: CacheWriteReason | None,
        error: float | None,
        error_threshold: float,
        warning_threshold: float,
        high_variance_test: bool,
        matched_alternate: bool,
        matched_baseline: str | None,
        candidate_baselines: list[str],
        image_format: str,
        env_info: str,
    ) -> ImageRecord | None:
        """
        Record one generated image.

        ``error`` from the caller is advisory only: the report recomputes it against the
        preserved baseline, because the plugin's own comparison is image-against-itself
        whenever the cache was written before comparing. Returns ``None`` when the
        resulting status is excluded by ``--summary_html_include``.
        """
        slug = slugify(f"{test_name}_{call_index}" if call_index else test_name)

        diff_image = None
        size_mismatch = False
        computed_error = error
        if baseline_source is not None and generated_source is not None:
            diff_image, size_mismatch = compute_diff_image(baseline_source, generated_source)
            computed_error = None if size_mismatch else self._error(baseline_source, generated_source)

        status = determine_status(
            skipped=skipped,
            baseline_existed=baseline_existed,
            cache_write_reason=cache_write_reason,
            error=computed_error,
            error_threshold=error_threshold,
            warning_threshold=warning_threshold,
            matched_alternate=matched_alternate,
        )
        if status not in self.statuses:
            return None

        baseline_rel = baseline_full = None
        if baseline_source is not None:
            baseline_rel, baseline_full = self.store.save_file(baseline_source, slug, "baseline", status=status)

        generated_rel = generated_full = None
        if generated_source is not None:
            generated_rel, generated_full = self.store.save_file(generated_source, slug, "generated", status=status)

        diff_rel = diff_full = None
        if diff_image is not None:
            diff_rel, diff_full = self.store.save_image(diff_image, slug, "diff", status=status)

        record = ImageRecord(
            run_id=self.run_id,
            test_name=test_name,
            image_name=image_name,
            call_index=call_index,
            status=status,
            error=computed_error,
            error_threshold=error_threshold,
            warning_threshold=warning_threshold,
            high_variance_test=high_variance_test,
            size_mismatch=size_mismatch,
            baseline_image=baseline_rel,
            generated_image=generated_rel,
            diff_image=diff_rel,
            baseline_image_full=baseline_full,
            generated_image_full=generated_full,
            diff_image_full=diff_full,
            matched_baseline=matched_baseline,
            candidate_baselines=candidate_baselines,
            cache_written=cache_write_reason is not None,
            cache_write_reason=cache_write_reason,
            skip_reason=skip_reason,
            generated_source=str(generated_source) if generated_source else None,
            cache_destination=str(cache_destination) if cache_destination else None,
            cache_dir=str(self.cache_dir),
            image_format=image_format,
            env_info=env_info,
        )
        write_record(self.records_dir, self.worker_id, record)
        return record

    @staticmethod
    def _error(baseline: Path, generated: Path) -> float | None:
        import pyvista  # noqa: PLC0415

        try:
            return float(pyvista.compare_images(str(generated), str(baseline)))
        except (RuntimeError, ValueError):
            return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_summary_capture.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Wire the session into `VerifyImageCache`**

In `pytest_pyvista/pytest_pyvista.py`, add a class attribute beside `reset_only_failed` (line 407):

```python
    summary_session = None
```

In `__call__`, capture the baseline **before** any cache write. Insert immediately after `current_cached_image = cached_image_paths[0]` (line 505):

```python
        summary = VerifyImageCache.summary_session
        preserved_baseline = None
        if summary is not None and current_cached_image.is_file():
            preserved_baseline = summary.preserve_baseline(current_cached_image, test_name, self.n_calls - 1)
```

In the skip branch (after `SKIPPED_CACHED_IMAGE_NAMES.add(image_name)`, line 493), record the skip. Because this returns before `image_filename` is computed, resolve the cache path locally:

```python
            if VerifyImageCache.summary_session is not None:
                skipped_baseline = Path(self.cache_dir, image_name)
                VerifyImageCache.summary_session.capture(
                    test_name=test_name,
                    image_name=image_name,
                    call_index=self.n_calls - 1,
                    baseline_source=skipped_baseline if skipped_baseline.is_file() else None,
                    generated_source=None,
                    cache_destination=skipped_baseline,
                    skipped=True,
                    skip_reason=self._skip_reason(),
                    baseline_existed=skipped_baseline.is_file(),
                    cache_write_reason=None,
                    error=None,
                    error_threshold=allowed_error,
                    warning_threshold=allowed_warning,
                    high_variance_test=self.high_variance_test,
                    matched_alternate=False,
                    matched_baseline=None,
                    candidate_baselines=[],
                    image_format=self.image_format,
                    env_info=str(self.env_info),
                )
            return
```

Add the reason helper as a method on `VerifyImageCache`:

```python
    def _skip_reason(self) -> str:
        """Return the flag responsible for skipping this image comparison."""
        if self.skip:
            return "skip"
        if self.ignore_image_cache:
            return "ignore_image_cache"
        if os.name == "nt" and self.windows_skip_image_cache:
            return "windows_skip_image_cache"
        if platform.system() == "Darwin" and self.macos_skip_image_cache:
            return "macos_skip_image_cache"
        return "skip"
```

Add `preserve_baseline` to `SummarySession`:

```python
    def preserve_baseline(self, path: Path, test_name: str, call_index: int) -> Path:
        """
        Copy a baseline aside before a cache-writing policy can overwrite it.

        Returns the path of the preserved copy, which is what the report diffs against.
        """
        import shutil  # noqa: PLC0415

        preserved_dir = self.records_dir / "preserved"
        preserved_dir.mkdir(parents=True, exist_ok=True)
        slug = slugify(f"{test_name}_{call_index}" if call_index else test_name)
        destination = preserved_dir / f"{slug}{path.suffix}"
        shutil.copy(path, destination)
        return destination
```

At the end of `__call__` — after the warning branch, so every path that reaches a comparison is covered — record the outcome. Insert as the final statements of `__call__`:

```python
        if summary is not None:
            summary.capture(
                test_name=test_name,
                image_name=image_name,
                call_index=self.n_calls - 1,
                baseline_source=preserved_baseline,
                generated_source=_get_generated_image_path(
                    parent=cast("Path", self.generated_image_dir),
                    image_name=image_name,
                    generate_subdirs=self.generate_subdirs,
                    env_info=self.env_info,
                ),
                cache_destination=current_cached_image,
                skipped=False,
                skip_reason=None,
                baseline_existed=preserved_baseline is not None,
                cache_write_reason=self._cache_write_reason(baseline_existed=preserved_baseline is not None, failed=bool(fail_msg)),
                error=None,
                error_threshold=allowed_error,
                warning_threshold=allowed_warning,
                high_variance_test=self.high_variance_test,
                matched_alternate=bool(warn_msg) and fail_msg is None and len(cached_image_paths) > 1,
                matched_baseline=str(current_cached_image),
                candidate_baselines=[str(path) for path in cached_image_paths],
                image_format=self.image_format,
                env_info=str(self.env_info),
            )
```

Add the reason helper:

```python
    def _cache_write_reason(self, *, baseline_existed: bool, failed: bool) -> str | None:
        """Return which policy wrote this image to the cache during this run, if any."""
        if self.add_missing_images and not baseline_existed:
            return "add_missing_images"
        if self.reset_image_cache and not self.reset_only_failed:
            return "reset_image_cache"
        if self.reset_only_failed and failed:
            return "reset_only_failed"
        return None
```

Because the failure path raises `RegressionError` at line 567 before reaching that capture, move the raise so the record is written first. Replace the `else` branch at lines 565-567 with a deferred raise: set `pending_error = RegressionError(fail_msg)` there, and after the capture block add:

```python
        if pending_error is not None:
            remove_plotter_close_callback()
            raise pending_error
```

Initialise `pending_error: RegressionError | None = None` at the top of `__call__`. The same applies to the `RegressionFileNotFoundError` path at line 518 — leave that one raising immediately, since no comparison happened and there is nothing to record.

In the `verify_image_cache` fixture (after line 1020), construct the session once per run:

```python
    VerifyImageCache.summary_session = _make_summary_session(pytestconfig)
```

Add the factory near the other option helpers, memoised on the config object so every test shares one session:

```python
def _make_summary_session(pytestconfig: pytest.Config) -> object | None:
    """Build (once) the SummarySession for this run, or None when the report is off."""
    if not _summary_html_enabled(pytestconfig):
        return None

    existing = getattr(pytestconfig, "_pyvista_summary_session", None)
    if existing is not None:
        return existing

    from pytest_pyvista.summary.session import SummarySession  # noqa: PLC0415
    from pytest_pyvista.summary.store import ReportImageStore  # noqa: PLC0415

    report_dir = pytestconfig.rootpath / str(_get_option_from_config_or_ini(pytestconfig, "summary_html_dir") or DEFAULT_SUMMARY_HTML_DIR)
    max_size = int(_get_option_from_config_or_ini(pytestconfig, "summary_html_max_image_size") or DEFAULT_SUMMARY_HTML_MAX_IMAGE_SIZE)
    full_size = str(_get_option_from_config_or_ini(pytestconfig, "summary_html_full_size") or DEFAULT_SUMMARY_HTML_FULL_SIZE)

    worker_input = getattr(pytestconfig, "workerinput", None)
    session = SummarySession(
        run_id=worker_input["pyvista_run_id"] if worker_input else str(uuid.uuid4()),
        records_dir=Path(getattr(pytestconfig, PYVISTA_SUMMARY_RECORDS_DIRNAME)),
        store=ReportImageStore(report_dir, max_image_size=max_size, full_size=full_size),  # type: ignore[arg-type]
        worker_id=worker_input["workerid"] if worker_input else "master",
        statuses=_summary_html_statuses(pytestconfig),
        cache_dir=cast("Path", _get_option_from_config_or_ini(pytestconfig, "image_cache_dir", is_dir=True)),
    )
    pytestconfig._pyvista_summary_session = session  # noqa: SLF001
    return session
```

For xdist, the run id and records directory must be shared. In `pytest_configure` where the records dir is created, store the run id on the config:

```python
        if is_master:
            _make_config_cache_dir(config, PYVISTA_SUMMARY_RECORDS_DIRNAME, clean=True)
            config.pyvista_run_id = str(uuid.uuid4())
```

and extend the existing `pytest_configure_node` (line 1001) to forward both:

```python
    def pytest_configure_node(node: xdist.workermanage.WorkerController) -> None:
        """Modify each xdist worker."""
        if paths := getattr(node.config, "paths", None):
            node.workerinput["paths"] = paths
        if run_id := getattr(node.config, "pyvista_run_id", None):
            node.workerinput["pyvista_run_id"] = run_id
            node.workerinput["pyvista_records_dir"] = str(getattr(node.config, PYVISTA_SUMMARY_RECORDS_DIRNAME))
```

and in `_make_summary_session`, prefer the forwarded records dir when running as a worker:

```python
    records_dir = Path(worker_input["pyvista_records_dir"]) if worker_input else Path(getattr(pytestconfig, PYVISTA_SUMMARY_RECORDS_DIRNAME))
```

Finally, when the report is enabled but `generated_image_dir` is unset, point it at a temp dir so renders exist to copy. In the `verify_image_cache` fixture where `generated_image_dir` is resolved, fall back to `_make_config_cache_dir(pytestconfig, PYVISTA_GENERATED_IMAGE_CACHE_DIRNAME)` when `_summary_html_enabled(pytestconfig)` is true.

- [ ] **Step 6: Run the full suite**

Run: `pytest tests/ -q`
Expected: PASS — existing tests unaffected, new capture tests pass

- [ ] **Step 7: Commit**

```bash
git add pytest_pyvista/summary/session.py pytest_pyvista/pytest_pyvista.py tests/test_summary_capture.py
git commit -m "feat: capture summary records during image comparison"
```

---

### Task 7: HTML rendering

**Files:**
- Create: `pytest_pyvista/summary/render.py`
- Create: `pytest_pyvista/summary/assets/report.css`
- Test: `tests/test_summary_render.py`

**Interfaces:**
- Consumes: `ImageRecord`, `ALL_STATUSES` (Task 1)
- Produces: `render_report(records: list[ImageRecord], *, run_id: str, metadata: dict[str, str], embed_dir: Path | None = None) -> str`, `write_report(records: list[ImageRecord], report_dir: Path, *, run_id: str, metadata: dict[str, str], embed: bool = False) -> Path`

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_render.py`:

```python
"""Tests for pytest_pyvista.summary.render."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from pytest_pyvista.summary.record import ImageRecord
from pytest_pyvista.summary.render import render_report
from pytest_pyvista.summary.render import write_report


def _record(**overrides) -> ImageRecord:
    kwargs = {
        "run_id": "run-1",
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": 0,
        "status": "failed",
        "error": 812.4,
        "error_threshold": 500.0,
        "baseline_image": "images/sphere.baseline.png",
        "generated_image": "images/sphere.generated.png",
        "diff_image": "images/sphere.diff.png",
    }
    kwargs.update(overrides)
    return ImageRecord(**kwargs)


def _metadata() -> dict[str, str]:
    return {"Generated": "2026-08-12T14:05:00Z", "Cache directory": "tests/image_cache"}


def test_report_contains_the_test_name_and_status() -> None:
    html = render_report([_record()], run_id="run-1", metadata=_metadata())

    assert "test_sphere" in html
    assert 'data-status="failed"' in html


def test_report_embeds_the_run_id_for_approval_state() -> None:
    html = render_report([_record()], run_id="run-1", metadata=_metadata())

    assert 'data-run-id="run-1"' in html


def test_report_escapes_test_names() -> None:
    html = render_report([_record(test_name="test_<script>")], run_id="run-1", metadata=_metadata())

    assert "test_&lt;script&gt;" in html
    assert "test_<script>" not in html


def test_approvable_statuses_get_a_checkbox() -> None:
    for status in ("new", "failed", "warned"):
        html = render_report([_record(status=status)], run_id="run-1", metadata=_metadata())
        assert 'class="approve"' in html, status


def test_already_cached_records_get_a_chip_not_a_checkbox() -> None:
    html = render_report(
        [_record(status="new", cache_written=True, cache_write_reason="add_missing_images")],
        run_id="run-1",
        metadata=_metadata(),
    )

    assert 'class="approve"' not in html
    assert "add_missing_images" in html


def test_passed_and_skipped_records_get_no_checkbox() -> None:
    for status in ("passed", "skipped"):
        html = render_report([_record(status=status)], run_id="run-1", metadata=_metadata())
        assert 'class="approve"' not in html, status


def test_size_mismatch_replaces_the_diff_panel() -> None:
    html = render_report([_record(size_mismatch=True, diff_image=None, error=None)], run_id="run-1", metadata=_metadata())

    assert "size mismatch" in html.lower()


def test_images_are_lazy_loaded() -> None:
    html = render_report([_record()], run_id="run-1", metadata=_metadata())

    assert 'loading="lazy"' in html


def test_empty_report_renders_without_crashing() -> None:
    html = render_report([], run_id="run-1", metadata=_metadata())

    assert "No image tests" in html


def test_write_report_creates_index_html(tmp_path: Path) -> None:
    path = write_report([_record()], tmp_path / "report", run_id="run-1", metadata=_metadata())

    assert path == tmp_path / "report" / "index.html"
    assert path.is_file()


def test_embed_mode_inlines_images_as_data_uris(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    images = report_dir / "images"
    images.mkdir(parents=True)
    for role in ("baseline", "generated", "diff"):
        Image.new("RGB", (4, 4), (1, 2, 3)).save(images / f"sphere.{role}.png")

    write_report([_record()], report_dir, run_id="run-1", metadata=_metadata(), embed=True)
    html = (report_dir / "index.html").read_text(encoding="utf-8")

    assert "data:image/png;base64," in html
    assert 'src="images/' not in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summary_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pytest_pyvista.summary.render'`

- [ ] **Step 3: Write minimal implementation**

Create `pytest_pyvista/summary/assets/report.css`:

```css
:root {
  --bg: #ffffff; --fg: #1b1f24; --muted: #5b6570; --line: #d8dee4; --card: #f6f8fa;
  --passed: #1a7f37; --warned: #bc4c00; --failed: #cf222e; --skipped: #6e7781; --new: #0969da; --reset: #8250df;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #30363d; --card: #161b22;
    --passed: #3fb950; --warned: #d29922; --failed: #f85149; --skipped: #8b949e; --new: #58a6ff; --reset: #bc8cff;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 1.5rem; background: var(--bg); color: var(--fg);
       font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
h1 { font-size: 1.4rem; margin: 0 0 .5rem; }
.meta { color: var(--muted); font-size: .85rem; margin-bottom: 1rem; }
.meta dl { display: grid; grid-template-columns: max-content 1fr; gap: .1rem .75rem; margin: 0; }
.meta dt { font-weight: 600; }
.meta dd { margin: 0; }
.controls { position: sticky; top: 0; z-index: 2; background: var(--bg); border-bottom: 1px solid var(--line);
            padding: .75rem 0; margin-bottom: 1rem; display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; }
.controls label { display: inline-flex; align-items: center; gap: .3rem; }
.controls input[type="search"], .controls select { padding: .3rem .5rem; border: 1px solid var(--line);
            border-radius: 6px; background: var(--bg); color: var(--fg); }
.badge { display: inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .75rem;
         font-weight: 600; color: #fff; text-transform: uppercase; letter-spacing: .03em; }
.badge.passed { background: var(--passed); } .badge.warned { background: var(--warned); }
.badge.failed { background: var(--failed); } .badge.skipped { background: var(--skipped); }
.badge.new { background: var(--new); } .badge.reset { background: var(--reset); }
.card { border: 1px solid var(--line); border-radius: 8px; margin-bottom: 1rem; overflow: hidden; }
.card.hidden { display: none; }
.card > header { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center;
                 padding: .6rem .8rem; background: var(--card); border-bottom: 1px solid var(--line); }
.card > header .name { font-weight: 600; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.card > header .env { color: var(--muted); font-size: .78rem; }
.panels { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; padding: .8rem; overflow-x: auto; }
.panel { min-width: 0; }
.panel h3 { font-size: .78rem; margin: 0 0 .3rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.panel img { width: 100%; height: auto; display: block; border: 1px solid var(--line); border-radius: 4px; background: #fff; }
.panel .placeholder { border: 1px dashed var(--line); border-radius: 4px; padding: 1.5rem .5rem;
                      text-align: center; color: var(--muted); font-size: .8rem; }
.info { display: grid; grid-template-columns: max-content 1fr; gap: .15rem .75rem;
        padding: 0 .8rem .8rem; font-size: .85rem; }
.info dt { color: var(--muted); }
.info dd { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.chip { display: inline-block; padding: .1rem .5rem; border-radius: 6px; background: var(--card);
        border: 1px solid var(--line); font-size: .8rem; }
.approve { display: inline-flex; align-items: center; gap: .35rem; }
footer { position: sticky; bottom: 0; background: var(--bg); border-top: 1px solid var(--line);
         padding: .75rem 0; display: flex; gap: .75rem; align-items: center; }
button { padding: .4rem .8rem; border: 1px solid var(--line); border-radius: 6px;
         background: var(--card); color: var(--fg); cursor: pointer; font: inherit; }
button:hover { border-color: var(--muted); }
.notice { padding: .5rem .75rem; border: 1px solid var(--line); border-left: 3px solid var(--new);
          border-radius: 4px; margin-bottom: 1rem; }
.empty { color: var(--muted); padding: 2rem 0; text-align: center; }
@media (max-width: 720px) { .panels { grid-template-columns: 1fr; } }
```

Create `pytest_pyvista/summary/render.py`:

```python
"""Rendering summary report records to HTML."""

from __future__ import annotations

import base64
import html
from importlib import resources
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_pyvista.summary.record import ImageRecord

APPROVABLE_STATUSES = frozenset({"new", "failed", "warned"})

_PANELS = (("baseline", "Baseline"), ("generated", "Generated"), ("diff", "Difference"))


def _asset(name: str) -> str:
    return resources.files("pytest_pyvista.summary.assets").joinpath(name).read_text(encoding="utf-8")


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _panel(record: ImageRecord, role: str, label: str, embed_dir: Path | None) -> str:
    source = getattr(record, f"{role}_image")
    full = getattr(record, f"{role}_image_full")

    if source is None:
        if role == "diff" and record.size_mismatch:
            note = "Size mismatch &mdash; a pixel difference is undefined"
        elif role == "diff":
            note = "No difference to show"
        elif role == "baseline":
            note = "No baseline in the cache"
        else:
            note = "No image generated"
        return f'<div class="panel"><h3>{label}</h3><p class="placeholder">{note}</p></div>'

    src = _data_uri(embed_dir / source) if embed_dir else html.escape(source)
    href = html.escape(full or source) if not embed_dir else src
    return (
        f'<div class="panel"><h3>{label}</h3>'
        f'<a href="{href}" target="_blank" rel="noreferrer">'
        f'<img loading="lazy" src="{src}" alt="{label} image for {html.escape(record.test_name)}"></a></div>'
    )


def _info(record: ImageRecord) -> str:
    rows: list[tuple[str, str]] = []

    if record.size_mismatch:
        rows.append(("Error", "not comparable (size mismatch)"))
    elif record.error is not None:
        threshold = f"{record.error_threshold:g}" if record.error_threshold is not None else "n/a"
        variance = " (high variance)" if record.high_variance_test else ""
        rows.append(("Image regression error", f"{record.error:g} / {threshold}{variance}"))

    rows.append(("Status", record.status))
    if record.skip_reason:
        rows.append(("Skipped by", record.skip_reason))
    if record.cache_write_reason:
        rows.append(("Approval reason", f"Already in cache &mdash; written by --{record.cache_write_reason}"))
    if record.matched_baseline and len(record.candidate_baselines) > 1:
        index = record.candidate_baselines.index(record.matched_baseline) + 1
        rows.append(("Matched baseline", f"{record.matched_baseline} ({index} of {len(record.candidate_baselines)})"))
    if record.env_info:
        rows.append(("Environment", record.env_info))

    cells = "".join(f"<dt>{html.escape(key)}</dt><dd>{value}</dd>" for key, value in rows)
    return f'<dl class="info">{cells}</dl>'


def _approval(record: ImageRecord) -> str:
    if record.cache_written:
        reason = html.escape(record.cache_write_reason or "policy")
        return f'<span class="chip">&check; In cache &mdash; --{reason}</span>'
    if record.status in APPROVABLE_STATUSES:
        return '<label class="approve"><input type="checkbox"> Approve this image</label>'
    return ""


def _card(record: ImageRecord, embed_dir: Path | None) -> str:
    key = html.escape(f"{record.test_name}::{record.call_index}")
    name = html.escape(record.test_name if not record.call_index else f"{record.test_name} [{record.call_index}]")
    panels = "".join(_panel(record, role, label, embed_dir) for role, label in _PANELS)
    error = "" if record.error is None else f'{record.error:g}'
    return (
        f'<article class="card" data-status="{record.status}" data-key="{key}" '
        f'data-name="{html.escape(record.test_name.lower())}" data-error="{error or 0}">'
        f'<header><span class="badge {record.status}">{record.status}</span>'
        f'<span class="name">{name}</span>'
        f'<span class="env">{html.escape(record.env_info)}</span>'
        f'<span style="margin-left:auto">{_approval(record)}</span></header>'
        f'<div class="panels">{panels}</div>{_info(record)}</article>'
    )


def render_report(records: list[ImageRecord], *, run_id: str, metadata: dict[str, str], embed_dir: Path | None = None) -> str:
    """Render the complete report page as a single HTML string."""
    from pytest_pyvista.summary.record import ALL_STATUSES  # noqa: PLC0415

    counts = {status: sum(1 for record in records if record.status == status) for status in ALL_STATUSES}

    filters = "".join(
        f'<label><input type="checkbox" class="status-filter" value="{status}" checked> '
        f'<span class="badge {status}">{status}</span> {counts[status]}</label>'
        for status in ALL_STATUSES
    )
    meta_rows = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in metadata.items())
    cards = "".join(_card(record, embed_dir) for record in records) or '<p class="empty">No image tests were recorded in this run.</p>'

    manifest = json.dumps(
        {
            "run_id": run_id,
            "cache_dir": records[0].cache_dir if records else "",
            "image_format": records[0].image_format if records else "png",
            "records": [
                {
                    "key": f"{record.test_name}::{record.call_index}",
                    "test_name": record.test_name,
                    "image_name": record.image_name,
                    "call_index": record.call_index,
                    "status": record.status,
                    "source": record.generated_source,
                    "destination": record.cache_destination,
                }
                for record in records
                if not record.cache_written and record.status in APPROVABLE_STATUSES
            ],
        }
    )

    return f"""<!doctype html>
<html lang="en" data-run-id="{html.escape(run_id)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PyVista Image Test Report</title>
<style>{_asset("report.css")}</style>
</head>
<body>
<h1>PyVista Image Test Report</h1>
<div class="meta"><dl>{meta_rows}</dl></div>
<div id="notice" class="notice" hidden></div>
<div class="controls">
  {filters}
  <input type="search" id="search" placeholder="Filter by test name" aria-label="Filter by test name">
  <select id="sort" aria-label="Sort order">
    <option value="error">Sort by error</option>
    <option value="name">Sort by name</option>
  </select>
  <button type="button" id="accept-new" hidden>Accept all new</button>
</div>
<main id="cards">{cards}</main>
<footer>
  <button type="button" id="export">Export approvals</button>
  <span id="count">0 images selected for approval</span>
</footer>
<script id="manifest" type="application/json">{manifest}</script>
<script>{_asset("report.js")}</script>
</body>
</html>
"""


def write_report(
    records: list[ImageRecord],
    report_dir: Path,
    *,
    run_id: str,
    metadata: dict[str, str],
    embed: bool = False,
) -> Path:
    """Write ``index.html`` into ``report_dir`` and return its path."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    html_text = render_report(records, run_id=run_id, metadata=metadata, embed_dir=report_dir if embed else None)
    path = report_dir / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return path
```

Create a placeholder `pytest_pyvista/summary/assets/report.js` so `_asset` resolves — Task 8 fills it in:

```javascript
// Interactive behaviour is implemented in Task 8.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_summary_render.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add pytest_pyvista/summary/render.py pytest_pyvista/summary/assets/ tests/test_summary_render.py
git commit -m "feat: render summary report records to HTML"
```

---

### Task 8: Interactive filters and approval export

**Files:**
- Modify: `pytest_pyvista/summary/assets/report.js` (replace the placeholder)
- Test: `tests/test_summary_interactive.py`

**Interfaces:**
- Consumes: the DOM contract from Task 7 — `data-run-id` on `<html>`, `.status-filter`, `#search`, `#sort`, `#accept-new`, `#export`, `#count`, `#notice`, `#manifest`, `article.card[data-status][data-key][data-name][data-error]`, `input[type=checkbox]` inside `label.approve`
- Produces: no Python interface; the exported manifest shape defined in Task 10

Behaviour is asserted by checking the shipped script satisfies the contract; the interaction itself is verified manually in Step 5 (no JS test runner is in this project's toolchain, and adding one is out of scope).

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_interactive.py`:

```python
"""Contract tests for the summary report's client-side script."""

from __future__ import annotations

from importlib import resources

import pytest

SCRIPT = resources.files("pytest_pyvista.summary.assets").joinpath("report.js").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "hook",
    [
        "status-filter",
        "accept-new",
        "export",
        "manifest",
        "localStorage",
        "pytest-pyvista:approvals:",
    ],
)
def test_script_wires_up_each_documented_hook(hook: str) -> None:
    assert hook in SCRIPT


def test_script_discards_state_from_a_different_run() -> None:
    assert "run_id" in SCRIPT
    assert "notice" in SCRIPT


def test_script_prunes_old_keys() -> None:
    assert "PRUNE_AFTER_DAYS" in SCRIPT


def test_script_has_no_external_requests() -> None:
    for forbidden in ("fetch(", "XMLHttpRequest", "https://", "http://"):
        assert forbidden not in SCRIPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summary_interactive.py -v`
Expected: FAIL — the placeholder script contains none of these

- [ ] **Step 3: Write minimal implementation**

Replace `pytest_pyvista/summary/assets/report.js`:

```javascript
(function () {
  "use strict";

  var PRUNE_AFTER_DAYS = 30;
  var runId = document.documentElement.getAttribute("data-run-id");
  var storeKey = "pytest-pyvista:approvals:" + runId;
  var manifest = JSON.parse(document.getElementById("manifest").textContent);
  var byKey = {};
  manifest.records.forEach(function (record) { byKey[record.key] = record; });

  var cards = Array.prototype.slice.call(document.querySelectorAll("article.card"));
  var filters = Array.prototype.slice.call(document.querySelectorAll(".status-filter"));
  var search = document.getElementById("search");
  var sort = document.getElementById("sort");
  var count = document.getElementById("count");
  var notice = document.getElementById("notice");
  var acceptNew = document.getElementById("accept-new");

  function pruneOldKeys() {
    var cutoff = Date.now() - PRUNE_AFTER_DAYS * 86400000;
    Object.keys(localStorage).forEach(function (key) {
      if (key.indexOf("pytest-pyvista:approvals:") !== 0) { return; }
      try {
        var saved = JSON.parse(localStorage.getItem(key));
        if (!saved.saved_at || saved.saved_at < cutoff) { localStorage.removeItem(key); }
      } catch (err) { localStorage.removeItem(key); }
    });
  }

  function load() {
    try {
      var saved = JSON.parse(localStorage.getItem(storeKey));
      if (!saved || saved.run_id !== runId) { return {}; }
      return saved.approved || {};
    } catch (err) { return {}; }
  }

  function save(approved) {
    localStorage.setItem(storeKey, JSON.stringify({ run_id: runId, saved_at: Date.now(), approved: approved }));
  }

  function staleKeysWereDiscarded() {
    var found = false;
    Object.keys(localStorage).forEach(function (key) {
      if (key.indexOf("pytest-pyvista:approvals:") === 0 && key !== storeKey) { found = true; }
    });
    return found;
  }

  var approved = load();

  if (staleKeysWereDiscarded()) {
    notice.textContent = "Approvals from an earlier run were not restored: this report describes different images.";
    notice.hidden = false;
  }

  function refreshCount() {
    var total = Object.keys(approved).filter(function (key) { return approved[key]; }).length;
    count.textContent = total + (total === 1 ? " image" : " images") + " selected for approval";
  }

  function applyFilters() {
    var wanted = {};
    filters.forEach(function (box) { if (box.checked) { wanted[box.value] = true; } });
    var term = search.value.trim().toLowerCase();
    cards.forEach(function (card) {
      var visible = wanted[card.getAttribute("data-status")] === true &&
        (term === "" || card.getAttribute("data-name").indexOf(term) !== -1);
      card.classList.toggle("hidden", !visible);
    });
  }

  function applySort() {
    var mode = sort.value;
    var main = document.getElementById("cards");
    cards.slice().sort(function (a, b) {
      if (mode === "name") {
        return a.getAttribute("data-name").localeCompare(b.getAttribute("data-name"));
      }
      return parseFloat(b.getAttribute("data-error")) - parseFloat(a.getAttribute("data-error"));
    }).forEach(function (card) { main.appendChild(card); });
  }

  cards.forEach(function (card) {
    var box = card.querySelector("label.approve input[type=checkbox]");
    if (!box) { return; }
    var key = card.getAttribute("data-key");
    box.checked = approved[key] === true;
    box.addEventListener("change", function () {
      if (box.checked) { approved[key] = true; } else { delete approved[key]; }
      save(approved);
      refreshCount();
    });
  });

  var newCards = cards.filter(function (card) {
    return card.getAttribute("data-status") === "new" && card.querySelector("label.approve input[type=checkbox]");
  });
  if (newCards.length) { acceptNew.hidden = false; }

  acceptNew.addEventListener("click", function () {
    newCards.forEach(function (card) {
      var box = card.querySelector("label.approve input[type=checkbox]");
      if (!box.checked) { box.checked = true; box.dispatchEvent(new Event("change")); }
    });
  });

  document.getElementById("export").addEventListener("click", function () {
    var selected = Object.keys(approved).filter(function (key) { return approved[key] && byKey[key]; });
    var payload = {
      schema_version: 1,
      run_id: runId,
      exported_at: new Date().toISOString(),
      cache_dir: manifest.cache_dir,
      image_format: manifest.image_format,
      approved: selected.map(function (key) {
        var record = byKey[key];
        return {
          test_name: record.test_name,
          image_name: record.image_name,
          call_index: record.call_index,
          status: record.status,
          source: record.source,
          destination: record.destination
        };
      })
    };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "approvals.json";
    link.click();
    URL.revokeObjectURL(link.href);
  });

  filters.forEach(function (box) { box.addEventListener("change", applyFilters); });
  search.addEventListener("input", applyFilters);
  sort.addEventListener("change", applySort);

  pruneOldKeys();
  applyFilters();
  applySort();
  refreshCount();
})();
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_summary_interactive.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Verify the interaction by hand**

Generate a report against this repo's own test images, open it, and confirm each of:
- unchecking a status hides those cards; the search box narrows further
- switching sort reorders cards
- ticking an approval box updates the footer count; reloading the page preserves it
- **Accept all new** ticks every New card and nothing else
- **Export approvals** downloads `approvals.json` containing exactly the ticked entries
- cards already in cache show a chip and no checkbox
- the page is legible in both light and dark system themes

- [ ] **Step 6: Commit**

```bash
git add pytest_pyvista/summary/assets/report.js tests/test_summary_interactive.py
git commit -m "feat: add interactive filtering and approval export to the report"
```

---

### Task 9: Generate the report at end of run

**Files:**
- Modify: `pytest_pyvista/pytest_pyvista.py` (`pytest_terminal_summary`, line 724)
- Test: `tests/test_summary_integration.py`

**Interfaces:**
- Consumes: `read_records` (Task 1), `write_report` (Task 7), options (Task 5), records written by Task 6
- Produces: an `index.html` at the configured report directory, and a terminal line naming it

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_integration.py`:

```python
"""End-to-end tests for the image summary report."""

from __future__ import annotations

from pathlib import Path

import pytest

CACHE_DIR = "image_cache_dir"

TEST_FILE = """
    import pyvista as pv
    pv.OFF_SCREEN = True

    def test_sphere(verify_image_cache):
        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere(), color="red")
        pl.show()
"""


def _report(pytester: pytest.Pytester) -> str:
    return Path(pytester.path, "image_test_report", "index.html").read_text(encoding="utf-8")


def test_report_is_generated_with_no_other_directories_configured(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(TEST_FILE)

    result = pytester.runpytest("--summary_html", "--add_missing_images")

    result.assert_outcomes(passed=1)
    assert Path(pytester.path, "image_test_report", "index.html").is_file()
    assert "test_sphere" in _report(pytester)


def test_report_path_is_printed_to_the_terminal(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(TEST_FILE)

    result = pytester.runpytest("--summary_html", "--add_missing_images")

    result.stdout.fnmatch_lines(["*image summary report*index.html*"])


def test_new_image_is_reported_as_new(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(TEST_FILE)

    pytester.runpytest("--summary_html", "--add_missing_images")

    assert 'data-status="new"' in _report(pytester)
    assert "add_missing_images" in _report(pytester)


def test_matching_image_is_reported_as_passed(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest("--add_missing_images")

    pytester.runpytest("--summary_html")

    assert 'data-status="passed"' in _report(pytester)


def test_reset_image_cache_is_reported_as_reset_with_a_real_diff(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest("--add_missing_images")
    pytester.makepyfile(TEST_FILE.replace('color="red"', 'color="blue"'))

    pytester.runpytest("--summary_html", "--reset_image_cache")

    html = _report(pytester)
    assert 'data-status="reset"' in html
    assert "reset_image_cache" in html
    assert "sphere.diff.png" in html


def test_report_is_generated_when_a_test_fails(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest("--add_missing_images")
    pytester.makepyfile(TEST_FILE.replace('color="red"', 'color="blue"'))

    result = pytester.runpytest("--summary_html")

    result.assert_outcomes(failed=1)
    assert 'data-status="failed"' in _report(pytester)


def test_skipped_test_is_reported_as_skipped(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True

        def test_sphere(verify_image_cache):
            verify_image_cache.skip = True
            pl = pv.Plotter()
            pl.add_mesh(pv.Sphere())
            pl.show()
        """
    )

    pytester.runpytest("--summary_html")

    assert 'data-status="skipped"' in _report(pytester)


def test_include_filter_restricts_what_is_written(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(TEST_FILE)

    pytester.runpytest("--summary_html", "--add_missing_images", "--summary_html_include", "failed")

    assert "No image tests" in _report(pytester)


def test_full_size_modes_control_retained_copies(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest("--add_missing_images")

    pytester.runpytest("--summary_html", "--summary_html_full_size", "none")
    assert not list(Path(pytester.path, "image_test_report", "images").glob("*.full.png"))

    pytester.runpytest("--summary_html", "--summary_html_full_size", "all")
    assert list(Path(pytester.path, "image_test_report", "images").glob("*.full.png"))


def test_custom_report_directory_is_honoured(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(TEST_FILE)

    pytester.runpytest("--summary_html_dir", "reports/images", "--add_missing_images")

    assert Path(pytester.path, "reports", "images", "index.html").is_file()


def test_one_report_is_produced_under_xdist(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_a="""
        import pyvista as pv
        pv.OFF_SCREEN = True

        def test_a(verify_image_cache):
            pl = pv.Plotter(); pl.add_mesh(pv.Sphere()); pl.show()
        """,
        test_b="""
        import pyvista as pv
        pv.OFF_SCREEN = True

        def test_b(verify_image_cache):
            pl = pv.Plotter(); pl.add_mesh(pv.Cube()); pl.show()
        """,
    )

    pytester.runpytest("-n", "2", "--summary_html", "--add_missing_images")

    html = _report(pytester)
    assert "test_a" in html
    assert "test_b" in html
    assert len(list(Path(pytester.path).rglob("index.html"))) == 1


def test_embed_mode_produces_a_single_file(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(TEST_FILE)

    pytester.runpytest("--summary_html", "--summary_html_embed", "--add_missing_images")

    assert "data:image/png;base64," in _report(pytester)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summary_integration.py -v`
Expected: FAIL — no `index.html` is written

- [ ] **Step 3: Write minimal implementation**

In `pytest_pyvista/pytest_pyvista.py`, at the end of `pytest_terminal_summary` (before the `VISITED_CACHED_IMAGE_NAMES.clear()` calls at line 764), add:

```python
    if _summary_html_enabled(config):
        _write_summary_report(config, terminalreporter)
```

Add the writer beside the other summary helpers:

```python
def _write_summary_report(config: pytest.Config, terminalreporter) -> None:  # noqa: ANN001
    """Combine every worker's records and write the HTML report."""
    from pytest_pyvista.summary.record import read_records  # noqa: PLC0415
    from pytest_pyvista.summary.render import write_report  # noqa: PLC0415

    records_dir = getattr(config, PYVISTA_SUMMARY_RECORDS_DIRNAME, None)
    if records_dir is None:
        return

    records = read_records(Path(records_dir))
    report_dir = config.rootpath / str(_get_option_from_config_or_ini(config, "summary_html_dir") or DEFAULT_SUMMARY_HTML_DIR)
    cache_dir = _get_option_from_config_or_ini(config, "image_cache_dir", is_dir=True)

    metadata = {
        "Generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Run id": str(getattr(config, "pyvista_run_id", "")),
        "Cache directory": str(cache_dir),
        "Python": platform.python_version(),
        "PyVista": pyvista.__version__,
        "VTK": vtkmodules.__version__,
        "pytest": pytest.__version__,
    }

    path = write_report(
        records,
        report_dir,
        run_id=str(getattr(config, "pyvista_run_id", "")),
        metadata=metadata,
        embed=bool(_get_option_from_config_or_ini(config, "summary_html_embed")),
    )

    terminalreporter.ensure_newline()
    terminalreporter.write_line(f"pytest-pyvista image summary report: {path}")
```

Add the imports at the top of the module, in sorted position:

```python
from datetime import datetime
from datetime import timezone
```

Note `pytest_terminal_summary` already returns early on xdist workers (line 726-728), so only the master writes the report.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_summary_integration.py -v`
Expected: PASS (12 passed)

Then the full suite: `pytest tests/ -q`

- [ ] **Step 5: Commit**

```bash
git add pytest_pyvista/pytest_pyvista.py tests/test_summary_integration.py
git commit -m "feat: write the image summary report at end of run"
```

---

### Task 10: Manifest validation

**Files:**
- Create: `pytest_pyvista/summary/approve.py`
- Test: `tests/test_summary_approve.py`

**Interfaces:**
- Consumes: `SCHEMA_VERSION` (Task 1)
- Produces: `ManifestError` (Exception), `ApprovedImage` (dataclass with `test_name: str`, `image_name: str`, `call_index: int`, `status: str`, `source: Path`, `destination: Path`), `load_manifest(path: Path, *, cache_dir: Path, target_root: Path, source_root: Path, force: bool = False) -> list[ApprovedImage]`

Task 11 adds the copying and the CLI on top of this.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_approve.py`:

```python
"""Tests for pytest_pyvista.summary.approve."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pytest_pyvista.summary.approve import ManifestError
from pytest_pyvista.summary.approve import load_manifest


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    source_root = tmp_path / "generated"
    cache_dir = tmp_path / "cache"
    source_root.mkdir()
    cache_dir.mkdir()
    (source_root / "sphere.png").write_bytes(b"png")
    return {"root": tmp_path, "source_root": source_root, "cache_dir": cache_dir}


def _manifest(workspace: dict[str, Path], **overrides) -> Path:
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
            }
        ],
    }
    payload.update(overrides)
    path = workspace["root"] / "approvals.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load(workspace: dict[str, Path], path: Path, **kwargs):
    return load_manifest(
        path,
        cache_dir=workspace["cache_dir"],
        target_root=kwargs.pop("target_root", workspace["cache_dir"]),
        source_root=workspace["source_root"],
        **kwargs,
    )


def test_valid_manifest_loads(workspace) -> None:
    approved = _load(workspace, _manifest(workspace))

    assert len(approved) == 1
    assert approved[0].test_name == "test_sphere"


def test_unknown_schema_version_is_rejected(workspace) -> None:
    with pytest.raises(ManifestError, match="schema version"):
        _load(workspace, _manifest(workspace, schema_version=99))


def test_mismatched_cache_dir_is_rejected(workspace) -> None:
    with pytest.raises(ManifestError, match="cache director"):
        _load(workspace, _manifest(workspace, cache_dir="/somewhere/else"))


def test_mismatched_cache_dir_is_allowed_with_force(workspace) -> None:
    assert len(_load(workspace, _manifest(workspace, cache_dir="/somewhere/else"), force=True)) == 1


def test_source_outside_the_generated_root_is_rejected(workspace) -> None:
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


def test_destination_escaping_the_target_root_is_rejected(workspace) -> None:
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


def test_missing_source_file_is_rejected(workspace) -> None:
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


def test_malformed_json_is_rejected(workspace) -> None:
    path = workspace["root"] / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ManifestError, match="not valid JSON"):
        _load(workspace, path)


def test_missing_required_key_is_rejected(workspace) -> None:
    entry = {"test_name": "test_x", "image_name": "x.png", "call_index": 0, "status": "new"}

    with pytest.raises(ManifestError, match="missing"):
        _load(workspace, _manifest(workspace, approved=[entry]))


def test_empty_approval_list_loads_as_empty(workspace) -> None:
    assert _load(workspace, _manifest(workspace, approved=[])) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summary_approve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pytest_pyvista.summary.approve'`

- [ ] **Step 3: Write minimal implementation**

Create `pytest_pyvista/summary/approve.py`:

```python
"""Applying exported approvals to the image cache."""

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
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


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

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        msg = f"Unsupported manifest schema version {version!r}; this pytest-pyvista expects {SCHEMA_VERSION}."
        raise ManifestError(msg)

    manifest_cache = payload.get("cache_dir", "")
    if not force and Path(manifest_cache).resolve() != Path(cache_dir).resolve():
        msg = (
            f"Manifest was exported against cache directory {manifest_cache!r}, but this run resolves "
            f"{str(cache_dir)!r}. Re-run from the right project, or pass --force to override."
        )
        raise ManifestError(msg)

    approved: list[ApprovedImage] = []
    for entry in payload.get("approved", []):
        missing = [key for key in _REQUIRED_KEYS if key not in entry]
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

        approved.append(
            ApprovedImage(
                test_name=str(entry["test_name"]),
                image_name=str(entry["image_name"]),
                call_index=int(entry["call_index"]),
                status=str(entry["status"]),
                source=source,
                destination=destination,
            )
        )

    return approved
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_summary_approve.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add pytest_pyvista/summary/approve.py tests/test_summary_approve.py
git commit -m "feat: validate exported approval manifests"
```

---

### Task 11: The `pytest-pyvista-approve` console script

**Files:**
- Modify: `pytest_pyvista/summary/approve.py` (add `apply_approvals` and `main`)
- Modify: `pyproject.toml` (add `[project.scripts]`)
- Test: `tests/test_summary_approve_cli.py`

**Interfaces:**
- Consumes: `load_manifest`, `ApprovedImage`, `ManifestError` (Task 10)
- Produces: `apply_approvals(approved: list[ApprovedImage], *, dry_run: bool = False) -> list[tuple[Path, Path]]`, `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_approve_cli.py`:

```python
"""Tests for the pytest-pyvista-approve console script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pytest_pyvista.summary.approve import main


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
            }
        ],
    }
    (tmp_path / "approvals.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_default_target_is_staging(project: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["approvals.json"]) == 0

    assert (project / "approved_images" / "sphere.png").read_bytes() == b"generated-bytes"
    assert not (project / "image_cache_dir" / "sphere.png").exists()
    assert "approved_images" in capsys.readouterr().out


def test_cache_target_writes_into_the_cache(project: Path) -> None:
    assert main(["approvals.json", "--target", "cache"]) == 0

    assert (project / "image_cache_dir" / "sphere.png").read_bytes() == b"generated-bytes"


def test_dry_run_copies_nothing(project: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["approvals.json", "--target", "cache", "--dry-run"]) == 0

    assert not (project / "image_cache_dir" / "sphere.png").exists()
    assert "dry run" in capsys.readouterr().out.lower()


def test_custom_staging_dir_is_honoured(project: Path) -> None:
    assert main(["approvals.json", "--staging_dir", "review"]) == 0

    assert (project / "review" / "sphere.png").is_file()


def test_invalid_manifest_exits_non_zero(project: Path, capsys: pytest.CaptureFixture) -> None:
    (project / "approvals.json").write_text("{not json", encoding="utf-8")

    assert main(["approvals.json"]) == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_missing_manifest_exits_non_zero(project: Path) -> None:
    assert main(["nope.json"]) == 1


def test_empty_manifest_reports_nothing_to_do(project: Path, capsys: pytest.CaptureFixture) -> None:
    payload = json.loads((project / "approvals.json").read_text(encoding="utf-8"))
    payload["approved"] = []
    (project / "approvals.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["approvals.json"]) == 0
    assert "no approved images" in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_summary_approve_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'main'`

- [ ] **Step 3: Write minimal implementation**

Append to `pytest_pyvista/summary/approve.py`:

```python
def apply_approvals(approved: list[ApprovedImage], *, dry_run: bool = False) -> list[tuple[Path, Path]]:
    """Copy each approved image to its destination, returning the (source, destination) pairs."""
    import shutil  # noqa: PLC0415

    copies: list[tuple[Path, Path]] = []
    for image in approved:
        if not dry_run:
            image.destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(image.source, image.destination)
        copies.append((image.source, image.destination))
    return copies


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``pytest-pyvista-approve`` console script."""
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="pytest-pyvista-approve",
        description="Apply images approved in a pytest-pyvista image summary report.",
    )
    parser.add_argument("manifest", help="Path to the approvals.json exported from the report.")
    parser.add_argument("--target", choices=["staging", "cache"], default="staging", help="Where to copy approved images.")
    parser.add_argument("--staging_dir", default="approved_images", help="Staging directory when --target=staging.")
    parser.add_argument("--generated_image_dir", default=None, help="Override the generated image directory the manifest must point into.")
    parser.add_argument("--force", action="store_true", help="Apply even if the manifest's cache directory does not match.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned copies without performing them.")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"error: {manifest_path} does not exist", file=sys.stderr)  # noqa: T201
        return 1

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        print(f"error: {manifest_path} is not valid JSON: {error}", file=sys.stderr)  # noqa: T201
        return 1

    cache_dir = Path(payload.get("cache_dir", ""))
    target_root = cache_dir if args.target == "cache" else Path(args.staging_dir).resolve()
    source_root = Path(args.generated_image_dir).resolve() if args.generated_image_dir else Path.cwd()

    try:
        approved = load_manifest(
            manifest_path,
            cache_dir=cache_dir,
            target_root=cache_dir,
            source_root=source_root,
            force=args.force,
        )
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)  # noqa: T201
        return 1

    if args.target == "staging":
        approved = [
            ApprovedImage(
                test_name=image.test_name,
                image_name=image.image_name,
                call_index=image.call_index,
                status=image.status,
                source=image.source,
                destination=target_root / image.destination.relative_to(cache_dir),
            )
            for image in approved
        ]

    if not approved:
        print("No approved images in manifest; nothing to do.")  # noqa: T201
        return 0

    copies = apply_approvals(approved, dry_run=args.dry_run)
    prefix = "would copy" if args.dry_run else "copied"
    for source, destination in copies:
        print(f"{prefix}: {source} -> {destination}")  # noqa: T201
    summary = f"{len(copies)} image(s) {'planned' if args.dry_run else 'applied'} to {target_root}"
    print(f"{summary}{' (dry run)' if args.dry_run else ''}")  # noqa: T201
    return 0
```

In `pyproject.toml`, add after the `[project.urls]` block:

```toml
[project.scripts]
pytest-pyvista-approve = "pytest_pyvista.summary.approve:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_summary_approve_cli.py -v`
Expected: PASS (7 passed)

Confirm the entry point installs: `pip install -e . && pytest-pyvista-approve --help`

- [ ] **Step 5: Commit**

```bash
git add pytest_pyvista/summary/approve.py pyproject.toml tests/test_summary_approve_cli.py
git commit -m "feat: add pytest-pyvista-approve console script"
```

---

### Task 12: Documentation

**Files:**
- Modify: `README.rst` (new section after "Documentation testing flags", plus entries in "Configuration")
- Test: manual — `python -m sphinx -b html doc doc/_build/html` must not warn on the new content

**Interfaces:**
- Consumes: every flag from Tasks 5 and 11
- Produces: no code interface

- [ ] **Step 1: Add the report section to `README.rst`**

Insert after the "Documentation testing flags" section and before "Customizing test cases":

```rst
Image summary report
--------------------
Use ``--summary_html`` to generate an interactive HTML report at the end of a test run.
The report is a catalogue of every image test — not just the failures — showing the
cached baseline, the generated render, and the pixel differences between them, along
with the error value and the threshold it was measured against.

.. code-block:: bash

    pytest --summary_html

This writes ``image_test_report/index.html`` relative to the `pytest root path`_. Open it
in a browser to filter by status, search by test name, and review each image.

Each image is given one of six statuses:

* ``passed`` — the render matches the baseline.
* ``warned`` — the error exceeded the warning threshold but not the error threshold. The
  test passed, but the image is drifting.
* ``failed`` — the error exceeded the error threshold.
* ``skipped`` — the comparison was skipped, so no render was generated.
* ``new`` — no baseline existed for this image.
* ``reset`` — a baseline existed and was overwritten during this run by
  ``--reset_image_cache`` or ``--reset_only_failed``. The report compares against the
  baseline as it was *before* the overwrite.

Approving images
================
Images that are not yet in the cache — ``new``, ``failed`` and ``warned`` — can be
approved individually in the report. Ticking is reversible, and **Accept all new**
approves every new image at once. Click **Export approvals** to download
``approvals.json``, then apply it:

.. code-block:: bash

    # copy approved images to a staging directory for a second look (default)
    pytest-pyvista-approve approvals.json

    # or write them straight into the image cache
    pytest-pyvista-approve approvals.json --target cache

Use ``--dry-run`` to print the planned copies without performing them.

Images that were already written to the cache during the run — by
``--add_missing_images``, ``--reset_image_cache`` or ``--reset_only_failed`` — are shown
for reference with the reason they were pre-approved, and have no checkbox. Unticking
them is not offered, because nothing would be removed from the cache.

Summary report flags
====================
* ``--summary_html`` enables report generation.

* ``--summary_html_dir <DIR>`` sets the report directory, relative to the
  `pytest root path`_. Defaults to ``image_test_report``. Setting it implies
  ``--summary_html``.

* ``--summary_html_include <STATUSES>`` restricts which statuses are written to the
  report, as a comma-separated list, e.g. ``failed,warned,new``. All statuses are
  included by default; the report's own filter controls narrow the view further without
  needing to re-run.

* ``--summary_html_max_image_size <N>`` limits the longest edge of the images shown
  inline on each card. Defaults to ``400``.

* ``--summary_html_full_size <MODE>`` controls which records also keep a
  full-resolution copy, linked from the card. One of ``none``, ``failing`` (the default,
  meaning any non-passing record) or ``all``.

* ``--summary_html_embed`` produces a single ``index.html`` with every image inlined as a
  data URI. Convenient for sharing a small report as one file, but impractical for large
  test suites.

.. note::
   The summary report cannot be used with ``--doc_mode``.

.. note::
   If ``--generated_image_dir`` is not configured, enabling the report writes generated
   images to a temporary directory so that they are available to the report.
```

Add to the "Configuration" section, after the failed-images example:

```rst
Configure the image summary report:

.. code-block:: toml

    [tool.pytest.ini_options]
    summary_html = true
    summary_html_dir = "reports/image_test_report"
    summary_html_include = "passed,warned,failed,skipped,new,reset"
    summary_html_max_image_size = 400
    summary_html_full_size = "failing"
```

Add the link target beside the other definitions at the bottom of the file:

```rst
.. _`pytest root path`: https://docs.pytest.org/en/latest/reference/reference.html#pytest.Config.rootpath
```

- [ ] **Step 2: Build the docs and check for warnings**

Run: `python -m sphinx -b html doc doc/_build/html -W`
Expected: build succeeds with no warnings

- [ ] **Step 3: Verify every documented flag exists**

Run: `pytest --help | grep summary_html`
Expected: all six flags listed

- [ ] **Step 4: Run the full suite and linters**

Run: `pytest tests/ -q && pre-commit run --all-files`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add README.rst
git commit -m "docs: document the image summary report and approval workflow"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: record capture and schema → Tasks 1, 6; diff and size mismatch → Tasks 2, 6; image store, downscaling and `full_size` → Task 3; six statuses and panel degradation → Tasks 4, 7; options and doc-mode exclusion → Task 5; xdist aggregation → Tasks 6, 9; UI, badges, metadata, approval chips → Task 7; filters, search, sort, reversible approvals, localStorage keying, export → Task 8; report generation and terminal line → Task 9; manifest contract and the five safety rules → Task 10; console script and staging/cache targets → Task 11; docs → Task 12.

**Known gap accepted deliberately:** the spec's "warn on stdout if embedding exceeds ~50 MB" is not implemented. It is a nicety on a mode already documented as unsuitable for large suites, and adding it would not change any behaviour worth a test. If it is wanted, it belongs as a two-line addition to `write_report` in Task 7.

**Type consistency.** `ImageStatus` and `CacheWriteReason` are defined once in Task 1 and imported everywhere. `ReportImageStore.save_file`/`save_image` return `(str, str | None)` in Task 3 and are consumed that way in Task 6. `determine_status`' keyword-only signature in Task 4 matches its call in Task 6. `read_records` (Task 1) and `write_report` (Task 7) match their calls in Task 9. `ApprovedImage`'s fields are identical across Tasks 10 and 11. The `data-*` DOM attributes emitted in Task 7 are exactly the ones Task 8's script reads.

**Riskiest task.** Task 6 is the only one that edits control flow in `VerifyImageCache.__call__`, including deferring a raise. Its integration coverage lives in Task 9, so run the full suite after both before assuming either is done.
