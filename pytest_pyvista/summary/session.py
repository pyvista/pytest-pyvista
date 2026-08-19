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

    def __init__(  # noqa: PLR0913
        self,
        *,
        run_id: str,
        records_dir: Path,
        store: ReportImageStore,
        worker_id: str,
        statuses: tuple[str, ...],
        cache_dir: Path,
    ) -> None:
        """Initialize the SummarySession."""
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
        baseline_width: int | None = None
        baseline_height: int | None = None
        generated_width: int | None = None
        generated_height: int | None = None
        computed_error = error
        if baseline_source is not None and generated_source is not None:
            diff = compute_diff_image(baseline_source, generated_source)
            diff_image = diff.image
            size_mismatch = diff.size_mismatch
            baseline_width, baseline_height = diff.baseline_size
            generated_width, generated_height = diff.generated_size
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
            baseline_width=baseline_width,
            baseline_height=baseline_height,
            generated_width=generated_width,
            generated_height=generated_height,
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

    @staticmethod
    def _error(baseline: Path, generated: Path) -> float | None:
        import pyvista  # noqa: PLC0415

        try:
            return float(pyvista.compare_images(str(generated), str(baseline)))
        except (RuntimeError, ValueError):
            return None
