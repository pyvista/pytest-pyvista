"""Turning a single image comparison into a report record."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_pyvista.summary.record import CacheWriteReason
    from pytest_pyvista.summary.record import ImageStatus


def determine_status(  # noqa: PLR0911,PLR0913
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
