"""Tests for pytest_pyvista.summary.render."""

from __future__ import annotations

from importlib import resources
import json
import re
from typing import TYPE_CHECKING

from PIL import Image

from pytest_pyvista.summary.record import ALL_STATUSES
from pytest_pyvista.summary.record import ImageRecord
from pytest_pyvista.summary.render import render_report
from pytest_pyvista.summary.render import write_report

if TYPE_CHECKING:
    from pathlib import Path

_MANIFEST_OPEN_TAG = '<script id="manifest" type="application/json">'

# WCAG 2.1 AA demands 4.5:1 for text below 18.66px; the badges are 0.75rem.
_WCAG_AA_CONTRAST = 4.5

# sRGB channel value below which the transfer function is linear rather than a power curve.
_SRGB_LINEAR_LIMIT = 0.03928


def _record(**overrides: object) -> ImageRecord:
    kwargs: dict[str, object] = {
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


def _css() -> str:
    """Return the packaged report stylesheet as text."""
    return resources.files("pytest_pyvista.summary.assets").joinpath("report.css").read_text(encoding="utf-8")


def _palettes(css: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return the custom-property palettes the stylesheet declares for the light and dark themes."""
    declarations = r"--([a-z-]+): (#[0-9a-f]{6});"
    dark_at = css.index("@media (prefers-color-scheme: dark)")
    light = dict(re.findall(declarations, css[:dark_at]))
    dark = dict(light)
    dark.update(re.findall(declarations, css[dark_at:]))
    return light, dark


def _badge_foreground(css: str, palette: dict[str, str]) -> str:
    """Resolve the color the ``.badge`` rule paints its text with, under ``palette``."""
    rule = css[css.index(".badge {") :]
    declared = re.findall(r"color: ([^;]+);", rule[: rule.index("}")])[0]
    variable = re.fullmatch(r"var\(--([a-z-]+)\)", declared)
    return palette[variable.group(1)] if variable is not None else declared


def _relative_luminance(color: str) -> float:
    """Return the WCAG relative luminance of a ``#rrggbb`` color."""
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [channel / 12.92 if channel <= _SRGB_LINEAR_LIMIT else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    """Return the WCAG 2.1 contrast ratio between two ``#rrggbb`` colors."""
    lighter, darker = sorted((_relative_luminance(foreground), _relative_luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _manifest_payload(document: str) -> str:
    """Return the raw text the browser would parse out of the embedded manifest element."""
    start = document.index(_MANIFEST_OPEN_TAG) + len(_MANIFEST_OPEN_TAG)
    return document[start : document.index("</script>", start)]


def test_report_contains_the_test_name_and_status() -> None:
    """A card carries the test name and its status as a filterable attribute."""
    document = render_report([_record()], run_id="run-1", metadata=_metadata())

    assert "test_sphere" in document
    assert 'data-status="failed"' in document


def test_report_embeds_the_run_id_for_approval_state() -> None:
    """The run id is exposed on the root element for the client-side store."""
    document = render_report([_record()], run_id="run-1", metadata=_metadata())

    assert 'data-run-id="run-1"' in document


def test_report_escapes_test_names() -> None:
    """A test name containing markup is escaped in the card."""
    document = render_report([_record(test_name="test_<script>")], run_id="run-1", metadata=_metadata())

    assert "test_&lt;script&gt;" in document
    assert "test_<script>" not in document


def test_approvable_statuses_get_a_checkbox() -> None:
    """New, failed and warned records awaiting approval get a live checkbox."""
    for status in ("new", "failed", "warned"):
        document = render_report([_record(status=status)], run_id="run-1", metadata=_metadata())
        assert 'class="approve"' in document, status


def test_already_cached_records_get_a_chip_not_a_checkbox() -> None:
    """A record written to the cache this run gets a static reason chip instead."""
    document = render_report(
        [_record(status="new", cache_written=True, cache_write_reason="add_missing_images")],
        run_id="run-1",
        metadata=_metadata(),
    )

    assert 'class="approve"' not in document
    assert "add_missing_images" in document


def test_passed_and_skipped_records_get_no_checkbox() -> None:
    """Nothing is offered for approval on records that are not approvable."""
    for status in ("passed", "skipped"):
        document = render_report([_record(status=status)], run_id="run-1", metadata=_metadata())
        assert 'class="approve"' not in document, status


def test_size_mismatch_replaces_the_diff_panel() -> None:
    """A size mismatch degrades the difference panel to an explanatory placeholder."""
    document = render_report([_record(size_mismatch=True, diff_image=None, error=None)], run_id="run-1", metadata=_metadata())

    assert "size mismatch" in document.lower()


def test_size_mismatch_notice_states_both_dimensions() -> None:
    """The notice replacing the diff panel names the baseline and the generated dimensions."""
    document = render_report(
        [
            _record(
                size_mismatch=True,
                diff_image=None,
                error=None,
                baseline_width=800,
                baseline_height=600,
                generated_width=801,
                generated_height=600,
            ),
        ],
        run_id="run-1",
        metadata=_metadata(),
    )

    assert "baseline 800x600" in document
    assert "generated 801x600" in document


def test_size_mismatch_notice_degrades_when_the_dimensions_are_absent() -> None:
    """A record carrying no dimensions still gets a readable notice rather than a rendered ``None``."""
    document = render_report([_record(size_mismatch=True, diff_image=None, error=None)], run_id="run-1", metadata=_metadata())

    assert "size mismatch" in document.lower()
    assert "NonexNone" not in document


def test_status_badges_meet_wcag_aa_contrast_in_both_themes() -> None:
    """Every status badge clears 4.5:1 between its text and its background, light theme and dark."""
    css = _css()
    light, dark = _palettes(css)

    for theme, palette in (("light", light), ("dark", dark)):
        foreground = _badge_foreground(css, palette)
        for status in ALL_STATUSES:
            ratio = _contrast(foreground, palette[status])
            assert ratio >= _WCAG_AA_CONTRAST, f"{theme} {status} badge: {ratio:.2f}:1"


def test_header_reports_the_total_tally() -> None:
    """The header counts every record, not only the per-status tallies."""
    records = [_record(status="failed"), _record(status="passed"), _record(status="new")]

    document = render_report(records, run_id="run-1", metadata=_metadata())

    assert '<span class="tally" id="total">Total 3</span>' in document


def test_the_total_tally_is_not_a_status_filter() -> None:
    """The total is a count, so the client-side filter logic can never mistake it for a status."""
    document = render_report([_record()], run_id="run-1", metadata=_metadata())

    assert document.count('class="status-filter"') == len(ALL_STATUSES)
    assert 'value="total"' not in document


def test_a_non_numeric_error_renders_as_no_error_at_all() -> None:
    """A record whose error is not a number degrades instead of destroying the whole report."""
    document = render_report([_record(error="corrupt")], run_id="run-1", metadata=_metadata())

    assert 'data-error="0"' in document
    assert "corrupt" not in document
    assert "Image regression error" not in document


def test_non_finite_errors_render_a_sortable_number() -> None:
    """NaN and infinite errors leave ``data-error`` finite so the client-side sort stays defined."""
    for error in (float("nan"), float("inf"), float("-inf")):
        document = render_report([_record(error=error)], run_id="run-1", metadata=_metadata())

        assert 'data-error="0"' in document, error
        assert "Image regression error" not in document, error


def test_a_non_numeric_threshold_renders_as_an_unknown_threshold() -> None:
    """An unusable threshold degrades to ``n/a`` rather than raising while formatting it."""
    document = render_report([_record(error=812.4, error_threshold="corrupt")], run_id="run-1", metadata=_metadata())

    assert "812.4 / n/a" in document
    assert "corrupt" not in document


def test_images_are_lazy_loaded() -> None:
    """Every rendered image defers loading until it is scrolled into view."""
    document = render_report([_record()], run_id="run-1", metadata=_metadata())

    assert 'loading="lazy"' in document


def test_empty_report_renders_without_crashing() -> None:
    """A run with no image tests still produces a readable page."""
    document = render_report([], run_id="run-1", metadata=_metadata())

    assert "No image tests" in document


def test_write_report_creates_index_html(tmp_path: Path) -> None:
    """write_report creates the report directory and returns the index path."""
    path = write_report([_record()], tmp_path / "report", run_id="run-1", metadata=_metadata())

    assert path == tmp_path / "report" / "index.html"
    assert path.is_file()


def test_embed_mode_inlines_images_as_data_uris(tmp_path: Path) -> None:
    """Embed mode produces a self-contained page with no references to images/."""
    report_dir = tmp_path / "report"
    images = report_dir / "images"
    images.mkdir(parents=True)
    for role in ("baseline", "generated", "diff"):
        Image.new("RGB", (4, 4), (1, 2, 3)).save(images / f"sphere.{role}.png")

    write_report([_record()], report_dir, run_id="run-1", metadata=_metadata(), embed=True)
    document = (report_dir / "index.html").read_text(encoding="utf-8")

    assert "data:image/png;base64," in document
    assert 'src="images/' not in document


def test_embed_mode_falls_back_to_the_relative_path_when_an_image_is_missing(tmp_path: Path) -> None:
    """A missing image file degrades to a relative reference rather than aborting the report."""
    report_dir = tmp_path / "report"

    write_report([_record()], report_dir, run_id="run-1", metadata=_metadata(), embed=True)
    document = (report_dir / "index.html").read_text(encoding="utf-8")

    assert 'src="images/sphere.generated.png"' in document


def test_manifest_cannot_break_out_of_its_script_element() -> None:
    """A record field containing a closing script tag cannot inject markup into the page."""
    hostile = "test_</script><img src=x onerror=alert(1)>"
    document = render_report([_record(test_name=hostile)], run_id="run-1", metadata=_metadata())

    payload = _manifest_payload(document)

    assert "<" not in payload
    assert json.loads(payload)["records"][0]["test_name"] == hostile


def test_manifest_round_trips_records_awaiting_approval() -> None:
    """The embedded manifest stays valid JSON and describes both ends of every copy."""
    document = render_report(
        [_record(generated_source="generated/sphere.png", cache_destination="cache/sphere.png")],
        run_id="run-1",
        metadata=_metadata(),
    )

    manifest = json.loads(_manifest_payload(document))

    assert manifest["run_id"] == "run-1"
    assert manifest["records"] == [
        {
            "key": "test_sphere::0",
            "test_name": "test_sphere",
            "image_name": "sphere.png",
            "call_index": 0,
            "status": "failed",
            "source": "generated/sphere.png",
            "destination": "cache/sphere.png",
        },
    ]


def test_record_data_in_the_metadata_panel_is_escaped() -> None:
    """Skip reason, environment and matched baseline are escaped, not injected as markup."""
    document = render_report(
        [
            _record(
                status="skipped",
                skip_reason="<script>alert('skip')</script>",
                env_info="<script>alert('env')</script>",
                matched_baseline="<script>alert('match')</script>",
                candidate_baselines=["<script>alert('match')</script>", "other.png"],
            ),
        ],
        run_id="run-1",
        metadata=_metadata(),
    )

    assert "<script>alert(" not in document
    assert "&lt;script&gt;alert(&#x27;skip&#x27;)&lt;/script&gt;" in document
    assert "&lt;script&gt;alert(&#x27;env&#x27;)&lt;/script&gt;" in document
    assert "&lt;script&gt;alert(&#x27;match&#x27;)&lt;/script&gt;" in document


def test_matched_baseline_shows_its_position_among_the_candidates() -> None:
    """A multi-baseline test reports which candidate matched."""
    document = render_report(
        [_record(matched_baseline="b.png", candidate_baselines=["a.png", "b.png", "c.png"])],
        run_id="run-1",
        metadata=_metadata(),
    )

    assert "b.png (2 of 3)" in document


def test_matched_baseline_absent_from_the_candidates_still_renders() -> None:
    """A matched baseline outside the candidate list degrades instead of aborting the report."""
    document = render_report(
        [_record(matched_baseline="ghost.png", candidate_baselines=["a.png", "b.png"])],
        run_id="run-1",
        metadata=_metadata(),
    )

    assert "ghost.png" in document
    assert "of 2)" not in document
