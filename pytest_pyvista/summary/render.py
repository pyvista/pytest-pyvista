"""Render summary report records as a self-contained HTML page."""

from __future__ import annotations

import base64
import contextlib
import html
from importlib import resources
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from pytest_pyvista.summary.record import ALL_STATUSES
from pytest_pyvista.summary.record import SCHEMA_VERSION

if TYPE_CHECKING:
    from pytest_pyvista.summary.record import ImageRecord

APPROVABLE_STATUSES = frozenset({"new", "failed", "warned"})

_ASSET_PACKAGE = "pytest_pyvista.summary.assets"

_PANELS = (("baseline", "Baseline"), ("generated", "Generated"), ("diff", "Difference"))

# Written as literal characters rather than as ``&mdash;``/``&check;`` entities so that
# every interpolated string in this module can be escaped without exception. See _info.
_EM_DASH = "\N{EM DASH}"
_CHECK_MARK = "\N{CHECK MARK}"

# ``json.dumps`` leaves these three characters literal, which lets any record field
# containing the text "</script>" terminate the embedded manifest element early and
# inject arbitrary markup into the page. Replacing them with their ``\uXXXX`` escapes
# keeps the payload valid JSON while making tag breakout impossible. Order is
# irrelevant: no replacement introduces a character that another rule matches.
_JSON_HTML_ESCAPES = (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"))

_PLACEHOLDER_NOTES = {
    "baseline": "No baseline in the cache",
    "generated": "No image generated",
    "diff": "No difference to show",
}
_SIZE_MISMATCH_NOTE = f"Size mismatch {_EM_DASH} a pixel difference is undefined"


def _asset(name: str) -> str:
    """Read one of the packaged report assets as text."""
    return resources.files(_ASSET_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def _embed_json(payload: dict[str, object]) -> str:
    """Serialize ``payload`` as JSON that is safe to place inside an HTML ``<script>`` element."""
    # ``ensure_ascii=True`` is the default and is load-bearing: it escapes U+2028 and
    # U+2029, which a JavaScript parser treats as line terminators, into their ASCII escape
    # form before the HTML escaping below runs. Never pass ``ensure_ascii=False`` here.
    text = json.dumps(payload)
    for character, escape in _JSON_HTML_ESCAPES:
        text = text.replace(character, escape)
    return text


def _data_uri(path: Path) -> str | None:
    """Return ``path`` as a base64 ``data:`` URI, or ``None`` when the file cannot be read."""
    with contextlib.suppress(OSError):
        # The result is confined to the base64 alphabet plus a fixed prefix, so it holds
        # no HTML-significant character and needs no escaping when interpolated.
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return None


def _finite(value: object) -> float | None:
    """
    Return ``value`` as a finite float, or ``None`` when it cannot be one.

    Record fields come straight off disk and ``read_records`` filters unknown keys without
    ever checking value types, so a hand-edited or corrupted line can carry a string, a
    ``NaN`` or an infinity where a number belongs. Formatting one with ``:g`` raises and
    would cost the whole report over a single bad record, and a non-finite sort key would
    leave the client-side ordering undefined, so both degrade to "no value" instead.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:  # A JSON integer too large to be a float.
        return None
    return number if math.isfinite(number) else None


def _dimensions(width: object, height: object) -> str | None:
    """Format one image's size as ``WxH``, or ``None`` when the record did not capture it."""
    return f"{width}x{height}" if isinstance(width, int) and isinstance(height, int) else None


def _size_mismatch_note(record: ImageRecord) -> str:
    """Explain why there is no difference image, naming both sizes when the record has them."""
    baseline = _dimensions(record.baseline_width, record.baseline_height)
    generated = _dimensions(record.generated_width, record.generated_height)
    if baseline is None or generated is None:
        # A record written before the sizes were captured; say what is known, not "NonexNone".
        return _SIZE_MISMATCH_NOTE
    return f"Size mismatch {_EM_DASH} baseline {baseline} vs generated {generated}, so a pixel difference is undefined"


def _panel(record: ImageRecord, role: str, label: str, embed_dir: Path | None) -> str:
    """Render one comparison panel, degrading to a placeholder when there is no image."""
    source: str | None = getattr(record, f"{role}_image")
    full: str | None = getattr(record, f"{role}_image_full")

    if source is None:
        note = _size_mismatch_note(record) if role == "diff" and record.size_mismatch else _PLACEHOLDER_NOTES[role]
        return f'<div class="panel"><h3>{label}</h3><p class="placeholder">{html.escape(note)}</p></div>'

    src = html.escape(source)
    href = html.escape(full or source)
    if embed_dir is not None:
        # A missing file must not abort the whole report; fall back to the relative path.
        embedded = _data_uri(embed_dir / source)
        if embedded is not None:
            src = href = embedded

    alt = html.escape(f"{label} image for {record.test_name}")
    link = f'<a href="{href}" target="_blank" rel="noreferrer"><img loading="lazy" src="{src}" alt="{alt}"></a>'
    return f'<div class="panel"><h3>{label}</h3>{link}</div>'


def _matched_baseline(record: ImageRecord) -> str:
    """Describe the baseline that matched, with its position among the candidates."""
    matched = record.matched_baseline or ""
    candidates = record.candidate_baselines
    if matched not in candidates:
        # ``list.index`` would raise here, and a rendering failure at the end of a run
        # would cost the whole report over one odd record. Label it plainly instead.
        return matched
    return f"{matched} ({candidates.index(matched) + 1} of {len(candidates)})"


def _info(record: ImageRecord) -> str:
    """
    Render the metadata list for one card.

    Rows are plain text, never markup, and every one of them is escaped where the cells
    are built. That is deliberate: most values are record data straight off disk, so a
    row added later has no unescaped path to fall into.
    """
    rows: list[tuple[str, str]] = []

    error = _finite(record.error)
    if record.size_mismatch:
        rows.append(("Error", "not comparable (size mismatch)"))
    elif error is not None:
        error_threshold = _finite(record.error_threshold)
        threshold = f"{error_threshold:g}" if error_threshold is not None else "n/a"
        variance = " (high variance)" if record.high_variance_test else ""
        rows.append(("Image regression error", f"{error:g} / {threshold}{variance}"))

    rows.append(("Status", record.status))
    if record.skip_reason:
        rows.append(("Skipped by", record.skip_reason))
    if record.cache_write_reason:
        rows.append(("Approval reason", f"Already in cache {_EM_DASH} written by --{record.cache_write_reason}"))
    if record.matched_baseline and len(record.candidate_baselines) > 1:
        rows.append(("Matched baseline", _matched_baseline(record)))
        others = [candidate for candidate in record.candidate_baselines if candidate != record.matched_baseline]
        if others:
            rows.append(("Other baselines", ", ".join(others)))
    if record.env_info:
        rows.append(("Environment", record.env_info))

    cells = "".join(f"<dt>{html.escape(key)}</dt><dd>{html.escape(value)}</dd>" for key, value in rows)
    return f'<dl class="info">{cells}</dl>'


def _wants_approval(record: ImageRecord) -> bool:
    """
    Return True when this record is one the reader is being asked to approve.

    Exactly the condition for a live checkbox: an image already written to the cache by a
    policy flag is shown for reference, not for a decision. Shared with the card's sort flag
    so that "offered for approval" and "floated to the top of the sort" cannot drift apart.
    """
    return not record.cache_written and record.status in APPROVABLE_STATUSES


def _approval(record: ImageRecord) -> str:
    """Render the approval control: a live checkbox, a static chip, or nothing at all."""
    if record.cache_written:
        reason = html.escape(record.cache_write_reason or "policy")
        return f'<span class="chip">{_CHECK_MARK} In cache {_EM_DASH} --{reason}</span>'
    if _wants_approval(record):
        return '<label class="approve"><input type="checkbox"> Approve this image</label>'
    return ""


def _card(record: ImageRecord, embed_dir: Path | None) -> str:
    """Render one record as a filterable, sortable card."""
    status = html.escape(record.status)
    key = html.escape(f"{record.test_name}::{record.call_index}")
    name = html.escape(record.test_name if not record.call_index else f"{record.test_name} [{record.call_index}]")
    panels = "".join(_panel(record, role, label, embed_dir) for role, label in _PANELS)
    # Always a finite number: the client-side sort comparator is undefined otherwise.
    value = _finite(record.error)
    error = f"{value:g}" if value is not None else "0"
    # A record with no comparable error is not a record with an error of zero. `new` cards are
    # the whole point of a first run and would otherwise sort beneath every passing test, so the
    # flag below lets the default error-descending sort float them to the top instead. Only
    # cards awaiting a decision float: a skipped comparison also has no error, and is the least
    # actionable card in the report - floating it would bury the failures it sorted above.
    missing_error = "1" if value is None and _wants_approval(record) else "0"
    return (
        f'<article class="card" data-status="{status}" data-key="{key}" '
        f'data-name="{html.escape(record.test_name.lower())}" data-error="{error}" '
        f'data-missing-error="{missing_error}">'
        f'<header><span class="badge {status}">{status}</span>'
        f'<span class="name">{name}</span>'
        f'<span class="env">{html.escape(record.env_info)}</span>'
        f'<span style="margin-left:auto">{_approval(record)}</span></header>'
        f'<div class="panels">{panels}</div>{_info(record)}</article>'
    )


def _manifest(records: list[ImageRecord], run_id: str) -> str:
    """Build the embedded approval manifest describing every record awaiting approval."""
    payload: dict[str, object] = {
        # Carried into the page so that the client-side export can echo it instead of
        # hardcoding a second copy of the number the approve CLI validates against.
        "schema_version": SCHEMA_VERSION,
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
    return _embed_json(payload)


def render_report(records: list[ImageRecord], *, run_id: str, metadata: dict[str, str], embed_dir: Path | None = None) -> str:
    """
    Render the complete report page as a single HTML string.

    ``embed_dir`` turns on embed mode: images are inlined as ``data:`` URIs read
    relative to that directory, producing a page with no external references.
    """
    counts = {status: sum(1 for record in records if record.status == status) for status in ALL_STATUSES}

    # A tally, deliberately not a filter: it carries no ``status-filter`` control, so the
    # client-side filter logic cannot mistake "total" for a seventh status.
    total = f'<span class="tally" id="total">Total {len(records)}</span>'
    filters = "".join(
        f'<label><input type="checkbox" class="status-filter" value="{status}" checked> '
        f'<span class="badge {status}">{status}</span> {counts[status]}</label>'
        for status in ALL_STATUSES
    )
    meta_rows = "".join(f"<dt>{html.escape(key)}</dt><dd>{html.escape(value)}</dd>" for key, value in metadata.items())
    cards = "".join(_card(record, embed_dir) for record in records) or '<p class="empty">No image tests were recorded in this run.</p>'

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
<p class="legend"><span class="swatch"></span> Difference panels paint changed pixels in magenta over a faded copy of the baseline.</p>
<div id="notice" class="notice" hidden></div>
<div class="controls">
  {total}
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
<script id="manifest" type="application/json">{_manifest(records, run_id)}</script>
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
    document = render_report(records, run_id=run_id, metadata=metadata, embed_dir=report_dir if embed else None)
    path = report_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path
