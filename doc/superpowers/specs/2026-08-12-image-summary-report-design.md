# Image Summary Report Design

**Date:** 2026-08-12
**Feature:** Interactive HTML summary report for PyVista image test results
**Status:** Design Document (revised)

---

## Overview

An opt-in HTML report, generated at the end of a normal `pytest` run, that presents every
image test as a card: baseline, generated render, and pixel difference side by side, with
the metadata needed to judge whether a difference matters. The report is a *catalogue* of
the image test suite, not a failure log — passing, warning, skipped and new images are all
included, and an interactive filter narrows the view.

For images that are not yet in the cache, the reader can selectively approve them and
export those selections, then apply them to the cache (or to a staging directory) in a
separate, explicit step.

### Goals

1. **Close the feedback loop** — see what changed without hunting through directories.
2. **Fine-grained control** — approve individual images rather than applying a blanket policy.
3. **Audit trail** — every image states why it is in the state it is in.
4. **Fits the existing plugin** — reuses the configured cache/generated/failed directories,
   the existing option-resolution rules, and the existing xdist aggregation pattern.
5. **Safe by default** — nothing touches the cache without a separate, deliberate command.

---

## Design Approach

Interactive HTML with browser-local approval state:

- Report is written as a **directory** (`index.html` plus an `images/` subdirectory).
- Approval state lives in the browser's `localStorage`, keyed by run id.
- An **Export Approvals** button downloads a JSON manifest.
- A separate console script applies that manifest to the cache or a staging directory.

**Why a directory rather than a single embedded file.** PyVista's own suite runs to
roughly a thousand image tests. Three 1024×768 PNGs per card, base64-inflated by ~33%,
puts a fully embedded report into the hundreds of megabytes — past the point any browser
opens it comfortably. Writing images as files, downscaling them for display, and
lazy-loading them keeps the report usable at suite scale. The directory is still
self-contained and trivially zipped or served. Single-file embedding remains available
via `--summary_html_embed` for small suites and for pasting into an issue.

---

## Architecture

The central constraint: **the report's data must be captured while each image comparison
happens, not reconstructed from disk afterwards.** Three facts force this.

1. **Nothing is persisted by default.** `_save_generated_image` runs only when
   `generated_image_dir` is set ([`pytest_pyvista.py:523`](../../../pytest_pyvista/pytest_pyvista.py#L523));
   `_save_failed_test_images` only when `failed_image_dir` is set
   ([`pytest_pyvista.py:557`](../../../pytest_pyvista/pytest_pyvista.py#L557)). Both default
   to `None`, and the `Plotter` is gone by teardown. A session-end scan of an unconfigured
   run finds nothing.

2. **Cache-writing policies destroy the baseline before or after comparison.**
   `--add_missing_images` and `--reset_image_cache` screenshot into the cache path *before*
   comparing ([`pytest_pyvista.py:520-521`](../../../pytest_pyvista/pytest_pyvista.py#L520-L521)),
   so the plugin's own comparison is image-against-itself and reports zero error.
   `--reset_only_failed` overwrites the baseline *after* a real comparison
   ([`pytest_pyvista.py:559-564`](../../../pytest_pyvista/pytest_pyvista.py#L559-L564)).
   In both cases the prior baseline no longer exists at session end — which is precisely
   the review a reader wants after a mass reset.

3. **Status is not recoverable from filenames.** The error value, the applicable threshold,
   which of several candidate baselines matched, and whether the cache was written are all
   known only inside `VerifyImageCache.__call__`.

### Pipeline

```
pytest run (--summary_html)
  │
  ├─ pytest_configure
  │    • create records temp dir (master only), mirroring
  │      _make_config_cache_dir(config, PYVISTA_SUMMARY_RECORDS_DIRNAME, clean=True)
  │    • if generated_image_dir is unset, point it at a session temp dir so
  │      renders are always persisted
  │    • allocate run_id (uuid4), share with xdist workers
  │
  ├─ per image, inside VerifyImageCache.__call__
  │    • copy the prior baseline (if any) into the report image store
  │      BEFORE any cache write
  │    • copy the generated render into the report image store
  │    • compute the report's own error + diff image against the prior baseline
  │    • append one ImageRecord as a line of JSONL to the worker's records file
  │
  ├─ pytest_sessionfinish (each worker)
  │    • flush/close the worker's JSONL file
  │
  └─ pytest_terminal_summary (master only, guarded by hasattr(config, "workerinput"))
       • combine all worker JSONL files into one ordered record list
       • render index.html
       • print the report path to the terminal
```

The temp-dir-plus-per-worker-JSON approach is the one the plugin already uses for
`VISITED_CACHED_IMAGE_NAMES` / `SKIPPED_CACHED_IMAGE_NAMES`
([`pytest_pyvista.py:1125-1148`](../../../pytest_pyvista/pytest_pyvista.py#L1125-L1148)).
`_combine_temp_jsons` combines into a `set[str]`, so records need a sibling helper that
combines JSONL into `list[dict]`. JSONL (append-only, one record per line) is preferred over
one JSON blob per worker so that a crashed worker still yields the records it completed.

### `ImageRecord` schema

One record per generated image (note a single test may produce several — `n_calls`
disambiguates via the `_0`, `_1` suffix convention at
[`pytest_pyvista.py:473`](../../../pytest_pyvista/pytest_pyvista.py#L473)).

```json
{
  "schema_version": 1,
  "run_id": "8f14e45f-…",
  "test_name": "test_sphere",
  "image_name": "sphere.png",
  "call_index": 0,
  "status": "failed",

  "error": 812.4,
  "error_threshold": 500.0,
  "warning_threshold": 200.0,
  "high_variance_test": false,
  "size_mismatch": false,

  "baseline_image": "images/sphere.baseline.png",
  "generated_image": "images/sphere.generated.png",
  "diff_image": "images/sphere.diff.png",

  "matched_baseline": "cache/sphere/1.png",
  "candidate_baselines": ["cache/sphere/0.png", "cache/sphere/1.png"],

  "cache_written": false,
  "cache_write_reason": null,
  "skip_reason": null,

  "cache_dir": "tests/image_cache",
  "image_format": "png",
  "env_info": "ubuntu-22.04_x86_64_gpu-NVIDIA_py-3.12.1_pyvista-0.44_vtk-9.3_no-CI"
}
```

`cache_write_reason` is one of `add_missing_images`, `reset_image_cache`,
`reset_only_failed`, or `null`. It is the source of the **Approval Reason** shown in the UI.

### Report directory layout

```
image_test_report/
├── index.html          # self-contained markup + CSS + JS, references images/ relatively
└── images/
    ├── sphere.baseline.png
    ├── sphere.generated.png
    ├── sphere.diff.png
    └── …
```

Images are written downscaled to `summary_html_max_image_size` (default 400px on the
longest edge, reusing the existing `_get_thumbnail_size` helper). Every `<img>` carries
`loading="lazy"`.

Full-resolution copies are retained according to `summary_html_full_size`, which takes
`none`, `failing` (default — any non-`passed` record) or `all`. Cards with a retained
full-resolution copy link to it; cards without show the downscaled image at full width
instead. The default keeps a report of a large, mostly-green suite small while preserving
detail exactly where someone is likely to zoom in, and `all` is available for reviewers
scrutinising passing renders.

With `--summary_html_embed`, the same markup is produced with images inlined as data URIs
and no `images/` directory. Intended for small suites; the report warns on stdout if it
embeds more than ~50 MB.

### Difference image

Computed by the report, not by the plugin, against the preserved prior baseline:

- Per-pixel absolute difference, thresholded, composited as a semi-transparent red overlay
  on a desaturated copy of the baseline, so unchanged structure stays legible.
- The numeric error shown on the card is `pyvista.compare_images`' metric — referred to
  throughout as the **image regression error**, not a pixel sum — evaluated against the
  threshold actually in force for that test (`var_error_value` / `var_warning_value` when
  `high_variance_test` is set, per
  [`pytest_pyvista.py:476-481`](../../../pytest_pyvista/pytest_pyvista.py#L476-L481)).
- **Size mismatch**: differing `window_size` across platforms, or `--max_image_size`, can
  make the two images different shapes, where a pixel diff is undefined. The record sets
  `size_mismatch: true`, the diff panel is replaced by a notice stating both dimensions,
  and no error value is claimed.

---

## Image Status Definitions

| Status | Meaning | Badge |
|---|---|---|
| **Passed** | Error at or below the warning threshold. | 🟢 green |
| **Warned** | Error above the warning threshold but at or below the error threshold — passing, but drifting. Also covers `errors_as_warnings`: a multi-baseline test that failed its primary baseline but matched an alternate. | 🟠 amber |
| **Failed** | Error above the error threshold. | 🔴 red |
| **Skipped** | Comparison skipped via `skip`, `windows_skip_image_cache`, `macos_skip_image_cache`, or `--ignore_image_cache`. | ⚪ grey |
| **New** | No baseline existed for this image. | 🔵 blue |
| **Reset** | A baseline existed and was overwritten this run by a cache-writing policy. | 🟣 purple |

Two notes on this taxonomy, both departures from the first draft:

**The warning tier gets its own status, and it is load-bearing.** The plugin has a middle tier —
`DEFAULT_WARNING_THRESHOLD = 200.0` against `DEFAULT_ERROR_THRESHOLD = 500.0` — that emits a
warning and saves images to `failed_image_dir/warnings/`
([`pytest_pyvista.py:569-573`](../../../pytest_pyvista/pytest_pyvista.py#L569-L573)). These
are the highest-value cards in the report: green in CI, but moving. Collapsing them into
Passed would hide exactly what a reviewer is looking for.

**Reset is distinguished from New.** Both are "the cache was written this run", but they
warrant different reading: New has nothing to compare against, while Reset has a prior
baseline that the report preserved and *can* diff against. Without the distinction, a
`--reset_image_cache` run renders as a wall of identical-looking green cards with empty diff
panels.

### Which panels each status has

Not every card can show three panels, and the layout must degrade rather than fail:

| Status | Baseline | Generated | Diff |
|---|---|---|---|
| Passed / Warned / Failed | ✓ | ✓ | ✓ |
| Reset | ✓ (preserved prior) | ✓ | ✓ |
| New | — (placeholder) | ✓ | — (placeholder) |
| Skipped | ✓ if one exists | — | — |

Skipped tests return before any screenshot is taken
([`pytest_pyvista.py:487-494`](../../../pytest_pyvista/pytest_pyvista.py#L487-L494)), so no
generated image exists for them at all; the card shows the baseline and the skip reason.

### Multiple baselines

A test may have several valid baselines in a subdirectory, and the comparison walks them
until one matches ([`pytest_pyvista.py:542-554`](../../../pytest_pyvista/pytest_pyvista.py#L542-L554)).
The card's baseline panel shows `matched_baseline`, labelled `matched 2 of 3`, with the
other candidates listed in the metadata panel.

---

## Configuration

Flags follow the plugin's existing snake_case convention, and resolve through
`_get_option_from_config_or_ini` so CLI beats `doc_`-prefixed ini beats plain ini.

### CLI flags

| Flag | Effect |
|---|---|
| `--summary_html` | Opt in to report generation. |
| `--summary_html_dir <DIR>` | Report output directory, relative to pytest rootpath. Default `image_test_report`. Setting it implies `--summary_html`. |
| `--summary_html_include <list>` | Comma-separated statuses to include. Default: all. Filters what is *written*; the in-report filter narrows further at read time. |
| `--summary_html_max_image_size <N>` | Longest-edge pixel limit for the images shown inline on each card. Default 400. |
| `--summary_html_full_size <MODE>` | Which records also retain a full-resolution copy: `none`, `failing` (default), or `all`. |
| `--summary_html_embed` | Produce a single self-contained `index.html` with images as data URIs. |

### ini options

```toml
[tool.pytest.ini_options]
summary_html = true
summary_html_dir = "reports/image_test_report"
summary_html_include = "passed,warned,failed,skipped,new,reset"
summary_html_max_image_size = 400
summary_html_full_size = "failing"
```

### Interaction with existing options

- `generated_image_dir` — if unset while `--summary_html` is active, the report points it at
  a session temp directory so renders exist to copy. If the user has set it, it is used
  as-is and left alone.
- `failed_image_dir` — unaffected; the report reads its own preserved copies, not this
  directory. Setting it remains useful independently.
- `image_cache_dir`, `image_format`, `max_image_size`, `generate_subdirs` — respected as
  configured; `generate_subdirs` affects the generated image path
  ([`pytest_pyvista.py:642-647`](../../../pytest_pyvista/pytest_pyvista.py#L642-L647)) and is
  recorded so the applier can resolve it.

### Scope: `--doc_mode` is out of scope for v1

Documentation mode is a parallel implementation with its own directories, its own
`_DocVerifyImageCache`, and failure modes this taxonomy has no badge for ("in cache but
missing from build", and the reverse). Passing `--summary_html` with `--doc_mode` raises
`pytest.UsageError` via the existing option-validation path in `pytest_configure`
([`pytest_pyvista.py:942-954`](../../../pytest_pyvista/pytest_pyvista.py#L942-L954)).
Extending the report to doc mode is a follow-up, and would need two additional statuses.

---

## UI/UX Design

### Header

- Title: **PyVista Image Test Report**
- Status tallies as clickable chips: Total | Passed | Warned | Failed | Skipped | New | Reset
- Run metadata: timestamp, run id, pytest / pytest-pyvista / Python / PyVista / VTK versions,
  cache directory, and the cache-affecting flags this run used

### Filter controls

- A checkbox per status, all checked by default; the header chips toggle the same state
- Free-text search over test name
- Sort: by error descending (default), or by test name
- **Accept all new** — shown only when unapproved New records exist. The bulk action stays
  deliberately limited to New: those have no baseline to lose, whereas bulk-accepting Failed
  or Warned images would overwrite baselines wholesale, which is the blanket behaviour this
  feature exists to replace.

### Test card

**Header** — status badge, test name (with `_0`/`_1` suffix when a test produced several
images), and `env_info`.

**Comparison panel** — three columns: `Baseline` | `Generated` | `Difference`, degrading per
the table above. Clicking a panel opens its full-resolution image where one was retained
(see `summary_html_full_size`), and the downscaled image at full width otherwise.

**Metadata panel**
- Image regression error, and the threshold in force (noting `high_variance_test` when set)
- Status, and skip reason for Skipped
- **Approval Reason** — e.g. *Already in cache — written by `--add_missing_images`* — or
  blank when the image awaits approval
- Matched baseline and other candidates, when there are several
- Environment details

**Approval control**
- Records awaiting approval — **New, Failed and Warned**, where the cache was not written
  this run — get a live checkbox: *Approve this image*. Toggling is free and reversible.
- Warned records are approvable even though they pass. A drifting image that stays green is
  the case most likely to go unexamined for months, and accepting the drift into the
  baseline deliberately is the point of reviewing it. Nothing forces the update; the
  checkbox simply exists.
- Records already written to the cache this run get **no checkbox**. They get a static
  `✓ In cache` chip carrying the reason. A pre-checked, greyed, inert checkbox sitting in
  the same column as live ones invites exactly the misreading it was meant to prevent; a
  chip reads as a statement of fact, which is what it is.

### Styling

- Semantic badge colours, each paired with a distinct shape/label so status does not rest on
  colour alone
- Responsive; comparison panels scroll horizontally on narrow viewports rather than
  overflowing the page
- Light and dark theme, following `prefers-color-scheme`
- Diff overlay uses a colour distinguishable under common colour-vision deficiencies
  (magenta rather than pure red), stated in a legend

### Footer

- **Export approvals** — downloads `approvals.json`
- Live count: *N images selected for approval*

---

## Approval Workflow

### In the browser

1. Open `image_test_report/index.html`.
2. Filter and review.
3. Tick *Approve this image* on New, Failed or Warned cards; untick freely.
4. Optionally **Accept all new**.
5. **Export approvals** → `approvals.json` downloads.

State persists in `localStorage` under `pytest-pyvista:approvals:<run_id>`. Because the run
id changes each run, a regenerated report starts clean rather than resurrecting selections
that referred to different pixels. On load, entries whose `run_id` does not match are
discarded and a dismissible notice says so. Keys older than 30 days are pruned.

### Applying approvals

A console script, not a pytest invocation:

```bash
# copy approved images into a staging directory for a second look (default)
pytest-pyvista-approve approvals.json

# or straight into the image cache
pytest-pyvista-approve approvals.json --target cache
```

`--target` accepts `staging` (default) or `cache`; `--staging_dir` overrides the staging
location. `--dry-run` prints the planned copies without performing them.

Routing this through `pytest` would mean collecting the entire suite and starting fixtures
in order to copy files, and would blur the line between "run tests" and "mutate baselines".
A console script is unit-testable without a pytest session and cannot be mistaken for a test
run. It is registered under `[project.scripts]`.

### Manifest format

The manifest must carry enough to locate both ends of each copy — `generate_subdirs` puts
generated images at `<dir>/<test>/<env_info>.png`, and multi-baseline destinations are
subdirectories, so a test name alone is insufficient.

```json
{
  "schema_version": 1,
  "run_id": "8f14e45f-…",
  "exported_at": "2026-08-12T14:05:00Z",
  "cache_dir": "tests/image_cache",
  "image_format": "png",
  "approved": [
    {
      "test_name": "test_sphere",
      "image_name": "sphere.png",
      "call_index": 0,
      "status": "failed",
      "source": "generated_images/sphere.png",
      "destination": "tests/image_cache/sphere.png"
    }
  ]
}
```

### Applier safety rules

The manifest round-trips through the user's Downloads directory, so it is treated as
untrusted input:

1. Reject unknown `schema_version`.
2. Reject if `cache_dir` does not match the resolved cache directory, unless `--force`.
3. Resolve every `source` and `destination` and reject any that escapes the run's generated
   directory or the target directory respectively — no traversal, no absolute paths outside
   those roots, no symlinks followed out.
4. Reject if a `source` file is missing rather than silently skipping it.
5. Print a summary of every copy performed, and a non-zero exit if any entry was rejected.

---

## Testing Strategy

**Unit**
- `ImageRecord` construction for each status, including `high_variance_test` thresholds
- JSONL combine helper: multiple workers, partial/truncated final line
- Diff image generation: identical images, differing images, mismatched sizes
- Manifest validation: each rejection rule above, including traversal attempts
- Applier copy resolution with and without `generate_subdirs`, single and multi-baseline

**Integration** (pytest's `pytester` fixture, as the existing suite uses)
- Bare `--summary_html` with no other directories configured produces a populated report —
  the case that fails under a session-end-scan design
- A run with each of `--add_missing_images`, `--reset_image_cache`, `--reset_only_failed`
  yields correct statuses and a diff against the *preserved prior* baseline
- `-n 2` under xdist produces exactly one report containing every worker's records
- `--summary_html` with `--doc_mode` raises `UsageError`
- `--summary_html_include` restricts what is written
- `--summary_html_full_size` at `none` / `failing` / `all` writes the expected set of
  full-resolution files and nothing more
- Report renders with zero image tests without crashing

**Manual**
- Open in a browser: filters, search, approve/unapprove, accept-all-new, export
- Confirm a several-hundred-test report opens and scrolls acceptably

---

## Out of Scope

- `--doc_mode` support (see above)
- Server-backed approval with direct commit from the browser
- Report-to-report history or trend tracking
- Automated approval flows in CI

---

## Acceptance Criteria

- [ ] `--summary_html` produces a report at the end of a normal run, whatever the exit status
- [ ] A bare `pytest --summary_html`, with no other directories configured, produces a
      populated report
- [ ] Records are captured at comparison time; prior baselines are preserved before any
      cache write
- [ ] All six statuses render with correct badges and correctly degraded panels
- [ ] Diff image is computed against the preserved prior baseline, with size mismatch handled
- [ ] Error values are shown against the threshold actually in force
- [ ] Multi-baseline tests show which candidate matched
- [ ] Exactly one report is produced under `pytest-xdist`, containing all workers' records
- [ ] Interactive filters, search and sort work against the rendered cards
- [ ] Approval checkboxes appear on New, Failed and Warned records awaiting approval;
      already-cached records show a static reason chip instead
- [ ] `--summary_html_max_image_size` controls inline resolution, and
      `--summary_html_full_size` controls full-resolution retention across `none` /
      `failing` / `all`
- [ ] Approval state is reversible and keyed by run id, discarding stale state
- [ ] Export produces a manifest that locates both ends of every copy
- [ ] `pytest-pyvista-approve` applies a manifest to staging or cache, enforcing every
      safety rule, with `--dry-run`
- [ ] `--summary_html` with `--doc_mode` raises `UsageError`
- [ ] A several-hundred-test report opens and scrolls acceptably in a browser
