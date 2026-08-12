# Image Summary Report Design

**Date:** 2026-08-12  
**Feature:** Interactive HTML summary report for PyVista image test results  
**Status:** Design Document

---

## Overview

This document describes the design for an interactive HTML summary report feature for pytest-pyvista. The report provides developers with a visual catalogue of all image test results—passed, failed, skipped, and new tests—with side-by-side comparisons, pixel-difference visualizations, and selective approval/export capabilities for cache updates.

### Goals

1. **Reduce feedback loop**: Developers see all image test results at a glance with visual comparisons
2. **Fine-grained control**: Selective approval of images for cache updates (not blanket policies)
3. **Audit trail**: Clear visibility into what was approved and why (approval reasons)
4. **Flexibility**: Works with existing pytest-pyvista flags and configuration
5. **Safety**: Reversible approvals with option to export to staging directory before committing

---

## Design Approach

**Option Selected: Interactive HTML with Browser-Local State (Option B)**

- Single, self-contained HTML file with embedded CSS and JavaScript
- Images stored as data URIs (no external dependencies)
- Approval state stored in browser's localStorage
- "Export Approvals" button downloads a JSON file
- Separate CLI step applies approvals to cache or staging directory
- Reversible: users can approve/unapprove before exporting

**Rationale:** Balances interactivity with simplicity; no server required; explicit two-step approval flow (export → commit) is safer for cache modifications.

---

## Architecture

### Workflow

```
User runs: pytest --summary-html

    ↓ pytest execution (normal pytest-pyvista flow)
    - Tests run, images generated, comparisons performed
    - Cache policies applied (--add_missing_images, --reset_image_cache, etc.)
    
    ↓ At teardown (after all tests complete)
    
    ↓ Report Generator hook fires:
       - Reads pytest-pyvista config (cache dirs, etc.)
       - Scans test results and image artifacts
       - Determines status for each image test
       - Generates self-contained HTML report
       - Saves to configured location (default: pytest_root/image_test_report.html)
    
    ↓ User opens: image_test_report.html in browser
    - Filters and reviews images
    - Selectively approves/unapproves images
    - Exports approval selections to JSON
    
    ↓ User runs: pytest --apply-approved-images approvals.json [--apply-target cache|staging]
    
    ↓ Approval Applier:
       - Reads exported JSON
       - Copies approved images to target directory
       - Reports what was copied/updated
```

### Components

#### 1. Report Generator (`SummaryReportGenerator` class)
- Triggered as a pytest hook at end of test session
- Reads pytest-pyvista configuration (cache dir, generated dir, failed dir)
- Collects test results from pytest's internal state
- Builds `TestResult` objects for each image test containing:
  - Test name, status (passed/failed/skipped/new)
  - Cached image path, generated image path, diff image path
  - Error value, threshold, environment info
  - Approval reason (why image is pre-approved, if applicable)
- Generates HTML with embedded images

#### 2. Difference Calculator
- Computes pixel-level differences between cached and generated images
- Creates visual difference map (highlights changed pixels in contrasting color)
- Returns error metrics for display

#### 3. HTML Report Template
- Self-contained single-file HTML with embedded CSS and JavaScript
- Header: Summary stats (total, passed, failed, skipped, new)
- Filter controls: Status checkboxes, search by test name, "Accept All New" convenience button
- Gallery view:
  - Card per test with color-coded badge
  - Three-panel image comparison: cached | generated | difference
  - Metadata panel with error value, threshold, test name, environment info, approval reason
  - Approval checkbox (only for Failed and New tests; pre-checked if pre-approved)
- Footer: "Export Approvals" button

#### 4. Approval State Manager (JavaScript)
- Manages approval selections in browser's localStorage
- Enables reversible approve/unapprove toggles
- Serializes selections to JSON on export
- Filters gallery based on user's status/search selections

#### 5. Approval Applier (CLI command)
- CLI command: `pytest --apply-approved-images <json_file> [--apply-target cache|staging]`
- Reads exported JSON file with approval selections
- Copies approved images to target directory
- Provides feedback on what was copied/updated

---

## Configuration

### CLI Flags

- `--summary-html [path]` - Generate report at given path (default: `image_test_report.html` at pytest root)
- `--summary-html-include passed,failed,new,skipped` - Filter what's included in report (default: all)
- `--apply-approved-images <json>` - Apply approvals from exported JSON file
- `--apply-target-dir cache|staging` - Where to copy approved images (default: staging)

### Configuration File (pyproject.toml or pytest.ini)

```toml
[tool.pytest.ini_options]
summary_html = "reports/image_test_report.html"  # Output path
summary_html_include = "passed,failed,new,skipped"  # What to include
```

---

## Image Status Definitions

- **Passed**: Generated image matches cached image within tolerance
- **Failed**: Generated image differs from cached image beyond tolerance
- **Skipped**: Test skipped or marked to skip image comparison
- **New**: Generated image with no corresponding cached baseline

### Pre-Approval Reasons

Images may be pre-approved (checkbox pre-checked, read-only) based on pytest flags used:
- `--add_missing_images`: New images automatically added to cache
- `--reset_image_cache`: All cached images regenerated
- `--reset_only_failed`: Failed images regenerated

Pre-approved images display an **Approval Reason** field in metadata explaining why they're pre-checked.

---

## UI/UX Design

### Header Section
- Title: "PyVista Image Test Report"
- Summary stats: Total | Passed | Failed | Skipped | New (each with count and color-coded badge)
- Metadata: Date, pytest version, pytest-pyvista version, Python version

### Filter Controls
- Checkboxes (default all checked): ☐ Passed | ☐ Failed | ☐ Skipped | ☐ New
- Search input for test name filtering
- "Accept All New" convenience checkbox (only visible if New tests exist)

### Test Gallery
Each test card contains:

**Card Header:**
- Color-coded badge (Passed 🟢 | Failed 🔴 | Skipped 🟡 | New 🔵)
- Test name
- Environment identifier (if applicable)

**Metadata Panel:**
- Error value (if applicable)
- Threshold value
- Status label
- Approval Reason (e.g., "Pre-approved via --add_missing_images" or blank if awaiting approval)
- Environment details (OS, Python version, PyVista version, VTK version, etc.)

**Image Comparison Panel (Three-column layout):**
- **Column 1 (Cached)**: "Cached Image" label + image thumbnail
- **Column 2 (Generated)**: "Generated Image" label + image thumbnail
- **Column 3 (Difference)**: "Pixel Differences" label + diff overlay (changed pixels highlighted in red with transparency)

**Approval Control:**
- Checkbox "Approve this image" (only visible for Failed and New tests)
- Pre-checked if image is pre-approved (read-only appearance)
- Can be toggled by user (reversible)

### Styling & Accessibility
- Color-coded badges using semantic colors (🟢 green for pass, 🔴 red for fail, 🟡 yellow for skip, 🔵 blue for new)
- Responsive layout (mobile-friendly with scrollable image panels)
- Dark/light theme support
- Accessible font sizes and contrast ratios
- Images displayed at reasonable size with zoom/expand option

### Footer Section
- "Export Approvals" button → downloads `approvals.json`
- Summary text: "X images selected for approval"

---

## Approval Workflow

### In Browser

1. User opens `image_test_report.html` in browser
2. Applies filters (status checkboxes, search term)
3. Reviews images in gallery
4. For Failed or New tests: clicks checkbox to "Approve this image"
5. Can toggle approval on/off at any time (reversible)
6. For convenience, clicks "Accept All New" to quickly approve all New images
7. Clicks "Export Approvals" button
   - Browser downloads `approvals.json` containing list of approved test names
   - JSON format: `{"approved": [{"test_name": "...", "type": "failed|new"}, ...]}`

### After Review

```bash
# Apply approvals to cache (direct update to cache directory)
pytest --apply-approved-images approvals.json --apply-target cache

# Or apply to staging directory (safer, for secondary review)
pytest --apply-approved-images approvals.json --apply-target staging
```

The applier reads the JSON, copies approved images to target, reports results.

### Key Behaviors

- **Pre-approved images** (from `--add_missing_images`, `--reset_image_cache`, etc.):
  - Checkbox is pre-checked
  - Checkbox appears disabled/grayed (read-only)
  - Unchecking has no effect; image remains in cache
  - Purpose: visibility/audit only
  
- **Images awaiting approval** (new or failed, not yet in cache):
  - Checkbox is unchecked
  - Checkbox is interactive
  - Checking adds image to export list
  - When applier runs, image is copied to cache or staging

---

## Implementation Notes

### Report Generation (Post-Test Hook)

The report generator runs as a pytest hook after all tests complete. It has access to:
- Pytest configuration (cache dirs, etc.)
- Test results (status, pass/fail, skip info)
- Generated images (from `generated_image_dir`)
- Cached images (from `image_cache_dir`)
- Failed image pairs (from `failed_image_dir` if available)

Image data is embedded as data URIs in the HTML for portability (no external file dependencies).

### Difference Calculation

Pixel differences are calculated once during report generation. The difference map uses:
- Red overlay (with transparency) to highlight changed pixels
- Side-by-side layout for easy visual comparison
- Metadata includes error value (sum of pixel differences) for quantitative assessment

### Browser Storage

Approval state is stored in browser's localStorage under a namespaced key to avoid conflicts. Clearing browser cache clears approvals (expected behavior; user can re-open report and re-select).

### Safety Considerations

- Approval export to JSON before cache commit provides explicit checkpoint
- Staging directory option allows secondary review before touching cache
- Pre-approved images are read-only (unchecking has no destructive effect)
- No automatic cache updates; all updates require explicit CLI step

---

## Configuration Examples

### Example 1: Default Configuration

```bash
pytest --summary-html
```

Generates `image_test_report.html` at pytest root with all test statuses included.

### Example 2: Configuration File

```toml
[tool.pytest.ini_options]
summary_html = "reports/image_summary.html"
summary_html_include = "passed,failed,new,skipped"
image_cache_dir = "tests/image_cache"
generated_image_dir = "generated_images"
failed_image_dir = "failed_images"
```

Then run:
```bash
pytest
```

Report is automatically generated if tests run successfully.

### Example 3: Using with Policies

```bash
# Add new images automatically, then review in report
pytest --add_missing_images --summary-html

# Open report, verify new images look correct, export approvals
# Then explicitly commit to cache:
pytest --apply-approved-images approvals.json --apply-target cache
```

---

## Testing Strategy

- **Unit tests**: Report generator, difference calculator, approval state logic
- **Integration tests**: Full flow (pytest run → report generation → HTML contains correct data and state)
- **Manual testing**: Open HTML in browser, verify filters work, toggle approvals, export JSON

---

## Future Enhancements (Out of Scope)

- Interactive approve/unapprove in report with automatic cache commit (requires server)
- Batch operations on filtered results
- Report history/diff between report runs
- Integration with CI/CD workflows (automated approval workflows)

---

## Acceptance Criteria

1. ✅ HTML report generated at end of pytest run with `--summary-html` flag
2. ✅ Report displays all image tests (passed, failed, skipped, new) with color-coded badges
3. ✅ Three-panel comparison view (cached | generated | difference) for each test
4. ✅ Metadata panel shows error value, threshold, test name, environment info, approval reason
5. ✅ Interactive filters (status checkboxes, search by name)
6. ✅ Selective, reversible approval checkboxes for failed/new images
7. ✅ Pre-approved images show approval reason and are read-only
8. ✅ "Accept All New" convenience button
9. ✅ "Export Approvals" button downloads JSON with approval selections
10. ✅ `--apply-approved-images` CLI command applies approvals to cache or staging
11. ✅ Configuration via CLI flags and pytest.ini/pyproject.toml
12. ✅ Works with existing pytest-pyvista policies (`--add_missing_images`, `--reset_image_cache`, etc.)
13. ✅ Report is self-contained (no external dependencies, images embedded)

