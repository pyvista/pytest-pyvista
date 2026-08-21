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
        "search",
        "sort",
        "count",
        "article.card",
        "label.approve input[type=checkbox]",
        "data-run-id",
        "data-status",
        "data-key",
        "data-name",
        "data-error",
        "data-missing-error",
    ],
)
def test_script_wires_up_each_documented_hook(hook: str) -> None:
    """Every element and attribute the renderer emits is referenced by the script."""
    assert hook in SCRIPT


def test_script_discards_state_from_a_different_run() -> None:
    """State is keyed by run id, and a discarded selection is explained in the notice."""
    assert "run_id" in SCRIPT
    assert "notice" in SCRIPT


def test_script_prunes_old_keys() -> None:
    """Stored approvals expire rather than accumulating forever."""
    assert "PRUNE_AFTER_DAYS" in SCRIPT


def test_script_has_no_external_requests() -> None:
    """The report is a local file: nothing in it may reach the network."""
    for forbidden in ("fetch(", "XMLHttpRequest", "https://", "http://"):
        assert forbidden not in SCRIPT


def test_script_cannot_close_the_element_it_is_inlined_into() -> None:
    """The script is embedded in the page, so a closing script tag would truncate it."""
    assert "</script>" not in SCRIPT


def test_script_takes_the_schema_version_from_the_embedded_manifest() -> None:
    """Python owns SCHEMA_VERSION; the export echoes it rather than holding a second copy."""
    assert "manifest.schema_version" in SCRIPT
    assert "schema_version: 1" not in SCRIPT
