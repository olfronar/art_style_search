"""Report façade.

The concrete HTML section renderers, charts, and document assembly now live
under ``art_style_search.reporting``. This module preserves the existing
entrypoints and helper exports used by tests and local tooling.
"""

from __future__ import annotations

import logging
import webbrowser
from pathlib import Path

from art_style_search.report_data import ReportData, load_report_data
from art_style_search.reporting.document import _assemble_html

logger = logging.getLogger(__name__)

__all__ = [
    "ReportData",
    "build_all_reports",
    "build_report",
    "load_report_data",
]


def build_report(run_dir: Path, *, open_browser: bool = False, offline: bool = False) -> Path:
    """Generate ``runs/<run_dir>/report.html`` and return its path."""
    data = load_report_data(run_dir)
    report_path = run_dir / "report.html"
    html_doc = _assemble_html(data, report_path.parent, offline=offline)
    report_path.write_text(html_doc, encoding="utf-8")
    logger.info("Report written to %s", report_path)
    if open_browser:
        webbrowser.open(report_path.resolve().as_uri())
    return report_path


def build_all_reports(runs_dir: Path) -> list[Path]:
    """Regenerate reports for every run under *runs_dir* that has a state.json."""
    if not runs_dir.is_dir():
        return []
    results: list[Path] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "state.json").is_file():
            logger.info("Skipping %s: no state.json", run_dir.name)
            continue
        try:
            results.append(build_report(run_dir))
        except Exception:
            logger.exception("Failed to build report for %s", run_dir.name)
    return results
