"""Final report document assembly."""

from __future__ import annotations

from importlib import resources

from art_style_search.report_data import ReportData
from art_style_search.reporting.charts import _build_composite_trajectory, _build_per_metric_trajectories
from art_style_search.reporting.render import (
    _h,
    _render_header,
    _render_holdout_section,
    _render_iteration_drilldown,
    _render_kb_section,
    _render_promotion_section,
    _render_protocol_section,
    _render_trajectories_section,
)

REPORT_CSS = resources.files("art_style_search.reporting").joinpath("report.css").read_text(encoding="utf-8")

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
_FONTS_CDN = (
    "https://fonts.googleapis.com/css2"
    "?family=Fraunces:ital,opsz,wght@0,9..144,300..900;1,9..144,300..900"
    "&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400"
    "&family=IBM+Plex+Mono:wght@400;500"
    "&display=swap"
)


def _plotly_script_tag(*, offline: bool = False) -> str:
    """Return a <script> tag that loads Plotly — CDN by default, inline if offline."""
    if not offline:
        return f'<script src="{_PLOTLY_CDN}"></script>'

    js_path = resources.files("plotly.package_data").joinpath("plotly.min.js")
    plotly_js = js_path.read_text(encoding="utf-8")
    return f"<script>{plotly_js}</script>"


def _assemble_html(data: ReportData, report_dir, *, offline: bool = False) -> str:
    composite_json = ""
    multi_json = ""
    if data.iteration_logs:
        composite_json = _build_composite_trajectory(data)
        multi_json = _build_per_metric_trajectories(data)

    header = _render_header(data)
    trajectories = _render_trajectories_section(composite_json, multi_json)
    iterations_section = _render_iteration_drilldown(data, report_dir)
    kb_section = _render_kb_section(data)
    protocol_section = _render_protocol_section(data)
    promotion_section = _render_promotion_section(data)
    holdout_section = _render_holdout_section(data)

    plot_script = ""
    if composite_json and multi_json:
        plot_script = """
<script>
(function() {
  function parseAndPlot(elemId, dataId) {
    var el = document.getElementById(dataId);
    if (!el) return;
    try {
      var fig = JSON.parse(el.textContent);
      Plotly.newPlot(elemId, fig.data, fig.layout, {responsive: true, displaylogo: false});
    } catch (e) {
      console.error("Failed to render " + elemId, e);
    }
  }
  parseAndPlot("composite-chart", "composite-data");
  parseAndPlot("metrics-chart", "metrics-data");
})();
</script>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Art Style Search · {_h(data.run_name)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="{_FONTS_CDN}">
  {_plotly_script_tag(offline=offline)}
  <style>{REPORT_CSS}</style>
</head>
<body>
  <main class="page">
    {header}
    {trajectories}
    {iterations_section}
    {kb_section}
    {protocol_section}
    {promotion_section}
    {holdout_section}
  </main>
  {plot_script}
</body>
</html>
"""
