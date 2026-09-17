#!/usr/bin/env python3
"""Build the complete static and interactive localisation comparison gallery.

Static pages are made from the campaign's existing per-group combined plots.
Each page is a 5x2 panel grid, with one panel per method and all ten seeds
plus the group mean already drawn in the source plot. The interactive viewer
is written alongside a metadata index; the local gallery server loads the
selected comparison's curves on demand.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROBLEM_ORDER = {"ho": 0, "heat": 1, "4d": 2}
MODE_ORDER = {"function_screen": 0, "same_network": 1, "mixed_size": 2}
VARIANT_ORDER = {"fixed": 0, "sigma": 1, "mu_sigma": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--page-width", type=int, default=1600)
    parser.add_argument("--page-height", type=int, default=2500)
    parser.add_argument("--limit", type=int, default=None, help="Build only the first N slices, for a smoke test")
    parser.add_argument("--skip-static", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_batch_root(project_root: Path) -> Path:
    candidates = [
        path for path in (project_root / "campaign_runs" / "project").glob("run_*")
        if (path / "run_manifest.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No campaign run with run_manifest.json was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def clean_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def display_name(family: str) -> str:
    if family == "baseline":
        return "Baseline"
    return family.replace("_", " ").title()


def variant_suffix(group: dict[str, Any]) -> str:
    family = group["localiser_family"]
    variant = group["variant"]
    prefix = f"{family}_"
    return variant[len(prefix) :] if variant.startswith(prefix) else variant


def slice_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        PROBLEM_ORDER.get(record["problem"], 99),
        float(record["domain_value"]),
        int(record["epochs"]),
        MODE_ORDER.get(record["comparison_mode"], 99),
        record["architecture_label"],
        VARIANT_ORDER.get(record["variant"], 99),
    )


def build_slice_index(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    groups = manifest.get("groups", [])
    group_by_path = {group["group_relpath"]: group for group in groups}
    families = [family for family in manifest.get("families", []) if family != "baseline"]
    buckets: dict[tuple[Any, ...], dict[str, Any]] = {}

    for group in groups:
        family = group.get("localiser_family")
        if family == "baseline":
            continue
        key = (
            group["problem"],
            group["domain_label"],
            group["domain_value"],
            group["budget_label"],
            group["epochs"],
            group["comparison_mode"],
            group["architecture_label"],
            group.get("baseline_architecture_label"),
            group.get("localised_architecture_label"),
            variant_suffix(group),
        )
        record = buckets.setdefault(
            key,
            {
                "problem": group["problem"],
                "domain_label": group["domain_label"],
                "domain_value": group["domain_value"],
                "budget_label": group["budget_label"],
                "epochs": group["epochs"],
                "comparison_mode": group["comparison_mode"],
                "architecture_label": group["architecture_label"],
                "baseline_architecture_label": group.get("baseline_architecture_label"),
                "localised_architecture_label": group.get("localised_architecture_label"),
                "variant": variant_suffix(group),
                "methods": OrderedDict(),
            },
        )
        record["methods"][family] = group["group_relpath"]
        record["baseline_relpath"] = group["pair_group_relpath"]

    records: list[dict[str, Any]] = []
    for record in sorted(buckets.values(), key=slice_sort_key):
        if not record.get("baseline_relpath"):
            continue
        ordered_methods = OrderedDict([("baseline", record["baseline_relpath"])])
        for family in families:
            if family in record["methods"]:
                ordered_methods[family] = record["methods"][family]
        record["methods"] = ordered_methods
        identity = [
            record["problem"],
            record["domain_label"],
            record["budget_label"],
            record["comparison_mode"],
            record["architecture_label"],
            record["baseline_architecture_label"] or "none",
            record["localised_architecture_label"] or "none",
            record["variant"],
        ]
        record["id"] = "__".join(clean_name(str(value)) for value in identity)
        record["method_labels"] = {family: display_name(family) for family in record["methods"]}
        record["method_count"] = len(record["methods"])
        records.append(record)
    return records, families


def page_title(record: dict[str, Any], metric: str, page: int) -> str:
    return (
        f"{record['problem'].upper()} {metric} | domain={record['domain_label']} | "
        f"budget={record['budget_label']} | mode={record['comparison_mode']} | "
        f"arch={record['architecture_label']} | variant={record['variant']} | page={page}"
    )


def load_font() -> ImageFont.ImageFont:
    for font_path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ):
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, 24)
            except OSError:
                pass
    return ImageFont.load_default()


def make_static_page(
    batch_root: Path,
    gallery_root: Path,
    record: dict[str, Any],
    metric: str,
    page_number: int,
    page_width: int,
    page_height: int,
) -> Path:
    methods = list(record["methods"].items())
    page_size = 10
    page_methods = methods[(page_number - 1) * page_size : page_number * page_size]
    canvas = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font()
    title = page_title(record, metric, page_number)
    draw.text((20, 12), title, fill="#172b4d", font=font)

    tile_width = page_width // 2
    tile_height = (page_height - 62) // 5
    image_margin_x = 10
    image_margin_top = 38
    image_width = tile_width - 2 * image_margin_x
    image_height = tile_height - image_margin_top - 8
    filename = "combined_loss_plot.png" if metric == "loss" else "combined_solution_plot.png"

    for index in range(page_size):
        x = (index % 2) * tile_width
        y = 62 + (index // 2) * tile_height
        if index >= len(page_methods):
            continue
        family, group_relpath = page_methods[index]
        label = display_name(family)
        draw.text((x + image_margin_x, y + 5), label, fill="#243b53", font=font)
        source = batch_root / group_relpath / filename
        if source.exists():
            try:
                with Image.open(source) as source_image:
                    image = source_image.convert("RGB")
                    image.thumbnail((image_width, image_height), Image.Resampling.LANCZOS)
                    paste_x = x + (tile_width - image.width) // 2
                    paste_y = y + image_margin_top + (image_height - image.height) // 2
                    canvas.paste(image, (paste_x, paste_y))
            except OSError:
                draw.text((x + image_margin_x, y + image_margin_top), "Unreadable source plot", fill="#9b2226")
        else:
            draw.text((x + image_margin_x, y + image_margin_top), "Missing source plot", fill="#9b2226")

    output_dir = gallery_root / "static" / record["id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{metric}_page_{page_number:02d}.png"
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


def write_gallery_html(gallery_root: Path, records: list[dict[str, Any]], families: list[str]) -> Path:
    index_json = json.dumps({"slices": records, "families": families}, separators=(",", ":"))
    html_path = gallery_root / "interactive_comparison.html"
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Localisation campaign explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{ color-scheme: light; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172b4d; background: #f7f9fc; }}
    header {{ padding: 20px 24px 10px; background: white; border-bottom: 1px solid #d9e2ec; }}
    h1 {{ margin: 0 0 6px; font-size: 23px; }}
    p {{ margin: 0; color: #52606d; }}
    .controls {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px 14px; padding: 14px 24px; background: #eef2f7; border-bottom: 1px solid #d9e2ec; }}
    label {{ display: grid; gap: 4px; font-size: 12px; font-weight: 700; color: #334e68; }}
    select {{ width: 100%; padding: 6px 8px; border: 1px solid #9fb3c8; border-radius: 4px; background: white; color: #172b4d; }}
    .functions {{ grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; }}
    .functions label {{ display: inline-flex; grid-template-columns: auto auto; align-items: center; gap: 4px; font-size: 12px; font-weight: 500; }}
    button {{ padding: 5px 9px; border: 1px solid #829ab1; border-radius: 4px; background: white; color: #243b53; cursor: pointer; }}
    #status {{ padding: 8px 24px; min-height: 18px; color: #52606d; background: white; font-size: 13px; }}
    #plot {{ width: 100%; height: calc(100vh - 250px); min-height: 600px; background: white; }}
  </style>
</head>
<body>
  <header><h1>Localisation campaign explorer</h1><p>All matched equation, domain, budget, architecture, comparison-mode, and variant slices from the completed campaign.</p></header>
  <div class="controls">
    <label>Equation<select id="problem"></select></label>
    <label>Domain<select id="domain"></select></label>
    <label>Budget<select id="budget"></select></label>
    <label>Comparison mode<select id="mode"></select></label>
    <label>Architecture<select id="architecture"></select></label>
    <label>Variant<select id="variant"></select></label>
    <label>Metric<select id="metric"><option value="loss">Loss</option><option value="solution">Recovered solution</option></select></label>
    <label>Runs<select id="view"><option value="runs">All 10 runs</option><option value="mean">Mean</option></select></label>
    <div class="functions" id="functions"><strong>Functions:</strong><button id="all">All</button><button id="none">None</button></div>
  </div>
  <div id="status">Loading selected comparison...</div>
  <div id="plot"></div>
  <script>
    const index = {index_json};
    const slices = index.slices;
    const selects = Object.fromEntries(['problem','domain','budget','mode','architecture','variant','metric','view'].map(id => [id, document.getElementById(id)]));
    const statusNode = document.getElementById('status');
    let currentData = null;
    let selectedFamilies = new Set(['baseline', ...index.families]);
    let requestNumber = 0;

    function unique(field, predicate = () => true) {{
      return [...new Set(slices.filter(predicate).map(item => item[field]))];
    }}
    function setOptions(select, values, preferred) {{
      const old = select.value;
      select.replaceChildren(...values.map(value => {{ const option = document.createElement('option'); option.value = value; option.textContent = value; return option; }}));
      select.value = values.includes(old) ? old : (values.includes(preferred) ? preferred : values[0] || '');
    }}
    function matching() {{
      return slices.filter(item =>
        item.problem === selects.problem.value &&
        item.domain_label === selects.domain.value &&
        item.budget_label === selects.budget.value &&
        item.comparison_mode === selects.mode.value &&
        item.architecture_label === selects.architecture.value &&
        item.variant === selects.variant.value
      );
    }}
    function refreshOptions(changed) {{
      const p = selects.problem.value;
      setOptions(selects.domain, unique('domain_label', item => item.problem === p));
      setOptions(selects.budget, unique('budget_label', item => item.problem === p && item.domain_label === selects.domain.value));
      setOptions(selects.mode, unique('comparison_mode', item => item.problem === p && item.domain_label === selects.domain.value && item.budget_label === selects.budget.value));
      setOptions(selects.architecture, unique('architecture_label', item => item.problem === p && item.domain_label === selects.domain.value && item.budget_label === selects.budget.value && item.comparison_mode === selects.mode.value));
      setOptions(selects.variant, unique('variant', item => item.problem === p && item.domain_label === selects.domain.value && item.budget_label === selects.budget.value && item.comparison_mode === selects.mode.value && item.architecture_label === selects.architecture.value));
      if (changed !== 'problem') {{
        setOptions(selects.domain, unique('domain_label', item => item.problem === p), selects.domain.value);
        setOptions(selects.budget, unique('budget_label', item => item.problem === p && item.domain_label === selects.domain.value), selects.budget.value);
        setOptions(selects.mode, unique('comparison_mode', item => item.problem === p && item.domain_label === selects.domain.value && item.budget_label === selects.budget.value), selects.mode.value);
        setOptions(selects.architecture, unique('architecture_label', item => item.problem === p && item.domain_label === selects.domain.value && item.budget_label === selects.budget.value && item.comparison_mode === selects.mode.value), selects.architecture.value);
        setOptions(selects.variant, unique('variant', item => item.problem === p && item.domain_label === selects.domain.value && item.budget_label === selects.budget.value && item.comparison_mode === selects.mode.value && item.architecture_label === selects.architecture.value), selects.variant.value);
      }}
      loadSelected();
    }}
    function renderFunctionToggles() {{
      document.querySelectorAll('.function-toggle').forEach(node => node.remove());
      const families = ['baseline', ...index.families];
      families.forEach(family => {{
        const label = document.createElement('label'); label.className = 'function-toggle';
        const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = selectedFamilies.has(family); checkbox.dataset.family = family;
        checkbox.addEventListener('change', () => {{ checkbox.checked ? selectedFamilies.add(family) : selectedFamilies.delete(family); renderPlot(); }});
        label.append(checkbox, document.createTextNode(family === 'baseline' ? 'Baseline' : family.replaceAll('_', ' ')));
        document.getElementById('functions').appendChild(label);
      }});
    }}
    function selectedSlice() {{ return matching()[0]; }}
    async function loadSelected() {{
      const record = selectedSlice(); if (!record) return;
      const id = ++requestNumber; statusNode.textContent = `Loading ${{record.problem}} ${{record.domain_label}} ${{record.budget_label}} ${{record.comparison_mode}} ${{record.variant}}...`;
      try {{
        const response = await fetch(`/api/data?id=${{encodeURIComponent(record.id)}}`);
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const data = await response.json(); if (id !== requestNumber) return;
        currentData = data; statusNode.textContent = `${{data.selection.problem.toUpperCase()}} | domain=${{data.selection.domain_label}} | budget=${{data.selection.budget_label}} | mode=${{data.selection.comparison_mode}} | arch=${{data.selection.architecture_label}} | variant=${{data.selection.variant}}`;
        renderPlot();
      }} catch (error) {{ statusNode.textContent = `Could not load selected comparison: ${{error.message}}. Serve this directory with serve_localisation_gallery.py.`; }}
    }}
    function renderPlot() {{
      if (!currentData) return;
      const traces = [];
      currentData.methods.filter(method => selectedFamilies.has(method.family)).forEach(method => {{
        const series = method[selects.metric.value]; if (!series) return;
        if (selects.view.value === 'mean') {{
          if (series.mean) traces.push({{ x: series.mean.x, y: series.mean.y, mode: 'lines', name: `${{method.label}} mean`, line: {{ width: 2.6, color: method.color }}, connectgaps: false }});
        }} else {{
          series.runs.forEach(run => traces.push({{ x: run.x, y: run.y, mode: 'lines', name: `${{method.label}} seed ${{String(run.seed).padStart(2,'0')}}`, line: {{ width: 1.0, color: method.color }}, opacity: 0.34, connectgaps: false }}));
        }}
      }});
      if (selects.metric.value === 'solution' && currentData.reference) traces.push({{ x: currentData.reference.x, y: currentData.reference.y, mode: 'lines', name: 'Reference', line: {{ width: 2.5, color: '#111111', dash: 'dash' }} }});
      Plotly.react('plot', traces, {{ hovermode: 'x unified', margin: {{ l: 70, r: 25, t: 40, b: 70 }}, paper_bgcolor: 'white', plot_bgcolor: 'white', xaxis: {{ title: selects.metric.value === 'loss' ? 'Epoch' : 'Domain coordinate', gridcolor: '#d9e2ec' }}, yaxis: {{ title: selects.metric.value === 'loss' ? 'log10(train loss)' : 'Recovered solution', gridcolor: '#d9e2ec' }}, legend: {{ orientation: 'h', y: -0.18 }}, title: `${{selects.metric.options[selects.metric.selectedIndex].text}} | ${{selects.view.options[selects.view.selectedIndex].text}}` }}, {{ responsive: true, displaylogo: false }});
    }}
    document.getElementById('all').addEventListener('click', () => {{ selectedFamilies = new Set(['baseline', ...index.families]); renderFunctionToggles(); renderPlot(); }});
    document.getElementById('none').addEventListener('click', () => {{ selectedFamilies = new Set(); renderFunctionToggles(); renderPlot(); }});
    ['problem','domain','budget','mode','architecture','variant'].forEach(id => selects[id].addEventListener('change', () => refreshOptions(id)));
    ['metric','view'].forEach(id => selects[id].addEventListener('change', renderPlot));
    setOptions(selects.problem, unique('problem'), 'ho');
    refreshOptions('problem');
    renderFunctionToggles();
  </script>
</body>
</html>
"""
    html_path.write_text(document, encoding="utf-8")
    return html_path


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    batch_root = args.batch_root.resolve() if args.batch_root else find_batch_root(project_root)
    gallery_root = args.output_dir.resolve() if args.output_dir else batch_root / "complete_gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(batch_root / "run_manifest.json")
    records, families = build_slice_index(manifest)
    if args.limit is not None:
        records = records[: max(0, args.limit)]

    for index, record in enumerate(records, start=1):
        record["static"] = {}
        if not args.skip_static:
            for metric in ("loss", "solution"):
                page_count = (record["method_count"] + 9) // 10
                record["static"][metric] = [
                    str(
                        make_static_page(
                            batch_root,
                            gallery_root,
                            record,
                            metric,
                            page,
                            args.page_width,
                            args.page_height,
                        ).relative_to(gallery_root)
                    )
                    for page in range(1, page_count + 1)
                ]
        if index == 1 or index % 50 == 0 or index == len(records):
            print(f"built_slices={index}/{len(records)}", flush=True)

    index_payload = {
        "batch_root": str(batch_root),
        "families": families,
        "slice_count": len(records),
        "slices": records,
    }
    (gallery_root / "gallery_index.json").write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")
    interactive_path = write_gallery_html(gallery_root, records, families)
    summary = {
        "batch_root": str(batch_root),
        "gallery_root": str(gallery_root),
        "slice_count": len(records),
        "methods_per_slice": sorted({record["method_count"] for record in records}),
        "static_pages": sum(sum(len(paths) for paths in record["static"].values()) for record in records),
        "interactive": str(interactive_path),
        "server_command": f".venv/bin/python experiments/serve_localisation_gallery.py --gallery-root {gallery_root}",
    }
    (gallery_root / "gallery_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
