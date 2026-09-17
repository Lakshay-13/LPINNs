#!/usr/bin/env python3
"""Make paired multi-function localisation comparison plots from a campaign.

The script never retrains a model. It reads the campaign's per-seed
``loss_history.npy`` and ``solution_slice.npz`` artifacts and writes:

* 5x2 seed grids for loss and recovered solution curves;
* 5x2 method grids (all ten seeds per method), split into pages when the
  campaign contains more than nine localiser families;
* a Plotly HTML with controls for metric, function, and all-runs versus mean.

The interactive loss payload is deterministically downsampled for large
budgets; the static PNGs still use the full stored histories.

The default selection is the HO function screen at 2pi, 100k epochs, 64x64,
and the learnable-mu-and-sigma variant. Use the CLI to select another matched
comparison slice.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SEEDS = tuple(range(1, 11))
DEFAULTS = {
    "ho": {"domain": "2pi", "budget": "100k", "architecture": "64x64"},
    "heat": {"domain": "8pi", "budget": "50k", "architecture": "64x64"},
    "4d": {"domain": "4pi", "budget": "10k", "architecture": "32x32"},
}
VARIANTS = {"fixed", "sigma", "mu_sigma"}
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

METHOD_COLORS = [
    "#1b4965",
    "#ca6702",
    "#2a9d8f",
    "#9b2226",
    "#6a4c93",
    "#588157",
    "#bc6c25",
    "#0077b6",
    "#c9184a",
    "#386641",
    "#7f5539",
    "#5a189a",
    "#0a9396",
    "#ae2012",
]


@dataclass
class MethodData:
    family: str
    label: str
    group: dict[str, Any]
    loss: dict[int, np.ndarray]
    x: np.ndarray | None
    solution: dict[int, np.ndarray]
    reference: np.ndarray | None
    invalid_seeds: list[int]
    missing_seeds: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=None)
    parser.add_argument("--problem", choices=["ho", "heat", "4d"], default="ho")
    parser.add_argument("--domain", default=None, help="Manifest domain label, e.g. 2pi or 8pi")
    parser.add_argument("--budget", default=None, help="Manifest budget label, e.g. 10k or 100k")
    parser.add_argument("--architecture", default=None, help="Manifest architecture label, e.g. 64x64")
    parser.add_argument("--comparison-mode", default="function_screen")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="mu_sigma")
    parser.add_argument(
        "--families",
        default="all",
        help="Comma-separated families, or 'all' for every family in the manifest",
    )
    parser.add_argument("--seeds", default=','.join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=2)
    parser.add_argument(
        "--interactive-points",
        type=int,
        default=2000,
        help="Maximum points per interactive curve; static plots are not downsampled",
    )
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
        raise FileNotFoundError("No completed campaign run with run_manifest.json was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def display_name(family: str) -> str:
    if family == "baseline":
        return "Baseline"
    return family.replace("_", " ").title()


def clean_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def find_group(
    groups: list[dict[str, Any]],
    *,
    problem: str,
    comparison_mode: str,
    domain: str,
    budget: str,
    architecture: str,
    family: str,
    variant: str | None,
) -> dict[str, Any] | None:
    for group in groups:
        if group.get("problem") != problem:
            continue
        if group.get("comparison_mode") != comparison_mode:
            continue
        if group.get("domain_label") != domain:
            continue
        if group.get("budget_label") != budget:
            continue
        if group.get("architecture_label") != architecture:
            continue
        if group.get("localiser_family") != family:
            continue
        if variant is not None and group.get("variant") != variant:
            continue
        return group
    return None


def load_method(batch_root: Path, group: dict[str, Any], seeds: list[int]) -> MethodData:
    loss: dict[int, np.ndarray] = {}
    solution: dict[int, np.ndarray] = {}
    invalid: list[int] = []
    missing: list[int] = []
    x: np.ndarray | None = None
    reference: np.ndarray | None = None

    for seed in seeds:
        seed_dir = batch_root / group["group_relpath"] / f"seed_{seed:02d}"
        loss_path = seed_dir / "loss_history.npy"
        slice_path = seed_dir / "solution_slice.npz"
        metrics_path = seed_dir / "metrics.json"
        if not loss_path.exists() or not slice_path.exists():
            missing.append(seed)
            continue
        try:
            history = np.asarray(np.load(loss_path), dtype=np.float64).reshape(-1)
            bundle = np.load(slice_path, allow_pickle=True)
            curve_x = np.asarray(bundle["curve_x"], dtype=np.float64).reshape(-1)
            prediction = np.asarray(bundle["prediction_curve"], dtype=np.float64).reshape(-1)
            curve_reference = np.asarray(bundle["reference_curve"], dtype=np.float64).reshape(-1)
            loss[seed] = history
            solution[seed] = prediction
            if x is None:
                x = curve_x
                reference = curve_reference
            if not np.isfinite(history).all() or not np.isfinite(prediction).all():
                invalid.append(seed)
            if metrics_path.exists():
                metrics = read_json(metrics_path)
                if metrics.get("status") != "completed" and seed not in invalid:
                    invalid.append(seed)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            missing.append(seed)

    return MethodData(
        family=group["localiser_family"],
        label=display_name(group["localiser_family"]),
        group=group,
        loss=loss,
        x=x,
        solution=solution,
        reference=reference,
        invalid_seeds=sorted(set(invalid)),
        missing_seeds=sorted(set(missing)),
    )


def finite_mean(series: list[np.ndarray]) -> np.ndarray | None:
    if not series:
        return None
    width = max(len(item) for item in series)
    padded = np.full((len(series), width), np.nan, dtype=np.float64)
    for row, item in enumerate(series):
        padded[row, : len(item)] = item
    finite = np.isfinite(padded)
    counts = finite.sum(axis=0)
    totals = np.where(finite, padded, 0.0).sum(axis=0)
    result = np.full(width, np.nan, dtype=np.float64)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def safe_log_loss(history: np.ndarray) -> np.ndarray:
    result = np.full(history.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(history) & (history > 0)
    result[finite] = np.log10(history[finite])
    return result


def add_grid_title(fig: plt.Figure, methods: list[MethodData], args: argparse.Namespace, title: str) -> None:
    config = (
        f"{args.problem.upper()} | domain={args.domain} | budget={args.budget} | "
        f"arch={args.architecture} | mode={args.comparison_mode} | variant={args.variant}"
    )
    fig.suptitle(f"{title}\n{config}", fontsize=16, y=0.995)
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[index % len(METHOD_COLORS)], linewidth=2.2, label=method.label)
        for index, method in enumerate(methods)
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), frameon=False)


def configure_axis(ax: plt.Axes, *, ylabel: str, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch" if ylabel == "log10(train loss)" else "Domain coordinate")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def plot_seed_grid(
    methods: list[MethodData],
    seeds: list[int],
    args: argparse.Namespace,
    output_path: Path,
    metric: str,
) -> None:
    fig, axes = plt.subplots(
        args.grid_rows,
        args.grid_cols,
        figsize=(7.2 * args.grid_cols, 4.6 * args.grid_rows),
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    for index, seed in enumerate(seeds):
        ax = flat_axes[index]
        for method_index, method in enumerate(methods):
            color = METHOD_COLORS[method_index % len(METHOD_COLORS)]
            if metric == "loss":
                values = method.loss.get(seed)
                if values is None:
                    continue
                ax.plot(np.arange(1, len(values) + 1), safe_log_loss(values), color=color, linewidth=1.0, alpha=0.78)
            else:
                values = method.solution.get(seed)
                if values is None or method.x is None:
                    continue
                ax.plot(method.x, values, color=color, linewidth=1.0, alpha=0.78)
        if metric == "solution":
            reference = next((method.reference for method in methods if method.reference is not None), None)
            x = next((method.x for method in methods if method.x is not None), None)
            if reference is not None and x is not None:
                ax.plot(x, reference, color="#111111", linewidth=2.3, linestyle="--", label="Reference")
        configure_axis(ax, ylabel="log10(train loss)" if metric == "loss" else "Recovered solution", title=f"Seed {seed:02d}")
    for ax in flat_axes[len(seeds) :]:
        ax.axis("off")
    add_grid_title(fig, methods, args, f"{metric.title()} comparison by paired seed")
    fig.subplots_adjust(top=0.91, bottom=0.1, hspace=0.35, wspace=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_method_pages(
    methods: list[MethodData],
    seeds: list[int],
    args: argparse.Namespace,
    output_dir: Path,
    metric: str,
) -> list[str]:
    page_size = args.grid_rows * args.grid_cols
    paths: list[str] = []
    for page_start in range(0, len(methods), page_size):
        page_methods = methods[page_start : page_start + page_size]
        page_number = page_start // page_size + 1
        fig, axes = plt.subplots(
            args.grid_rows,
            args.grid_cols,
            figsize=(7.2 * args.grid_cols, 4.6 * args.grid_rows),
            squeeze=False,
        )
        flat_axes = list(axes.flat)
        for index, method in enumerate(page_methods):
            ax = flat_axes[index]
            color = METHOD_COLORS[(page_start + index) % len(METHOD_COLORS)]
            if metric == "loss":
                for seed in seeds:
                    values = method.loss.get(seed)
                    if values is not None:
                        ax.plot(
                            np.arange(1, len(values) + 1),
                            safe_log_loss(values),
                            color=color,
                            linewidth=0.8,
                            alpha=0.24,
                        )
                mean = finite_mean(list(method.loss.values()))
                if mean is not None:
                    ax.plot(np.arange(1, len(mean) + 1), safe_log_loss(mean), color=color, linewidth=2.2, label="mean")
            else:
                for seed in seeds:
                    values = method.solution.get(seed)
                    if values is not None and method.x is not None:
                        ax.plot(method.x, values, color=color, linewidth=0.8, alpha=0.22)
                mean = finite_mean(list(method.solution.values()))
                if mean is not None and method.x is not None:
                    ax.plot(method.x, mean, color=color, linewidth=2.2, label="mean")
                if method.reference is not None and method.x is not None:
                    ax.plot(method.x, method.reference, color="#111111", linewidth=1.8, linestyle="--", label="reference")
            title = method.label
            if method.invalid_seeds or method.missing_seeds:
                title += f"\ninvalid={len(method.invalid_seeds)} missing={len(method.missing_seeds)}"
            configure_axis(ax, ylabel="log10(train loss)" if metric == "loss" else "Recovered solution", title=title)
        for ax in flat_axes[len(page_methods) :]:
            ax.axis("off")
        add_grid_title(fig, page_methods, args, f"{metric.title()} comparison by function, page {page_number}")
        fig.subplots_adjust(top=0.91, bottom=0.1, hspace=0.35, wspace=0.25)
        path = output_dir / f"{metric}_by_function_page_{page_number:02d}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def json_series(values: np.ndarray) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def downsample(values: np.ndarray, max_points: int) -> np.ndarray:
    if max_points < 2:
        raise ValueError("--interactive-points must be at least 2")
    if len(values) <= max_points:
        return values
    indices = np.linspace(0, len(values) - 1, max_points, dtype=np.int64)
    return values[np.unique(indices)]


def build_interactive_html(
    methods: list[MethodData],
    seeds: list[int],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    traces: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    method_options = [{"value": "all", "label": "All functions"}]

    for method in methods:
        method_options.append({"value": method.family, "label": method.label})

    def add_trace(trace: dict[str, Any], spec: dict[str, Any]) -> None:
        traces.append(trace)
        specs.append(spec)

    for metric in ("loss", "solution"):
        for view in ("all_runs", "mean"):
            for method_index, method in enumerate(methods):
                color = METHOD_COLORS[method_index % len(METHOD_COLORS)]
                if view == "all_runs":
                    for seed in seeds:
                        if metric == "loss":
                            values = method.loss.get(seed)
                            if values is None:
                                continue
                            sampled = downsample(values, args.interactive_points)
                            x = (np.linspace(1, len(values), len(sampled))).tolist()
                            y = json_series(safe_log_loss(sampled))
                            y_title = "log10(train loss)"
                        else:
                            values = method.solution.get(seed)
                            if values is None or method.x is None:
                                continue
                            sampled_indices = np.linspace(0, len(values) - 1, min(len(values), args.interactive_points), dtype=np.int64)
                            x = json_series(method.x[np.unique(sampled_indices)])
                            y = json_series(values[np.unique(sampled_indices)])
                            y_title = "Recovered solution"
                        add_trace(
                            {
                                "x": x,
                                "y": y,
                                "mode": "lines",
                                "name": f"{method.label} seed {seed:02d}",
                                "line": {"color": color, "width": 1.1},
                                "opacity": 0.34,
                                "hovertemplate": f"{method.label} seed {seed:02d}<br>%{{x}}<br>%{{y:.5g}}<extra></extra>",
                            },
                            {"metric": metric, "view": view, "family": method.family, "y_title": y_title},
                        )
                else:
                    if metric == "loss":
                        values = finite_mean(list(method.loss.values()))
                        if values is None:
                            continue
                        sampled = downsample(values, args.interactive_points)
                        x = (np.linspace(1, len(values), len(sampled))).tolist()
                        y = json_series(safe_log_loss(sampled))
                        y_title = "log10(train loss)"
                    else:
                        values = finite_mean(list(method.solution.values()))
                        if values is None or method.x is None:
                            continue
                        sampled_indices = np.linspace(0, len(values) - 1, min(len(values), args.interactive_points), dtype=np.int64)
                        x = json_series(method.x[np.unique(sampled_indices)])
                        y = json_series(values[np.unique(sampled_indices)])
                        y_title = "Recovered solution"
                    add_trace(
                        {
                            "x": x,
                            "y": y,
                            "mode": "lines",
                            "name": f"{method.label} mean",
                            "line": {"color": color, "width": 2.8},
                            "hovertemplate": f"{method.label} mean<br>%{{x}}<br>%{{y:.5g}}<extra></extra>",
                        },
                        {"metric": metric, "view": view, "family": method.family, "y_title": y_title},
                    )

    reference = next((method for method in methods if method.reference is not None and method.x is not None), None)
    if reference is not None:
        add_trace(
            {
                "x": json_series(reference.x),
                "y": json_series(reference.reference),
                "mode": "lines",
                "name": "Reference",
                "line": {"color": "#111111", "width": 2.5, "dash": "dash"},
                "hovertemplate": "Reference<br>%{x}<br>%{y:.5g}<extra></extra>",
            },
            {"metric": "solution", "view": "reference", "family": "__reference__", "y_title": "Recovered solution"},
        )

    payload = {
        "traces": traces,
        "specs": specs,
        "method_options": method_options,
        "title": f"{args.problem.upper()} localisation comparison",
        "config": {
            "problem": args.problem,
            "domain": args.domain,
            "budget": args.budget,
            "architecture": args.architecture,
            "comparison_mode": args.comparison_mode,
            "variant": args.variant,
        },
    }
    payload_json = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    options_html = "".join(
        f'<option value="{html.escape(option["value"])}">{html.escape(option["label"])}</option>'
        for option in method_options
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(args.problem.upper())} localisation comparison</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; }}
    header {{ padding: 18px 24px 8px; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    p {{ margin: 0; color: #52606d; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 12px 18px; padding: 14px 24px; background: #f0f4f8; border-bottom: 1px solid #d9e2ec; }}
    label {{ display: flex; gap: 8px; align-items: center; font-size: 14px; font-weight: 600; }}
    select {{ padding: 5px 8px; border: 1px solid #9fb3c8; border-radius: 4px; background: white; }}
    #plot {{ width: 100%; height: calc(100vh - 155px); min-height: 560px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(args.problem.upper())} localisation comparison</h1>
    <p>domain={html.escape(args.domain)} | budget={html.escape(args.budget)} | architecture={html.escape(args.architecture)} | mode={html.escape(args.comparison_mode)} | variant={html.escape(args.variant)}</p>
  </header>
  <div class="controls">
    <label>Metric <select id="metric"><option value="loss">Loss</option><option value="solution">Solution</option></select></label>
    <label>Function <select id="function">{options_html}</select></label>
    <label>Runs <select id="view"><option value="all_runs">All runs</option><option value="mean">Mean</option></select></label>
  </div>
  <div id="plot"></div>
  <script>
    const payload = {payload_json};
    const plot = document.getElementById('plot');
    const metric = document.getElementById('metric');
    const functionSelect = document.getElementById('function');
    const view = document.getElementById('view');

    function visibleTrace(spec) {{
      if (spec.family === '__reference__') return metric.value === 'solution';
      return spec.metric === metric.value && spec.view === view.value &&
        (functionSelect.value === 'all' || spec.family === functionSelect.value);
    }}

    function render() {{
      const data = payload.traces.map((trace, index) => ({{ ...trace, visible: visibleTrace(payload.specs[index]) }}));
      const yTitle = metric.value === 'loss' ? 'log10(train loss)' : 'Recovered solution';
      const title = `${{payload.title}} | ${{metric.options[metric.selectedIndex].text}} | ${{functionSelect.options[functionSelect.selectedIndex].text}} | ${{view.options[view.selectedIndex].text}}`;
      Plotly.react(plot, data, {{
        title: {{ text: title, x: 0.02 }},
        paper_bgcolor: 'white', plot_bgcolor: 'white',
        hovermode: 'x unified',
        margin: {{ l: 70, r: 25, t: 70, b: 55 }},
        xaxis: {{ title: metric.value === 'loss' ? 'Epoch' : 'Domain coordinate', gridcolor: '#d9e2ec' }},
        yaxis: {{ title: yTitle, gridcolor: '#d9e2ec' }},
        legend: {{ orientation: 'h', y: -0.15 }},
      }}, {{ responsive: true, displaylogo: false }});
    }}
    metric.addEventListener('change', render);
    functionSelect.addEventListener('change', render);
    view.addEventListener('change', render);
    render();
  </script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    batch_root = args.batch_root.resolve() if args.batch_root else find_batch_root(project_root)
    manifest = read_json(batch_root / "run_manifest.json")
    args.domain = args.domain or DEFAULTS[args.problem]["domain"]
    args.budget = args.budget or DEFAULTS[args.problem]["budget"]
    args.architecture = args.architecture or DEFAULTS[args.problem]["architecture"]
    seeds = parse_seeds(args.seeds)

    groups = manifest.get("groups", [])
    baseline_group = find_group(
        groups,
        problem=args.problem,
        comparison_mode=args.comparison_mode,
        domain=args.domain,
        budget=args.budget,
        architecture=args.architecture,
        family="baseline",
        variant="baseline_fc",
    )
    if baseline_group is None:
        raise SystemExit("No matched baseline group was found for the requested slice")

    available_families = manifest.get("families", [])
    if args.families.strip().lower() == "all":
        requested_families = available_families
    else:
        requested_families = [value.strip() for value in args.families.split(",") if value.strip()]

    selected_groups = [("baseline", baseline_group)]
    warnings: list[str] = []
    for family in requested_families:
        group = find_group(
            groups,
            problem=args.problem,
            comparison_mode=args.comparison_mode,
            domain=args.domain,
            budget=args.budget,
            architecture=args.architecture,
            family=family,
            variant=f"{family}_{args.variant}",
        )
        if group is None:
            warnings.append(f"missing matched group for {family}_{args.variant}")
        else:
            selected_groups.append((family, group))

    if len(selected_groups) == 1:
        raise SystemExit("No matched localiser groups were found for the requested slice")

    methods = [load_method(batch_root, group, seeds) for _, group in selected_groups]
    if args.output_dir is None:
        output_dir = batch_root / "comparisons" / "_".join(
            [args.problem, args.domain, args.budget, clean_name(args.architecture), args.comparison_mode, args.variant]
        )
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_seed_grid(methods, seeds, args, output_dir / "loss_by_seed.png", "loss")
    plot_seed_grid(methods, seeds, args, output_dir / "solution_by_seed.png", "solution")
    loss_pages = plot_method_pages(methods, seeds, args, output_dir, "loss")
    solution_pages = plot_method_pages(methods, seeds, args, output_dir, "solution")
    interactive_path = output_dir / "interactive_comparison.html"
    build_interactive_html(methods, seeds, args, interactive_path)

    output_manifest = {
        "batch_root": str(batch_root),
        "selection": {
            "problem": args.problem,
            "domain": args.domain,
            "budget": args.budget,
            "architecture": args.architecture,
            "comparison_mode": args.comparison_mode,
            "variant": args.variant,
            "seeds": seeds,
        },
        "methods": [
            {
                "family": method.family,
                "label": method.label,
                "group_relpath": method.group["group_relpath"],
                "available_seeds": sorted(set(method.loss) & set(method.solution)),
                "invalid_seeds": method.invalid_seeds,
                "missing_seeds": method.missing_seeds,
            }
            for method in methods
        ],
        "warnings": warnings,
        "artifacts": {
            "loss_by_seed": str(output_dir / "loss_by_seed.png"),
            "solution_by_seed": str(output_dir / "solution_by_seed.png"),
            "loss_by_function_pages": loss_pages,
            "solution_by_function_pages": solution_pages,
            "interactive": str(interactive_path),
        },
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"batch_root={batch_root}")
    print(f"output_dir={output_dir}")
    print(f"methods={len(methods)} seeds={len(seeds)}")
    print(f"loss_by_seed={output_dir / 'loss_by_seed.png'}")
    print(f"solution_by_seed={output_dir / 'solution_by_seed.png'}")
    print(f"interactive={interactive_path}")
    for warning in warnings:
        print(f"warning={warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
