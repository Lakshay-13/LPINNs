#!/usr/bin/env python3
"""Build a per-run wall-clock timing study from a localisation batch."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PRIMARY_METRICS = ("final_train_loss", "residual_rmse", "solution_rmse")
PROBLEM_ORDER = {"ho": 0, "heat": 1, "4d": 2}
MODE_ORDER = {"function_screen": 0, "same_network": 1, "mixed_size": 2}
FAMILY_ORDER = {
    "baseline": 0,
    "gaussian": 1,
    "activated": 2,
    "super_gaussian": 3,
    "ricker": 4,
    "boxcar": 5,
    "laplace": 6,
    "cauchy": 7,
    "raised_cosine": 8,
    "bump": 9,
    "gabor": 10,
    "morlet": 11,
    "inverse_quadratic": 12,
    "triangular": 13,
}
VARIANT_ORDER = {"baseline": 0, "fixed": 1, "sigma": 2, "mu_sigma": 3}
PROBLEM_LABELS = {"ho": "HO", "heat": "Heat", "4d": "4D"}
MODE_LABELS = {
    "function_screen": "function screen",
    "same_network": "same network",
    "mixed_size": "mixed size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build time_study.md for a localisation campaign batch.")
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: Any, digits: int = 6) -> str:
    if not finite(value):
        return "n/a"
    parsed = float(value)
    if parsed == 0:
        return "0"
    if abs(parsed) >= 1e4 or abs(parsed) < 1e-4:
        return f"{parsed:.{digits}e}"
    return f"{parsed:.{digits}g}"


def md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def relative_link(batch_root: Path, path: Path) -> str:
    try:
        return path.relative_to(batch_root).as_posix()
    except ValueError:
        return path.as_posix()


def write_table(handle: Any, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        handle.write("| " + " | ".join(md(value) for value in row) + " |\n")


def group_record(batch_root: Path, manifest_group: dict[str, Any]) -> dict[str, Any]:
    group_relpath = manifest_group["group_relpath"]
    aggregate_path = batch_root / group_relpath / "aggregate.json"
    aggregate = load_json(aggregate_path) or {}
    group = aggregate.get("group") or manifest_group
    rows = aggregate.get("per_seed")
    if not isinstance(rows, list):
        rows = []
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        for seed in manifest_group.get("seeds", []):
            path = batch_root / group_relpath / f"seed_{int(seed):02d}" / "metrics.json"
            row = load_json(path)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: int(row.get("seed", 0)))
    return {"group": group, "rows": rows, "aggregate_path": aggregate_path}


def timing_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["wall_clock_seconds"]) for row in rows if row.get("status") == "completed" and finite(row.get("wall_clock_seconds"))]
    if not values:
        return {"n": 0, "mean": None, "median": None, "stdev": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def metric_valid(row: dict[str, Any]) -> bool:
    return row.get("status") == "completed" and all(finite(row.get(metric)) for metric in PRIMARY_METRICS)


def variant_kind(group: dict[str, Any]) -> str:
    family = str(group.get("localiser_family", ""))
    variant = str(group.get("variant", ""))
    if family == "baseline":
        return "baseline"
    prefix = f"{family}_"
    return variant[len(prefix):] if variant.startswith(prefix) else variant


def group_key(record: dict[str, Any]) -> tuple[Any, ...]:
    group = record["group"]
    return (
        PROBLEM_ORDER.get(str(group.get("problem")), 99),
        MODE_ORDER.get(str(group.get("comparison_mode")), 99),
        str(group.get("experiment", "")),
        str(group.get("architecture_label", "")),
        float(group.get("domain_value", 0.0)),
        int(group.get("epochs", 0)),
        FAMILY_ORDER.get(str(group.get("localiser_family")), 99),
        VARIANT_ORDER.get(variant_kind(group), 99),
        str(group.get("group_relpath", "")),
    )


def variant_label(group: dict[str, Any]) -> str:
    family = str(group.get("localiser_family", "unknown"))
    variant = str(group.get("variant", "unknown"))
    return "baseline_fc" if family == "baseline" else f"{family}/{variant}"


def comparison(record: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    local_rows = {int(row["seed"]): row for row in record["rows"] if "seed" in row}
    base_rows = {int(row["seed"]): row for row in baseline["rows"] if "seed" in row}
    pairs = []
    for seed in sorted(set(local_rows) & set(base_rows)):
        local = local_rows[seed]
        base = base_rows[seed]
        if local.get("status") == "completed" and base.get("status") == "completed":
            if finite(local.get("wall_clock_seconds")) and finite(base.get("wall_clock_seconds")):
                pairs.append((local, base))
    local_values = [float(local["wall_clock_seconds"]) for local, _ in pairs]
    base_values = [float(base["wall_clock_seconds"]) for _, base in pairs]
    if not pairs:
        return {"paired": 0, "local_mean": None, "baseline_mean": None, "delta": None, "ratio": None}
    return {
        "paired": len(pairs),
        "local_mean": statistics.fmean(local_values),
        "baseline_mean": statistics.fmean(base_values),
        "delta": statistics.fmean(local - base for local, base in zip(local_values, base_values)),
        "ratio": statistics.median(local / base for local, base in zip(local_values, base_values) if base != 0),
    }


def summary_rows(records: dict[str, dict[str, Any]]) -> list[list[Any]]:
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records.values():
        by_experiment[str(record["group"].get("experiment", "unknown"))].append(record)
    rows = []
    for experiment in sorted(by_experiment):
        group_records = by_experiment[experiment]
        all_rows = [row for record in group_records for row in record["rows"]]
        timing = timing_stats(all_rows)
        valid_metrics = sum(metric_valid(row) for row in all_rows)
        nonfinite = sum(row.get("status") == "completed" and not metric_valid(row) for row in all_rows)
        rows.append(
            [
                experiment,
                PROBLEM_LABELS.get(str(group_records[0]["group"].get("problem")), "unknown"),
                MODE_LABELS.get(str(group_records[0]["group"].get("comparison_mode")), "unknown"),
                len(group_records),
                len(all_rows),
                timing["n"],
                valid_metrics,
                nonfinite,
                fmt(timing["mean"]),
                fmt(timing["median"]),
            ]
        )
    return rows


def write_report(batch_root: Path, output_path: Path) -> None:
    manifest = load_json(batch_root / "run_manifest.json")
    if manifest is None:
        raise SystemExit(f"Missing or invalid manifest: {batch_root / 'run_manifest.json'}")
    manifest_groups = [item for item in manifest.get("groups", []) if isinstance(item, dict) and item.get("group_relpath")]
    records = {item["group_relpath"]: group_record(batch_root, item) for item in manifest_groups}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# LPINNs Time Study\n\n")
        handle.write(f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`  \nBatch root: `{batch_root}`  \n")
        handle.write(f"Groups: `{len(records)}` | Seed rows: `{sum(len(record['rows']) for record in records.values())}` | Max workers: `{manifest.get('effective_max_workers', 'n/a')}`\n\n")
        handle.write("## Method\n\n")
        handle.write("Mean time is recomputed from finite per-seed `wall_clock_seconds` values in each group. Each detailed row is one run configuration, normally aggregated over 10 paired seeds. Localised rows include their paired baseline mean, mean time delta, and median per-seed local/baseline time ratio. Timing is retained even when a run's solution metrics are non-finite, so invalid numerical runs remain visible rather than disappearing from the runtime study.\n\n")
        handle.write("All times are seconds. Lower is faster. `valid metric seeds` requires finite train loss, residual RMSE, and solution RMSE; it is separate from timing validity.\n\n")

        handle.write("## Experiment timing summary\n\n")
        write_table(
            handle,
            ["experiment", "problem", "mode", "groups", "seed rows", "timing n", "valid metric seeds", "completed nonfinite", "mean seed s", "median seed s"],
            summary_rows(records),
        )

        handle.write("\n## Detailed timing tables\n\n")
        handle.write("Tables are ordered by equation, experiment mode, architecture, domain, budget, and then localiser. Within each domain, all functions for `1k` appear before `3k`, followed by the longer budgets.\n\n")
        baseline_records = {key: record for key, record in records.items() if record["group"].get("localiser_family") == "baseline"}
        sorted_records = sorted(records.values(), key=group_key)
        sections: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in sorted_records:
            group = record["group"]
            key = (
                str(group.get("experiment", "unknown")),
                str(group.get("problem", "unknown")),
                str(group.get("comparison_mode", "unknown")),
                str(group.get("architecture_label", "unknown")),
            )
            sections[key].append(record)

        for (experiment, problem, mode, architecture), section_records in sections.items():
            handle.write(f"### {PROBLEM_LABELS.get(problem, problem)} | {MODE_LABELS.get(mode, mode)} | `{architecture}`\n\n")
            domains: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in section_records:
                domains[str(record["group"].get("domain_label", "unknown"))].append(record)
            ordered_domains = sorted(domains.items(), key=lambda item: float(item[1][0]["group"].get("domain_value", 0.0)))
            for domain_label, domain_records in ordered_domains:
                handle.write(f"#### Domain `{domain_label}`\n\n")
                output_rows = []
                for record in sorted(domain_records, key=group_key):
                    group = record["group"]
                    timing = timing_stats(record["rows"])
                    baseline = records.get(group.get("pair_group_relpath")) if group.get("pair_group_relpath") else None
                    baseline_timing = timing_stats(baseline["rows"]) if baseline else None
                    paired = comparison(record, baseline)
                    valid_metric_seeds = sum(metric_valid(row) for row in record["rows"])
                    nonfinite_metric_seeds = sum(row.get("status") == "completed" and not metric_valid(row) for row in record["rows"])
                    aggregate_path = record["aggregate_path"]
                    output_rows.append(
                        [
                            group.get("budget_label", "n/a"),
                            group.get("epochs", "n/a"),
                            variant_label(group),
                            timing["n"],
                            valid_metric_seeds,
                            nonfinite_metric_seeds,
                            fmt(timing["mean"]),
                            fmt(timing["median"]),
                            fmt(timing["stdev"]),
                            fmt(baseline_timing["mean"] if baseline_timing else None),
                            fmt(paired["delta"] if paired else None),
                            fmt(paired["ratio"] if paired else None),
                            f"[{relative_link(batch_root, aggregate_path)}]({relative_link(batch_root, aggregate_path)})",
                        ]
                    )
                write_table(
                    handle,
                    ["budget", "epochs", "run", "timing n", "valid metric seeds", "nonfinite metric seeds", "mean s", "median s", "stdev s", "baseline mean s", "delta s", "median ratio", "aggregate"],
                    output_rows,
                )
                handle.write("\n")


def main() -> int:
    args = parse_args()
    batch_root = args.batch_root.expanduser().resolve()
    output = (args.output or batch_root / "time_study.md").expanduser().resolve()
    write_report(batch_root, output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
