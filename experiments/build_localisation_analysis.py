#!/usr/bin/env python3
"""Build a complete Markdown analysis ledger from one localisation batch."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PRIMARY_METRICS = (
    "final_train_loss",
    "final_train_residual_rmse",
    "residual_rmse",
    "solution_rmse",
    "solution_mae",
    "wall_clock_seconds",
)
COMPARE_METRICS = (
    "final_train_loss",
    "residual_rmse",
    "solution_rmse",
    "wall_clock_seconds",
)
METRIC_LABELS = {
    "final_train_loss": "final train loss",
    "final_train_residual_rmse": "final train residual RMSE",
    "residual_rmse": "solution residual RMSE",
    "solution_rmse": "solution RMSE",
    "solution_mae": "solution MAE",
    "wall_clock_seconds": "wall-clock seconds",
}
PROBLEM_LABELS = {"ho": "HO", "heat": "Heat", "4d": "4D"}
MODE_LABELS = {
    "function_screen": "function screen",
    "same_network": "same network",
    "mixed_size": "mixed size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build analysis.md for a localisation campaign batch.")
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


def number(value: Any) -> float | None:
    return float(value) if finite(value) else None


def fmt(value: Any, digits: int = 6) -> str:
    parsed = number(value)
    if parsed is None:
        return "n/a"
    if parsed == 0:
        return "0"
    absolute = abs(parsed)
    if absolute >= 1e4 or absolute < 1e-4:
        return f"{parsed:.{digits}e}"
    return f"{parsed:.{digits}g}"


def pct(value: Any) -> str:
    parsed = number(value)
    return "n/a" if parsed is None else f"{100.0 * parsed:.1f}%"


def md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def relative_link(batch_root: Path, path: Path) -> str:
    try:
        return path.relative_to(batch_root).as_posix()
    except ValueError:
        return path.as_posix()


def metric_values(rows: Iterable[dict[str, Any]], metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if finite(row.get(metric))]


def stats(rows: Iterable[dict[str, Any]], metric: str) -> dict[str, float | int | None]:
    values = metric_values(rows, metric)
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


def load_group_record(batch_root: Path, group_manifest: dict[str, Any]) -> dict[str, Any]:
    group_relpath = group_manifest["group_relpath"]
    aggregate_path = batch_root / group_relpath / "aggregate.json"
    aggregate = load_json(aggregate_path) or {}
    group = aggregate.get("group") or group_manifest
    rows = aggregate.get("per_seed")
    if not isinstance(rows, list):
        rows = []
    rows = [row for row in rows if isinstance(row, dict)]

    # This fallback keeps the report useful if an aggregate file is missing.
    if not rows:
        for seed in group_manifest.get("seeds", []):
            metrics = load_json(batch_root / group_relpath / f"seed_{int(seed):02d}" / "metrics.json")
            if metrics is not None:
                rows.append(metrics)

    rows.sort(key=lambda row: int(row.get("seed", 0)))
    return {
        "group": group,
        "rows": rows,
        "aggregate_path": aggregate_path,
        "exists": aggregate_path.exists(),
    }


def process_status(row: dict[str, Any]) -> str:
    status = str(row.get("status", "missing"))
    if status == "completed" and not all(finite(row.get(metric)) for metric in ("final_train_loss", "residual_rmse", "solution_rmse")):
        return "completed_nonfinite"
    return status


def row_map(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in record["rows"]:
        try:
            result[int(row["seed"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return result


def paired_rows(local_record: dict[str, Any], baseline_record: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    local = row_map(local_record)
    baseline = row_map(baseline_record)
    pairs = []
    for seed in sorted(set(local) & set(baseline)):
        local_row = local[seed]
        baseline_row = baseline[seed]
        if process_status(local_row) != "completed" or process_status(baseline_row) != "completed":
            continue
        pairs.append((local_row, baseline_row))
    return pairs


def comparison(local_record: dict[str, Any], baseline_record: dict[str, Any] | None) -> dict[str, Any] | None:
    if baseline_record is None:
        return None
    pairs = paired_rows(local_record, baseline_record)
    result: dict[str, Any] = {"paired_seeds": len(pairs), "metrics": {}}
    for metric in COMPARE_METRICS:
        metric_pairs = [(local, base) for local, base in pairs if finite(local.get(metric)) and finite(base.get(metric))]
        local_values = [float(local[metric]) for local, _ in metric_pairs]
        baseline_values = [float(base[metric]) for _, base in metric_pairs]
        ratios = [local_value / base_value for local_value, base_value in zip(local_values, baseline_values) if base_value != 0]
        deltas = [local_value - base_value for local_value, base_value in zip(local_values, baseline_values)]
        wins = sum(local_value < base_value for local_value, base_value in zip(local_values, baseline_values))
        result["metrics"][metric] = {
            "paired": len(metric_pairs),
            "wins": wins,
            "win_rate": wins / len(metric_pairs) if metric_pairs else None,
            "local_mean": statistics.fmean(local_values) if local_values else None,
            "baseline_mean": statistics.fmean(baseline_values) if baseline_values else None,
            "mean_delta": statistics.fmean(deltas) if deltas else None,
            "median_ratio": statistics.median(ratios) if ratios else None,
        }
    return result


def aggregate_comparison_rollups(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records.values():
        group = record["group"]
        family = group.get("localiser_family", "unknown")
        if family == "baseline":
            continue
        key = (
            str(group.get("problem", "unknown")),
            str(group.get("comparison_mode", "unknown")),
            family,
            str(group.get("variant", "unknown")),
        )
        bucket = buckets.setdefault(
            key,
            {
                "problem": key[0],
                "mode": key[1],
                "family": key[2],
                "variant": key[3],
                "groups": 0,
                "completed_processes": 0,
                "valid_solution_seeds": 0,
                "nonfinite_solution_seeds": 0,
                "metrics": {metric: {"paired": 0, "wins": 0, "ratios": [], "deltas": []} for metric in COMPARE_METRICS},
            },
        )
        bucket["groups"] += 1
        bucket["completed_processes"] += sum(row.get("status") == "completed" for row in record["rows"])
        bucket["valid_solution_seeds"] += sum(process_status(row) == "completed" and finite(row.get("solution_rmse")) for row in record["rows"])
        bucket["nonfinite_solution_seeds"] += sum(process_status(row) != "completed" or not finite(row.get("solution_rmse")) for row in record["rows"])
        pair_path = group.get("pair_group_relpath")
        pair_record = records.get(pair_path) if pair_path else None
        result = comparison(record, pair_record)
        if result is None:
            continue
        for metric in COMPARE_METRICS:
            source = result["metrics"][metric]
            target = bucket["metrics"][metric]
            target["paired"] += source["paired"]
            target["wins"] += source["wins"]
            if finite(source.get("median_ratio")):
                target["ratios"].append(float(source["median_ratio"]))
            if finite(source.get("mean_delta")):
                target["deltas"].append(float(source["mean_delta"]))

    output = []
    for bucket in buckets.values():
        for metric in COMPARE_METRICS:
            target = bucket["metrics"][metric]
            target["win_rate"] = target["wins"] / target["paired"] if target["paired"] else None
            target["median_group_ratio"] = statistics.median(target.pop("ratios")) if target.get("ratios") else None
            target["mean_group_delta"] = statistics.fmean(target.pop("deltas")) if target.get("deltas") else None
        output.append(bucket)
    return sorted(output, key=lambda item: (item["problem"], item["mode"], item["family"], item["variant"]))


def coverage_rollups(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for record in records.values():
        group = record["group"]
        experiment = str(group.get("experiment", "unknown"))
        bucket = buckets.setdefault(
            experiment,
            {
                "experiment": experiment,
                "problem": str(group.get("problem", "unknown")),
                "mode": str(group.get("comparison_mode", "unknown")),
                "groups": 0,
                "requested": 0,
                "process_completed": 0,
                "process_failed": 0,
                "valid_solution": 0,
                "valid_residual": 0,
                "valid_loss": 0,
                "nonfinite": 0,
            },
        )
        bucket["groups"] += 1
        bucket["requested"] += len(record["rows"])
        for row in record["rows"]:
            status = process_status(row)
            bucket["process_completed"] += status == "completed"
            bucket["process_failed"] += status == "failed"
            bucket["valid_solution"] += status == "completed" and finite(row.get("solution_rmse"))
            bucket["valid_residual"] += status == "completed" and finite(row.get("residual_rmse"))
            bucket["valid_loss"] += status == "completed" and finite(row.get("final_train_loss"))
            bucket["nonfinite"] += status == "completed_nonfinite"
    return [buckets[key] for key in sorted(buckets)]


def write_table(handle: Any, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        handle.write("| " + " | ".join(md(value) for value in row) + " |\n")


def write_metric_table(handle: Any, rows: list[dict[str, Any]]) -> None:
    output = []
    for metric in PRIMARY_METRICS:
        item = stats(rows, metric)
        output.append(
            [
                METRIC_LABELS[metric],
                item["n"],
                fmt(item["mean"]),
                fmt(item["median"]),
                fmt(item["stdev"]),
                fmt(item["min"]),
                fmt(item["max"]),
            ]
        )
    write_table(handle, ["metric", "valid n", "mean", "median", "stdev", "min", "max"], output)


def write_comparison_table(handle: Any, result: dict[str, Any] | None) -> None:
    if result is None:
        handle.write("No paired baseline for this group.\n")
        return
    output = []
    for metric in COMPARE_METRICS:
        item = result["metrics"][metric]
        output.append(
            [
                METRIC_LABELS[metric],
                f"{item['wins']}/{item['paired']} ({pct(item['win_rate'])})" if item["paired"] else "n/a",
                fmt(item["local_mean"]),
                fmt(item["baseline_mean"]),
                fmt(item["mean_delta"]),
                fmt(item["median_ratio"]),
            ]
        )
    write_table(handle, ["metric", "local wins", "local mean", "baseline mean", "mean delta", "median local/baseline"], output)


def write_seed_table(handle: Any, batch_root: Path, record: dict[str, Any], baseline_record: dict[str, Any] | None) -> None:
    baseline_rows = row_map(baseline_record) if baseline_record else {}
    headers = [
        "seed",
        "status",
        "best epoch",
        "time s",
        "train loss",
        "residual",
        "solution RMSE",
        "solution MAE",
    ]
    if baseline_record is not None:
        headers += ["base loss", "base residual", "base solution", "delta solution", "delta residual", "delta time"]
    output = []
    group = record["group"]
    group_relpath = group["group_relpath"]
    for row in record["rows"]:
        seed = int(row.get("seed", 0))
        best = row.get("best_checkpoint") if isinstance(row.get("best_checkpoint"), dict) else {}
        values: list[Any] = [
            seed,
            process_status(row),
            best.get("epoch", "n/a"),
            fmt(row.get("wall_clock_seconds")),
            fmt(row.get("final_train_loss")),
            fmt(row.get("residual_rmse")),
            fmt(row.get("solution_rmse")),
            fmt(row.get("solution_mae")),
        ]
        if baseline_record is not None:
            base = baseline_rows.get(seed, {})
            values += [
                fmt(base.get("final_train_loss")),
                fmt(base.get("residual_rmse")),
                fmt(base.get("solution_rmse")),
                fmt(number(row.get("solution_rmse")) - number(base.get("solution_rmse"))) if finite(row.get("solution_rmse")) and finite(base.get("solution_rmse")) else "n/a",
                fmt(number(row.get("residual_rmse")) - number(base.get("residual_rmse"))) if finite(row.get("residual_rmse")) and finite(base.get("residual_rmse")) else "n/a",
                fmt(number(row.get("wall_clock_seconds")) - number(base.get("wall_clock_seconds"))) if finite(row.get("wall_clock_seconds")) and finite(base.get("wall_clock_seconds")) else "n/a",
            ]
        metrics_path = batch_root / group_relpath / f"seed_{seed:02d}" / "metrics.json"
        raw_path = batch_root / group_relpath / f"seed_{seed:02d}" / "raw.log"
        values[0] = f"[{seed}]({relative_link(batch_root, metrics_path)}) / [log]({relative_link(batch_root, raw_path)})"
        output.append(values)
    write_table(handle, headers, output)


def write_rollup_sections(handle: Any, records: dict[str, dict[str, Any]], batch_root: Path) -> None:
    handle.write("## Coverage\n\n")
    coverage = coverage_rollups(records)
    write_table(
        handle,
        ["experiment", "problem", "mode", "groups", "requested seeds", "process completed", "process failed", "valid loss", "valid residual", "valid solution", "completed but nonfinite"],
        [
            [
                item["experiment"],
                PROBLEM_LABELS.get(item["problem"], item["problem"]),
                MODE_LABELS.get(item["mode"], item["mode"]),
                item["groups"],
                item["requested"],
                item["process_completed"],
                item["process_failed"],
                item["valid_loss"],
                item["valid_residual"],
                item["valid_solution"],
                item["nonfinite"],
            ]
            for item in coverage
        ],
    )

    handle.write("\n## Paired comparison rollups\n\n")
    handle.write("Each row pools all groups for one problem, comparison mode, localiser family, and variant. Wins are paired seed wins where lower is better. Ratios are medians of group-level local/baseline mean ratios.\n\n")
    rollups = aggregate_comparison_rollups(records)
    rows = []
    for item in rollups:
        solution = item["metrics"]["solution_rmse"]
        residual = item["metrics"]["residual_rmse"]
        loss = item["metrics"]["final_train_loss"]
        time_metric = item["metrics"]["wall_clock_seconds"]
        rows.append(
            [
                PROBLEM_LABELS.get(item["problem"], item["problem"]),
                MODE_LABELS.get(item["mode"], item["mode"]),
                f"{item['family']}/{item['variant']}",
                item["groups"],
                item["valid_solution_seeds"],
                item["nonfinite_solution_seeds"],
                f"{solution['wins']}/{solution['paired']} ({pct(solution['win_rate'])})",
                fmt(solution["median_group_ratio"]),
                f"{residual['wins']}/{residual['paired']} ({pct(residual['win_rate'])})",
                fmt(residual["median_group_ratio"]),
                f"{loss['wins']}/{loss['paired']} ({pct(loss['win_rate'])})",
                fmt(loss["median_group_ratio"]),
                f"{time_metric['wins']}/{time_metric['paired']} ({pct(time_metric['win_rate'])})",
                fmt(time_metric["median_group_ratio"]),
            ]
        )
    write_table(
        handle,
        ["problem", "mode", "localiser/variant", "groups", "valid solution seeds", "nonfinite solution seeds", "solution wins", "solution ratio", "residual wins", "residual ratio", "loss wins", "loss ratio", "faster", "time ratio"],
        rows,
    )

    handle.write("\n## Invalid and non-finite runs\n\n")
    invalid_buckets: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(lambda: {"groups": 0, "rows": 0, "nonfinite_loss": 0, "nonfinite_residual": 0, "nonfinite_solution": 0})
    for record in records.values():
        group = record["group"]
        key = (str(group.get("problem", "unknown")), str(group.get("comparison_mode", "unknown")), str(group.get("localiser_family", "unknown")), str(group.get("variant", "unknown")))
        bucket = invalid_buckets[key]
        group_has_invalid = False
        for row in record["rows"]:
            if not finite(row.get("final_train_loss")):
                bucket["nonfinite_loss"] += 1
                group_has_invalid = True
            if not finite(row.get("residual_rmse")):
                bucket["nonfinite_residual"] += 1
                group_has_invalid = True
            if not finite(row.get("solution_rmse")):
                bucket["nonfinite_solution"] += 1
                group_has_invalid = True
            if process_status(row) != "completed":
                bucket["rows"] += 1
        if group_has_invalid:
            bucket["groups"] += 1
    invalid_rows = []
    for key in sorted(invalid_buckets):
        bucket = invalid_buckets[key]
        if bucket["groups"]:
            invalid_rows.append([PROBLEM_LABELS.get(key[0], key[0]), MODE_LABELS.get(key[1], key[1]), f"{key[2]}/{key[3]}", bucket["groups"], bucket["rows"], bucket["nonfinite_loss"], bucket["nonfinite_residual"], bucket["nonfinite_solution"]])
    write_table(handle, ["problem", "mode", "localiser/variant", "groups affected", "non-completed rows", "nonfinite loss", "nonfinite residual", "nonfinite solution"], invalid_rows or [["none", "", "", 0, 0, 0, 0, 0]])


def write_group_details(handle: Any, batch_root: Path, records: dict[str, dict[str, Any]]) -> None:
    handle.write("\n## Complete group ledger\n\n")
    handle.write("Every manifest group is listed below. Aggregate statistics are recomputed from the seed records. A process marked `completed_nonfinite` exited but produced at least one non-finite primary metric.\n\n")
    for index, group_relpath in enumerate(sorted(records), start=1):
        record = records[group_relpath]
        group = record["group"]
        baseline_record = records.get(group.get("pair_group_relpath")) if group.get("pair_group_relpath") else None
        group_dir = batch_root / group_relpath
        valid_solution = sum(process_status(row) == "completed" and finite(row.get("solution_rmse")) for row in record["rows"])
        process_completed = sum(row.get("status") == "completed" for row in record["rows"])
        handle.write(f"### {index}. `{group_relpath}`\n\n")
        handle.write(
            f"Problem: **{PROBLEM_LABELS.get(group.get('problem'), group.get('problem', 'unknown'))}** | "
            f"Mode: **{MODE_LABELS.get(group.get('comparison_mode'), group.get('comparison_mode', 'unknown'))}** | "
            f"Family: **{group.get('localiser_family', 'unknown')}** | Variant: **{group.get('variant', 'unknown')}**  \n"
            f"Domain: `{group.get('domain_label', 'n/a')}` | Budget: `{group.get('budget_label', 'n/a')}` ({group.get('epochs', 'n/a')} epochs) | "
            f"Architecture: `{group.get('architecture_label', 'n/a')}` | Process completed: `{process_completed}/{len(record['rows'])}` | "
            f"Valid solution metrics: `{valid_solution}/{len(record['rows'])}`\n\n"
        )
        handle.write("#### Aggregate metrics\n\n")
        write_metric_table(handle, record["rows"])
        handle.write("\n#### Paired baseline comparison\n\n")
        write_comparison_table(handle, comparison(record, baseline_record))
        handle.write("\n#### Artifacts\n\n")
        artifact_rows = [
            [f"[aggregate.json]({relative_link(batch_root, record['aggregate_path'])})", f"[aggregate.csv]({relative_link(batch_root, group_dir / 'aggregate.csv')})"],
        ]
        for filename in ("combined_loss_plot.png", "combined_solution_plot.png", "summary.md"):
            path = group_dir / filename
            if path.exists():
                artifact_rows.append([f"[{filename}]({relative_link(batch_root, path)})", ""])
        write_table(handle, ["primary artifact", "additional artifact"], artifact_rows)
        handle.write("\n#### Per-seed ledger\n\n")
        write_seed_table(handle, batch_root, record, baseline_record)
        handle.write("\n")


def build_report(batch_root: Path, output_path: Path) -> None:
    manifest_path = batch_root / "run_manifest.json"
    manifest = load_json(manifest_path)
    if manifest is None:
        raise SystemExit(f"Missing or invalid manifest: {manifest_path}")
    group_manifests = [item for item in manifest.get("groups", []) if isinstance(item, dict) and item.get("group_relpath")]
    records = {item["group_relpath"]: load_group_record(batch_root, item) for item in group_manifests}
    summary = load_json(batch_root / "summary.json") or {}
    counts = summary.get("counts", {})
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# LPINNs Campaign Analysis\n\n")
        handle.write(f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`  \nBatch root: `{batch_root}`  \n")
        handle.write(f"Manifest groups: `{len(records)}` | Manifest seed records: `{sum(len(record['rows']) for record in records.values())}`\n\n")
        handle.write("## Scope and interpretation\n\n")
        handle.write("This report is generated from `run_manifest.json`, every group `aggregate.json`, and every per-seed `metrics.json`. It contains the complete group ledger and per-seed ledger, recomputed means/medians/standard deviations/minima/maxima, paired baseline deltas, paired seed win counts, timing comparisons, artifact links, and non-finite-run accounting.\n\n")
        handle.write("Lower is better for all reported comparison metrics. `solution_rmse` measures recovered-solution accuracy; `residual_rmse` measures PDE residual; `final_train_loss` is the final training loss; wall-clock comparisons are localised versus paired baseline. These are not interchangeable metrics.\n\n")
        handle.write("The runner can mark a process `completed` even if training produced NaN values. This report reclassifies such rows as `completed_nonfinite` for validity accounting while preserving their raw status and links.\n\n")
        handle.write("## Batch totals\n\n")
        write_table(
            handle,
            ["field", "value"],
            [
                ["requested seed runs", counts.get("requested_seed_runs", "n/a")],
                ["process-completed seed runs", counts.get("completed_seed_runs", "n/a")],
                ["process-failed seed runs", counts.get("failed_seed_runs", "n/a")],
                ["manifest groups", len(records)],
                ["effective max workers", manifest.get("effective_max_workers", "n/a")],
                ["experiments", ", ".join(manifest.get("experiments", []))],
                ["localiser families", ", ".join(manifest.get("families", []))],
            ],
        )
        write_rollup_sections(handle, records, batch_root)
        write_group_details(handle, batch_root, records)


def main() -> int:
    args = parse_args()
    batch_root = args.batch_root.expanduser().resolve()
    output = (args.output or batch_root / "analysis.md").expanduser().resolve()
    build_report(batch_root, output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
