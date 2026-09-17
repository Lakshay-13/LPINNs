#!/usr/bin/env python3
"""Serve the complete localisation gallery and its lazy curve API."""

from __future__ import annotations

import argparse
import json
import math
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


COLORS = [
    "#1b4965", "#ca6702", "#2a9d8f", "#9b2226", "#6a4c93", "#588157",
    "#bc6c25", "#0077b6", "#c9184a", "#386641", "#7f5539", "#5a189a",
    "#0a9396", "#ae2012",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interactive-points", type=int, default=1200)
    return parser.parse_args()


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


def downsample(values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= max_points:
        return np.arange(len(values), dtype=np.int64), values
    indices = np.unique(np.linspace(0, len(values) - 1, max_points, dtype=np.int64))
    return indices, values[indices]


def json_values(values: np.ndarray) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def group_curve(batch_root: Path, group_relpath: str, seed: int, points: int) -> tuple[dict, dict, bool]:
    seed_dir = batch_root / group_relpath / f"seed_{seed:02d}"
    loss_path = seed_dir / "loss_history.npy"
    solution_path = seed_dir / "solution_slice.npz"
    metrics_path = seed_dir / "metrics.json"
    if not loss_path.exists() or not solution_path.exists():
        return {"seed": seed, "x": [], "y": []}, {"seed": seed, "x": [], "y": []}, True
    history = np.asarray(np.load(loss_path), dtype=np.float64).reshape(-1)
    loss_indices, loss_values = downsample(history, points)
    loss_y = np.full(loss_values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(loss_values) & (loss_values > 0)
    loss_y[finite] = np.log10(loss_values[finite])
    bundle = np.load(solution_path, allow_pickle=True)
    curve_x = np.asarray(bundle["curve_x"], dtype=np.float64).reshape(-1)
    prediction = np.asarray(bundle["prediction_curve"], dtype=np.float64).reshape(-1)
    solution_indices, solution_values = downsample(prediction, points)
    metrics_status = "completed"
    if metrics_path.exists():
        try:
            metrics_status = json.loads(metrics_path.read_text()).get("status", "completed")
        except json.JSONDecodeError:
            metrics_status = "unknown"
    invalid = metrics_status != "completed" or not np.isfinite(history).all() or not np.isfinite(prediction).all()
    return (
        {"seed": seed, "x": json_values(loss_indices.astype(np.float64) + 1.0), "y": json_values(loss_y)},
        {"seed": seed, "x": json_values(curve_x[solution_indices]), "y": json_values(solution_values)},
        invalid,
    )


def load_data(gallery_root: Path, record: dict, points: int) -> dict:
    batch_root = Path(json.loads((gallery_root / "gallery_summary.json").read_text())["batch_root"])
    methods = []
    reference = None
    seeds = list(range(1, 11))
    for method_index, (family, group_relpath) in enumerate(record["methods"].items()):
        loss_runs = []
        solution_runs = []
        invalid_seeds = []
        for seed in seeds:
            loss_run, solution_run, invalid = group_curve(batch_root, group_relpath, seed, points)
            if loss_run["x"]:
                loss_runs.append(loss_run)
            if solution_run["x"]:
                solution_runs.append(solution_run)
            if invalid:
                invalid_seeds.append(seed)
        loss_arrays = [np.asarray(run["y"], dtype=np.float64) for run in loss_runs]
        solution_arrays = [np.asarray(run["y"], dtype=np.float64) for run in solution_runs]
        loss_mean = finite_mean(loss_arrays)
        solution_mean = finite_mean(solution_arrays)
        mean_loss = None
        mean_solution = None
        if loss_mean is not None and loss_runs:
            mean_loss = {"x": loss_runs[0]["x"][: len(loss_mean)], "y": json_values(loss_mean)}
        if solution_mean is not None and solution_runs:
            mean_solution = {"x": solution_runs[0]["x"][: len(solution_mean)], "y": json_values(solution_mean)}
        methods.append(
            {
                "family": family,
                "label": "Baseline" if family == "baseline" else family.replace("_", " ").title(),
                "color": COLORS[method_index % len(COLORS)],
                "invalid_seeds": invalid_seeds,
                "loss": {"runs": loss_runs, "mean": mean_loss},
                "solution": {"runs": solution_runs, "mean": mean_solution},
            }
        )
        if reference is None and solution_runs:
            solution_path = batch_root / group_relpath / "seed_01" / "solution_slice.npz"
            if solution_path.exists():
                bundle = np.load(solution_path, allow_pickle=True)
                curve_x = np.asarray(bundle["curve_x"], dtype=np.float64).reshape(-1)
                curve_reference = np.asarray(bundle["reference_curve"], dtype=np.float64).reshape(-1)
                indices, _ = downsample(curve_reference, points)
                reference = {"x": json_values(curve_x[indices]), "y": json_values(curve_reference[indices])}
    return {
        "selection": {key: record[key] for key in (
            "problem", "domain_label", "domain_value", "budget_label", "epochs",
            "comparison_mode", "architecture_label", "baseline_architecture_label",
            "localised_architecture_label", "variant",
        )},
        "methods": methods,
        "reference": reference,
    }


class GalleryHandler(SimpleHTTPRequestHandler):
    gallery_root: Path
    records: dict[str, dict]
    points: int

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/index":
            self.send_json({"slices": list(self.records.values())})
            return
        if parsed.path == "/api/data":
            query = parse_qs(parsed.query)
            slice_id = query.get("id", [""])[0]
            record = self.records.get(slice_id)
            if record is None:
                self.send_error(404, "Unknown comparison slice")
                return
            self.send_json(load_data(self.gallery_root, record, self.points))
            return
        super().do_GET()

    def send_json(self, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        print(f"gallery_server {self.address_string()} {format % args}", flush=True)


def main() -> int:
    args = parse_args()
    gallery_root = args.gallery_root.resolve()
    index = json.loads((gallery_root / "gallery_index.json").read_text())
    GalleryHandler.gallery_root = gallery_root
    GalleryHandler.records = {record["id"]: record for record in index["slices"]}
    GalleryHandler.points = args.interactive_points
    os.chdir(gallery_root)
    server = ThreadingHTTPServer((args.host, args.port), GalleryHandler)
    print(f"gallery_url=http://{args.host}:{args.port}/interactive_comparison.html", flush=True)
    print(f"slice_count={len(GalleryHandler.records)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
