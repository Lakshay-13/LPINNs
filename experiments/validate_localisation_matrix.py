#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import importlib.util
import inspect
import json
import math
import os
import statistics
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")


ROOT = Path(__file__).resolve().parents[1]
LOCALISATION_MODULE_PATHS = {
    "gaussian": ROOT / "HO/gaussian/Single Layer/harmonic_oscillator.py",
    "activated": ROOT / "HO/activated localization/harmonic_oscillator.py",
    "super_gaussian": ROOT / "HO/super gaussian/singe layer/harmonic_oscillator.py",
    "ricker": ROOT / "HO/ricker wavelet/harmonic_oscillator.py",
}
DEFAULT_ACTIVATIONS = {
    "gaussian": "sine",
    "activated": "sine",
    "super_gaussian": "silu",
    "ricker": "silu",
}
FAMILY_MODE_OPTIONS = {
    "gaussian": [
        ("gaussian_fixed", True, False, False, "Gaussian localisation with fixed mu/sigma"),
        ("gaussian_sigma", True, False, True, "Gaussian localisation with learnable sigma only"),
        ("gaussian_mu_sigma", True, True, True, "Gaussian localisation with learnable mu and sigma"),
    ],
    "activated": [
        ("activated_fixed", True, False, False, "Activated localisation with fixed mu/sigma"),
        ("activated_sigma", True, False, True, "Activated localisation with learnable sigma only"),
        ("activated_mu_sigma", True, True, True, "Activated localisation with learnable mu and sigma"),
    ],
    "super_gaussian": [
        ("super_gaussian_fixed", True, False, False, "Super-Gaussian localisation with fixed mu/sigma"),
        ("super_gaussian_sigma", True, False, True, "Super-Gaussian localisation with learnable sigma only"),
        ("super_gaussian_mu_sigma", True, True, True, "Super-Gaussian localisation with learnable mu and sigma"),
    ],
    "ricker": [
        ("ricker_fixed", True, False, False, "Ricker-wavelet localisation with fixed sigma"),
        ("ricker_sigma", True, False, True, "Ricker-wavelet localisation with learnable sigma"),
    ],
}


@dataclass(frozen=True)
class ValidationSpec:
    key: str
    base_key: str
    mode: str
    localization: bool
    mu_learnable: bool
    sigma_learnable: bool
    domain_span: float
    domain_label: str
    epochs: int
    duration_label: str
    seed: int
    input_neurons: tuple[int, ...]
    lr: float
    localisation_family: str
    activation_name: str
    initial_sigma: float | None
    sigma_min: float | None
    sigma_max: float | None
    train_batch_size: int
    valid_batch_size: int
    scaling: bool
    note: str = ""


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Gaussian localisation on HO across domains and run lengths.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "auto"], default="auto")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--short-epochs", type=int, default=500)
    parser.add_argument("--long-epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--input-neurons", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--baseline-input-neurons", nargs="+", type=int, default=None)
    parser.add_argument(
        "--localisation-family",
        choices=sorted(LOCALISATION_MODULE_PATHS),
        default="gaussian",
    )
    parser.add_argument(
        "--activation",
        choices=["auto", "sine", "silu", "tanh"],
        default="auto",
    )
    parser.add_argument("--initial-sigma", type=str, default="0.1")
    parser.add_argument("--sigma-min", type=str, default="none")
    parser.add_argument("--sigma-max", type=str, default="none")
    parser.add_argument("--milestones", nargs="+", type=int, default=None)
    parser.add_argument("--target-solution-rmse", type=float, default=None)
    parser.add_argument("--target-residual-rmse", type=float, default=None)
    parser.add_argument("--lr-schedule", nargs="+", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--experiments", nargs="*", default=None)
    return parser.parse_args()


def resolve_activation_name(requested: str, family: str):
    if requested != "auto":
        return requested
    return DEFAULT_ACTIVATIONS[family]


def load_localisation_module(family: str):
    module_path = LOCALISATION_MODULE_PATHS[family]
    module_name = f"localisation_{family}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_output_root(output_root: Path | None) -> Path:
    if output_root is not None:
        root = output_root
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = ROOT / "repro_runs" / "gaussian_validation" / stamp
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path, payload: dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def save_loss_plot(train_loss: list[float], out_path: Path, title: str):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.log10(np.maximum(np.array(train_loss), 1e-30)))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log10(train_loss)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def configure_torch(seed: int):
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    thread_count = os.environ.get("LOCALISATION_TORCH_THREADS")
    if thread_count:
        torch.set_num_threads(max(1, int(thread_count)))
    try:
        torch.set_default_device("cpu")
    except Exception:
        pass


def resolve_seeds(args):
    if args.seeds:
        seen = set()
        ordered = []
        for seed in args.seeds:
            if seed in seen:
                continue
            seen.add(seed)
            ordered.append(seed)
        return ordered
    if args.seed is not None:
        return [args.seed]
    return list(range(1, 11))


def parse_optional_float(value: str | None):
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return float(value)


def build_specs(
    seeds: list[int],
    short_epochs: int,
    long_epochs: int,
    input_neurons: tuple[int, ...],
    baseline_input_neurons: tuple[int, ...] | None,
    localisation_family: str,
    activation_name: str,
    initial_sigma: float | None,
    sigma_min: float | None,
    sigma_max: float | None,
    lr: float,
):
    domain_options = [
        ("small", math.pi),
        ("large", 4 * math.pi),
    ]
    duration_options = [
        ("short", short_epochs),
        ("long", long_epochs),
    ]
    mode_options = [
        ("baseline", False, False, False, "No localisation; FCNN baseline"),
        *FAMILY_MODE_OPTIONS[localisation_family],
    ]

    specs: list[ValidationSpec] = []
    for seed in seeds:
        for mode_name, localization, mu_learnable, sigma_learnable, note in mode_options:
            for domain_label, domain_span in domain_options:
                for duration_label, epochs in duration_options:
                    base_key = f"{mode_name}__{domain_label}__{duration_label}"
                    spec_input_neurons = (
                        input_neurons if localization else (baseline_input_neurons or input_neurons)
                    )
                    specs.append(
                        ValidationSpec(
                            key=f"{base_key}__seed_{seed:04d}",
                            base_key=base_key,
                            mode=mode_name,
                            localization=localization,
                            mu_learnable=mu_learnable,
                            sigma_learnable=sigma_learnable,
                            domain_span=domain_span,
                            domain_label=domain_label,
                            epochs=epochs,
                            duration_label=duration_label,
                            seed=seed,
                            input_neurons=spec_input_neurons,
                            lr=lr,
                            localisation_family=localisation_family,
                            activation_name=activation_name,
                            initial_sigma=initial_sigma,
                            train_batch_size=256,
                            valid_batch_size=64,
                            scaling=True,
                            sigma_min=sigma_min,
                            sigma_max=sigma_max,
                            note=note,
                        )
                    )
    return specs


def choose_specs(specs: list[ValidationSpec], selected: list[str] | None):
    if not selected:
        return specs
    selected_set = set(selected)
    chosen_keys: set[str] = set()
    baseline_pairs: set[tuple[int, str, str]] = set()
    for spec in specs:
        if spec.key in selected_set or spec.base_key in selected_set:
            chosen_keys.add(spec.key)
            if spec.mode != "baseline":
                baseline_pairs.add((spec.seed, spec.domain_label, spec.duration_label))
    for spec in specs:
        if spec.mode == "baseline" and (spec.seed, spec.domain_label, spec.duration_label) in baseline_pairs:
            chosen_keys.add(spec.key)
    return [spec for spec in specs if spec.key in chosen_keys]


def build_solver(spec: ValidationSpec):
    import torch
    import torch.nn as nn
    from neurodiffeq import diff
    from neurodiffeq.conditions import IVP
    from neurodiffeq.generators import Generator1D
    from neurodiffeq.networks import FCNN
    from neurodiffeq.solvers import Solver1D

    module = load_localisation_module(spec.localisation_family)
    CustomNN = module.CustomNN

    class SineActivation(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    def activation_factory(name: str):
        if name == "sine":
            return SineActivation
        if name == "silu":
            return nn.SiLU
        if name == "tanh":
            return nn.Tanh
        raise ValueError(f"Unsupported activation: {name}")

    actv = activation_factory(spec.activation_name)

    if spec.scaling:
        t_min = 0.0
        t_max = 1.0
        domain_span = spec.domain_span
        derivative_0 = domain_span
        coeff = domain_span**2

        def ode_system(u, x):
            return [diff(u, x, order=2) + coeff * torch.sin(u)]

    else:
        t_min = 0.0
        t_max = spec.domain_span
        derivative_0 = 1.0

        def ode_system(u, x):
            return [diff(u, x, order=2) + torch.sin(u)]

    conditions = [IVP(t_0=t_min, u_0=0.0, u_0_prime=derivative_0)]

    if spec.localization:
        custom_kwargs = {
            "n_input_units": 1,
            "n_output_units": 1,
            "hidden_units": list(spec.input_neurons),
            "actv": actv,
            "t_min": t_min,
            "t_max": t_max,
            "mu_learnable": spec.mu_learnable,
            "sigma_learnable": spec.sigma_learnable,
            "initial_sigma": spec.initial_sigma,
            "sigma_min": spec.sigma_min,
            "sigma_max": spec.sigma_max,
        }
        custom_signature = inspect.signature(CustomNN)
        filtered_kwargs = {
            key: value
            for key, value in custom_kwargs.items()
            if key in custom_signature.parameters
        }
        nets = [
            CustomNN(**filtered_kwargs)
        ]
    else:
        nets = [
            FCNN(
                n_input_units=1,
                n_output_units=1,
                hidden_units=tuple(spec.input_neurons),
                actv=actv,
            )
        ]

    train_g = Generator1D(spec.train_batch_size, t_min, t_max, method="equally-spaced-noisy")
    valid_g = Generator1D(spec.valid_batch_size, t_min, t_max, method="uniform")
    optimizer = torch.optim.Adam([p for net in nets for p in net.parameters()], lr=spec.lr)

    return Solver1D(
        ode_system=ode_system,
        conditions=conditions,
        t_min=t_min,
        t_max=t_max,
        n_batches_valid=0,
        train_generator=train_g,
        valid_generator=valid_g,
        nets=nets,
        optimizer=optimizer,
    )


def numerical_solution(domain_span: float, num_points: int):
    import numpy as np
    from scipy.integrate import solve_ivp

    def system(t, y):
        return (y[1], -math.sin(y[0]))

    t_vals = np.linspace(0.0, domain_span, num_points)
    sol = solve_ivp(system, (0.0, domain_span), (0.0, 1.0), t_eval=t_vals, rtol=1e-10, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"Reference solve_ivp failed: {sol.message}")
    return t_vals, sol.y[0]


def compute_solution_metrics(spec: ValidationSpec, solver):
    import matplotlib.pyplot as plt
    import numpy as np

    x_vals = np.linspace(solver.t_min, solver.t_max, 1000)
    pred = solver.get_solution()(x_vals, to_numpy=True)
    residuals = solver.get_residuals(x_vals, to_numpy=True)

    t_truth, truth = numerical_solution(spec.domain_span, len(x_vals))
    if spec.scaling:
        t_plot = t_truth
    else:
        t_plot = x_vals

    error = pred - truth
    return {
        "x_vals": x_vals,
        "t_plot": t_plot,
        "truth": truth,
        "prediction": pred,
        "residuals": residuals,
        "solution_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "solution_mae": float(np.mean(np.abs(error))),
        "solution_abs_max": float(np.max(np.abs(error))),
        "residual_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "residual_abs_max": float(np.max(np.abs(residuals))),
    }


def save_solution_artifacts(spec: ValidationSpec, solver, exp_dir: Path):
    import matplotlib.pyplot as plt

    metrics = compute_solution_metrics(spec, solver)
    t_plot = metrics["t_plot"]
    truth = metrics["truth"]
    pred = metrics["prediction"]
    residuals = metrics["residuals"]
    x_vals = metrics["x_vals"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_plot, truth, label="reference")
    ax.plot(t_plot, pred, label="prediction", alpha=0.85)
    ax.set_title(spec.key)
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(exp_dir / "solution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_plot, pred - truth)
    ax.set_title("Prediction Error")
    ax.set_xlabel("t")
    ax.set_ylabel("pred - truth")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(exp_dir / "error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_vals, residuals)
    ax.set_title("Residuals")
    ax.set_xlabel("x")
    ax.set_ylabel("Residual")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(exp_dir / "residuals.png", dpi=180)
    plt.close(fig)

    return {
        key: value
        for key, value in metrics.items()
        if key not in {"x_vals", "t_plot", "truth", "prediction", "residuals"}
    }


def resolve_milestones(spec: ValidationSpec, milestones: list[int] | None):
    if not milestones:
        return None
    normalized = sorted({int(value) for value in milestones if int(value) > 0 and int(value) <= spec.epochs})
    if spec.epochs not in normalized:
        normalized.append(spec.epochs)
    return normalized


def find_convergence(milestone_rows: list[dict[str, Any]], target_solution_rmse: float | None, target_residual_rmse: float | None):
    if target_solution_rmse is None and target_residual_rmse is None:
        return None
    for row in milestone_rows:
        if target_solution_rmse is not None and row["solution_rmse"] > target_solution_rmse:
            continue
        if target_residual_rmse is not None and row["residual_rmse"] > target_residual_rmse:
            continue
        return {
            "epoch": row["epoch"],
            "elapsed_sec": row["elapsed_sec"],
            "solution_rmse": row["solution_rmse"],
            "residual_rmse": row["residual_rmse"],
        }
    return None


def find_milestone_hit(
    milestone_rows: list[dict[str, Any]],
    target_solution_rmse: float,
    target_residual_rmse: float,
):
    for row in milestone_rows:
        if row["solution_rmse"] <= target_solution_rmse and row["residual_rmse"] <= target_residual_rmse:
            return {
                "epoch": row["epoch"],
                "elapsed_sec": row["elapsed_sec"],
                "solution_rmse": row["solution_rmse"],
                "residual_rmse": row["residual_rmse"],
        }
    return None


def find_snapshot_within_elapsed(row: dict[str, Any], elapsed_budget_sec: float):
    snapshot = None
    for milestone in row.get("milestones", []):
        if milestone["elapsed_sec"] <= elapsed_budget_sec:
            snapshot = {
                "epoch": milestone["epoch"],
                "elapsed_sec": milestone["elapsed_sec"],
                "solution_rmse": milestone["solution_rmse"],
                "residual_rmse": milestone["residual_rmse"],
            }
        else:
            break
    if row.get("elapsed_sec") is not None and row["elapsed_sec"] <= elapsed_budget_sec:
        return {
            "epoch": row["epochs_run"],
            "elapsed_sec": row["elapsed_sec"],
            "solution_rmse": row["solution_rmse"],
            "residual_rmse": row["residual_rmse"],
        }
    return snapshot


def parse_lr_schedule(values: list[str] | None):
    if not values:
        return []
    schedule = []
    for value in values:
        epoch_text, lr_text = value.split(":", 1)
        schedule.append((int(epoch_text), float(lr_text)))
    return sorted(schedule)


def apply_scheduled_lr(optimizer, completed_epochs: int, base_lr: float, lr_schedule: list[tuple[int, float]]):
    current_lr = base_lr
    for epoch, lr in lr_schedule:
        if completed_epochs >= epoch:
            current_lr = lr
        else:
            break
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    return current_lr


def run_spec(
    spec: ValidationSpec,
    output_root: Path,
    milestones: list[int] | None = None,
    target_solution_rmse: float | None = None,
    target_residual_rmse: float | None = None,
    lr_schedule: list[tuple[int, float]] | None = None,
):
    exp_dir = output_root / "artifacts" / spec.key
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "logs" / f"{spec.key}.log"

    summary: dict[str, Any] = {
        "key": spec.key,
        "base_key": spec.base_key,
        "mode": spec.mode,
        "localization": spec.localization,
        "mu_learnable": spec.mu_learnable,
        "sigma_learnable": spec.sigma_learnable,
        "domain_span": spec.domain_span,
        "domain_label": spec.domain_label,
        "epochs_run": spec.epochs,
        "duration_label": spec.duration_label,
        "seed": spec.seed,
        "scaling": spec.scaling,
        "note": spec.note,
        "status": "failed",
        "log_path": str(log_path.relative_to(output_root)),
        "artifact_dir": str(exp_dir.relative_to(output_root)),
    }

    start = time.time()
    with log_path.open("w") as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
        print(f"=== {spec.key} ===")
        print(json.dumps(asdict(spec), indent=2, default=str))
        try:
            configure_torch(spec.seed)
            solver = build_solver(spec)
            milestone_plan = resolve_milestones(spec, milestones)
            milestone_rows: list[dict[str, Any]] = []
            schedule = lr_schedule or []
            if milestone_plan:
                completed_epochs = 0
                for milestone in milestone_plan:
                    delta = milestone - completed_epochs
                    if delta <= 0:
                        continue
                    current_lr = apply_scheduled_lr(solver.optimizer, completed_epochs, spec.lr, schedule)
                    solver.fit(max_epochs=delta)
                    train_loss = [float(x) for x in solver.metrics_history["train_loss"]]
                    snapshot = compute_solution_metrics(spec, solver)
                    milestone_rows.append(
                        {
                            "epoch": milestone,
                            "elapsed_sec": time.time() - start,
                            "lr": current_lr,
                            "final_train_loss": float(train_loss[-1]),
                            "min_train_loss": float(min(train_loss)),
                            "solution_rmse": snapshot["solution_rmse"],
                            "residual_rmse": snapshot["residual_rmse"],
                        }
                    )
                    write_json(
                        exp_dir / "milestones.json",
                        {
                            "status": "running",
                            "key": spec.key,
                            "milestones": milestone_rows,
                        },
                    )
                    write_json(
                        exp_dir / "progress.json",
                        {
                            "status": "running",
                            "key": spec.key,
                            "base_key": spec.base_key,
                            "mode": spec.mode,
                            "seed": spec.seed,
                            "domain_label": spec.domain_label,
                            "duration_label": spec.duration_label,
                            "completed_epochs": milestone,
                            "elapsed_sec": time.time() - start,
                            "latest_milestone": milestone_rows[-1],
                        },
                    )
                    completed_epochs = milestone
            else:
                apply_scheduled_lr(solver.optimizer, 0, spec.lr, schedule)
                solver.fit(max_epochs=spec.epochs)
                milestone_rows = []
            train_loss = [float(x) for x in solver.metrics_history["train_loss"]]
            save_loss_plot(train_loss, exp_dir / "loss.png", spec.key)
            metrics = save_solution_artifacts(spec, solver, exp_dir)
            convergence = find_convergence(milestone_rows, target_solution_rmse, target_residual_rmse)
            summary.update(
                {
                    "status": "ok",
                    "elapsed_sec": time.time() - start,
                    "final_train_loss": float(train_loss[-1]),
                    "min_train_loss": float(min(train_loss)),
                }
            )
            if milestone_rows:
                summary["milestones"] = milestone_rows
                write_json(exp_dir / "milestones.json", {"milestones": milestone_rows})
            progress_path = exp_dir / "progress.json"
            if progress_path.exists():
                progress_path.unlink()
            if convergence:
                summary["converged_epoch"] = convergence["epoch"]
                summary["converged_elapsed_sec"] = convergence["elapsed_sec"]
                summary["target_solution_rmse"] = target_solution_rmse
                summary["target_residual_rmse"] = target_residual_rmse
            summary.update(metrics)
        except Exception as exc:
            traceback.print_exc()
            if milestone_rows:
                write_json(
                    exp_dir / "milestones.json",
                    {
                        "status": "failed",
                        "key": spec.key,
                        "milestones": milestone_rows,
                    },
                )
            summary.update(
                {
                    "status": "failed",
                    "elapsed_sec": time.time() - start,
                    "error": repr(exc),
                }
            )

    write_json(exp_dir / "summary.json", summary)
    return summary


def write_summary(
    output_root: Path,
    results: list[dict[str, Any]],
    target_solution_rmse: float | None = None,
    target_residual_rmse: float | None = None,
):
    counts = {
        "total": len(results),
        "ok": sum(1 for row in results if row.get("status") == "ok"),
        "failures": sum(1 for row in results if row.get("status") != "ok"),
        "seeds": len({row.get("seed") for row in results}),
    }
    write_json(
        output_root / "summary.json",
        {
            "counts": counts,
            "target_solution_rmse": target_solution_rmse,
            "target_residual_rmse": target_residual_rmse,
            "results": results,
        },
    )

    keys = sorted({key for row in results for key in row})
    with (output_root / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((row["seed"], row["domain_label"], row["duration_label"]), []).append(row)

    per_seed: dict[str, Any] = {}
    for (seed, domain_label, duration_label), rows in grouped.items():
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        if not ok_rows:
            continue
        per_seed[f"seed_{seed:04d}__{domain_label}__{duration_label}"] = {
            "best_solution_rmse": min(ok_rows, key=lambda row: row["solution_rmse"])["key"],
            "best_residual_rmse": min(ok_rows, key=lambda row: row["residual_rmse"])["key"],
            "ranked_solution_rmse": [
                {"key": row["key"], "mode": row["mode"], "solution_rmse": row["solution_rmse"]}
                for row in sorted(ok_rows, key=lambda row: row["solution_rmse"])
            ],
        }

    aggregates: dict[str, Any] = {}
    aggregate_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in results:
        if row.get("status") != "ok":
            continue
        aggregate_rows.setdefault((row["mode"], row["domain_label"], row["duration_label"]), []).append(row)

    for (mode, domain_label, duration_label), rows in sorted(aggregate_rows.items()):
        solution_values = [row["solution_rmse"] for row in rows]
        residual_values = [row["residual_rmse"] for row in rows]
        aggregates[f"{mode}__{domain_label}__{duration_label}"] = {
            "mode": mode,
            "domain_label": domain_label,
            "duration_label": duration_label,
            "runs": len(rows),
            "seeds": [row["seed"] for row in rows],
            "solution_rmse_mean": float(statistics.fmean(solution_values)),
            "solution_rmse_median": float(statistics.median(solution_values)),
            "solution_rmse_std": float(statistics.pstdev(solution_values)) if len(solution_values) > 1 else 0.0,
            "residual_rmse_mean": float(statistics.fmean(residual_values)),
            "residual_rmse_median": float(statistics.median(residual_values)),
            "residual_rmse_std": float(statistics.pstdev(residual_values)) if len(residual_values) > 1 else 0.0,
        }

    milestone_aggregates: dict[str, Any] = {}
    milestone_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in results:
        for milestone in row.get("milestones", []):
            milestone_groups.setdefault(
                (row["mode"], row["domain_label"], row["duration_label"], milestone["epoch"]),
                [],
            ).append(milestone)

    for (mode, domain_label, duration_label, epoch), rows in sorted(milestone_groups.items()):
        milestone_aggregates[f"{mode}__{domain_label}__{duration_label}__epoch_{epoch}"] = {
            "mode": mode,
            "domain_label": domain_label,
            "duration_label": duration_label,
            "epoch": epoch,
            "runs": len(rows),
            "solution_rmse_mean": float(statistics.fmean(row["solution_rmse"] for row in rows)),
            "solution_rmse_median": float(statistics.median(row["solution_rmse"] for row in rows)),
            "residual_rmse_mean": float(statistics.fmean(row["residual_rmse"] for row in rows)),
            "residual_rmse_median": float(statistics.median(row["residual_rmse"] for row in rows)),
            "elapsed_sec_mean": float(statistics.fmean(row["elapsed_sec"] for row in rows)),
        }

    convergence_rows = [row for row in results if row.get("converged_epoch") is not None]
    convergence = {}
    if convergence_rows:
        by_mode: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in convergence_rows:
            by_mode.setdefault((row["mode"], row["domain_label"], row["duration_label"]), []).append(row)
        for (mode, domain_label, duration_label), rows in sorted(by_mode.items()):
            convergence[f"{mode}__{domain_label}__{duration_label}"] = {
                "mode": mode,
                "domain_label": domain_label,
                "duration_label": duration_label,
                "target_solution_rmse": target_solution_rmse,
                "target_residual_rmse": target_residual_rmse,
                "solved_runs": len(rows),
                "solved_seeds": [row["seed"] for row in rows],
                "converged_epoch_mean": float(statistics.fmean(row["converged_epoch"] for row in rows)),
                "converged_epoch_median": float(statistics.median(row["converged_epoch"] for row in rows)),
                "converged_elapsed_sec_mean": float(statistics.fmean(row["converged_elapsed_sec"] for row in rows)),
                "converged_elapsed_sec_median": float(statistics.median(row["converged_elapsed_sec"] for row in rows)),
            }

    paired_baseline_details: dict[str, Any] = {}
    paired_baseline_aggregates: dict[str, Any] = {}
    paired_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    epoch_budget_details: dict[str, Any] = {}
    epoch_budget_aggregates: dict[str, Any] = {}
    epoch_budget_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    time_budget_details: dict[str, Any] = {}
    time_budget_aggregates: dict[str, Any] = {}
    time_budget_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for (seed, domain_label, duration_label), rows in grouped.items():
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        baseline = next((row for row in ok_rows if row["mode"] == "baseline"), None)
        if baseline is None:
            continue
        for row in ok_rows:
            if row["mode"] == "baseline":
                continue
            baseline_converged = baseline.get("converged_epoch") is not None
            candidate_converged = row.get("converged_epoch") is not None
            beats_baseline_epoch = candidate_converged and (
                not baseline_converged or row["converged_epoch"] < baseline["converged_epoch"]
            )
            beats_baseline_time = candidate_converged and (
                not baseline_converged or row["converged_elapsed_sec"] < baseline["converged_elapsed_sec"]
            )
            detail = {
                "seed": seed,
                "domain_label": domain_label,
                "duration_label": duration_label,
                "mode": row["mode"],
                "baseline_key": baseline["key"],
                "candidate_key": row["key"],
                "baseline_solution_rmse": baseline["solution_rmse"],
                "candidate_solution_rmse": row["solution_rmse"],
                "solution_rmse_delta": row["solution_rmse"] - baseline["solution_rmse"],
                "baseline_residual_rmse": baseline["residual_rmse"],
                "candidate_residual_rmse": row["residual_rmse"],
                "residual_rmse_delta": row["residual_rmse"] - baseline["residual_rmse"],
                "final_solution_better": row["solution_rmse"] < baseline["solution_rmse"],
                "final_residual_better": row["residual_rmse"] < baseline["residual_rmse"],
                "baseline_converged": baseline_converged,
                "candidate_converged": candidate_converged,
                "baseline_converged_epoch": baseline.get("converged_epoch"),
                "candidate_converged_epoch": row.get("converged_epoch"),
                "baseline_converged_elapsed_sec": baseline.get("converged_elapsed_sec"),
                "candidate_converged_elapsed_sec": row.get("converged_elapsed_sec"),
                "solved_when_baseline_failed": candidate_converged and not baseline_converged,
                "beats_baseline_epoch": beats_baseline_epoch,
                "beats_baseline_time": beats_baseline_time,
                "strict_speedup": candidate_converged and (
                    not baseline_converged
                    or (
                        row["converged_epoch"] < baseline["converged_epoch"]
                        and row["converged_elapsed_sec"] < baseline["converged_elapsed_sec"]
                    )
                ),
            }
            detail_key = f"{row['mode']}__vs_baseline__seed_{seed:04d}__{domain_label}__{duration_label}"
            paired_baseline_details[detail_key] = detail
            paired_groups.setdefault((row["mode"], domain_label, duration_label), []).append(detail)
            epoch_budget_detail = {
                "seed": seed,
                "domain_label": domain_label,
                "duration_label": duration_label,
                "mode": row["mode"],
                "baseline_key": baseline["key"],
                "candidate_key": row["key"],
                "baseline_epoch_budget": baseline["epochs_run"],
                "candidate_epoch_budget": row["epochs_run"],
                "baseline_elapsed_sec": baseline["elapsed_sec"],
                "candidate_elapsed_sec": row["elapsed_sec"],
                "baseline_solution_rmse": baseline["solution_rmse"],
                "candidate_solution_rmse": row["solution_rmse"],
                "solution_rmse_delta": row["solution_rmse"] - baseline["solution_rmse"],
                "baseline_residual_rmse": baseline["residual_rmse"],
                "candidate_residual_rmse": row["residual_rmse"],
                "residual_rmse_delta": row["residual_rmse"] - baseline["residual_rmse"],
                "better_at_same_epoch_solution": row["solution_rmse"] < baseline["solution_rmse"],
                "better_at_same_epoch_residual": row["residual_rmse"] < baseline["residual_rmse"],
                "faster_for_same_epoch_budget": row["elapsed_sec"] < baseline["elapsed_sec"],
                "strict_same_epoch_and_time_win": (
                    row["solution_rmse"] < baseline["solution_rmse"]
                    and row["residual_rmse"] < baseline["residual_rmse"]
                    and row["elapsed_sec"] < baseline["elapsed_sec"]
                ),
            }
            epoch_budget_key = (
                f"{row['mode']}__same_epoch_budget__seed_{seed:04d}__{domain_label}__{duration_label}"
            )
            epoch_budget_details[epoch_budget_key] = epoch_budget_detail
            epoch_budget_groups.setdefault((row["mode"], domain_label, duration_label), []).append(
                epoch_budget_detail
            )

            candidate_time_snapshot = find_snapshot_within_elapsed(row, baseline["elapsed_sec"])
            time_budget_detail = {
                "seed": seed,
                "domain_label": domain_label,
                "duration_label": duration_label,
                "mode": row["mode"],
                "baseline_key": baseline["key"],
                "candidate_key": row["key"],
                "baseline_time_budget_sec": baseline["elapsed_sec"],
                "baseline_solution_rmse": baseline["solution_rmse"],
                "baseline_residual_rmse": baseline["residual_rmse"],
                "candidate_snapshot_epoch": candidate_time_snapshot["epoch"] if candidate_time_snapshot else None,
                "candidate_snapshot_elapsed_sec": (
                    candidate_time_snapshot["elapsed_sec"] if candidate_time_snapshot else None
                ),
                "candidate_solution_rmse_at_baseline_time": (
                    candidate_time_snapshot["solution_rmse"] if candidate_time_snapshot else None
                ),
                "candidate_residual_rmse_at_baseline_time": (
                    candidate_time_snapshot["residual_rmse"] if candidate_time_snapshot else None
                ),
                "has_candidate_snapshot_within_budget": candidate_time_snapshot is not None,
                "better_at_same_time_solution": candidate_time_snapshot is not None
                and candidate_time_snapshot["solution_rmse"] < baseline["solution_rmse"],
                "better_at_same_time_residual": candidate_time_snapshot is not None
                and candidate_time_snapshot["residual_rmse"] < baseline["residual_rmse"],
                "reached_baseline_final_quality_by_same_time": candidate_time_snapshot is not None
                and candidate_time_snapshot["solution_rmse"] <= baseline["solution_rmse"]
                and candidate_time_snapshot["residual_rmse"] <= baseline["residual_rmse"],
            }
            time_budget_key = (
                f"{row['mode']}__same_time_budget__seed_{seed:04d}__{domain_label}__{duration_label}"
            )
            time_budget_details[time_budget_key] = time_budget_detail
            time_budget_groups.setdefault((row["mode"], domain_label, duration_label), []).append(
                time_budget_detail
            )

    for (mode, domain_label, duration_label), rows in sorted(paired_groups.items()):
        paired_baseline_aggregates[f"{mode}__vs_baseline__{domain_label}__{duration_label}"] = {
            "mode": mode,
            "domain_label": domain_label,
            "duration_label": duration_label,
            "paired_runs": len(rows),
            "final_solution_better_count": sum(1 for row in rows if row["final_solution_better"]),
            "final_residual_better_count": sum(1 for row in rows if row["final_residual_better"]),
            "candidate_converged_count": sum(1 for row in rows if row["candidate_converged"]),
            "baseline_converged_count": sum(1 for row in rows if row["baseline_converged"]),
            "both_converged_count": sum(
                1 for row in rows if row["candidate_converged"] and row["baseline_converged"]
            ),
            "solved_when_baseline_failed_count": sum(
                1 for row in rows if row["solved_when_baseline_failed"]
            ),
            "beats_baseline_epoch_count": sum(1 for row in rows if row["beats_baseline_epoch"]),
            "beats_baseline_time_count": sum(1 for row in rows if row["beats_baseline_time"]),
            "strict_speedup_count": sum(1 for row in rows if row["strict_speedup"]),
            "solution_rmse_delta_mean": float(statistics.fmean(row["solution_rmse_delta"] for row in rows)),
            "residual_rmse_delta_mean": float(statistics.fmean(row["residual_rmse_delta"] for row in rows)),
        }

    for (mode, domain_label, duration_label), rows in sorted(epoch_budget_groups.items()):
        epoch_budget_aggregates[f"{mode}__same_epoch_budget__{domain_label}__{duration_label}"] = {
            "mode": mode,
            "domain_label": domain_label,
            "duration_label": duration_label,
            "paired_runs": len(rows),
            "better_at_same_epoch_solution_count": sum(
                1 for row in rows if row["better_at_same_epoch_solution"]
            ),
            "better_at_same_epoch_residual_count": sum(
                1 for row in rows if row["better_at_same_epoch_residual"]
            ),
            "faster_for_same_epoch_budget_count": sum(
                1 for row in rows if row["faster_for_same_epoch_budget"]
            ),
            "strict_same_epoch_and_time_win_count": sum(
                1 for row in rows if row["strict_same_epoch_and_time_win"]
            ),
            "solution_rmse_delta_mean": float(statistics.fmean(row["solution_rmse_delta"] for row in rows)),
            "residual_rmse_delta_mean": float(statistics.fmean(row["residual_rmse_delta"] for row in rows)),
        }

    for (mode, domain_label, duration_label), rows in sorted(time_budget_groups.items()):
        valid_rows = [row for row in rows if row["has_candidate_snapshot_within_budget"]]
        time_budget_aggregates[f"{mode}__same_time_budget__{domain_label}__{duration_label}"] = {
            "mode": mode,
            "domain_label": domain_label,
            "duration_label": duration_label,
            "paired_runs": len(rows),
            "rows_with_candidate_snapshot_count": len(valid_rows),
            "better_at_same_time_solution_count": sum(
                1 for row in rows if row["better_at_same_time_solution"]
            ),
            "better_at_same_time_residual_count": sum(
                1 for row in rows if row["better_at_same_time_residual"]
            ),
            "reached_baseline_final_quality_by_same_time_count": sum(
                1 for row in rows if row["reached_baseline_final_quality_by_same_time"]
            ),
        }

    duration_target_details: dict[str, Any] = {}
    duration_target_aggregates: dict[str, Any] = {}
    duration_target_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for (seed, domain_label, duration_label), rows in grouped.items():
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        baseline = next((row for row in ok_rows if row["mode"] == "baseline"), None)
        if baseline is None:
            continue
        baseline_milestones = baseline.get("milestones", [])
        if not baseline_milestones:
            continue
        baseline_hit = find_milestone_hit(
            baseline_milestones,
            baseline["solution_rmse"],
            baseline["residual_rmse"],
        )
        if baseline_hit is None:
            continue
        for row in ok_rows:
            if row["mode"] == "baseline":
                continue
            candidate_hit = None
            if row.get("milestones"):
                candidate_hit = find_milestone_hit(
                    row["milestones"],
                    baseline["solution_rmse"],
                    baseline["residual_rmse"],
                )
            detail = {
                "seed": seed,
                "domain_label": domain_label,
                "duration_label": duration_label,
                "mode": row["mode"],
                "baseline_key": baseline["key"],
                "candidate_key": row["key"],
                "target_solution_rmse": baseline["solution_rmse"],
                "target_residual_rmse": baseline["residual_rmse"],
                "baseline_hit_epoch": baseline_hit["epoch"],
                "baseline_hit_elapsed_sec": baseline_hit["elapsed_sec"],
                "candidate_hit_epoch": candidate_hit["epoch"] if candidate_hit else None,
                "candidate_hit_elapsed_sec": candidate_hit["elapsed_sec"] if candidate_hit else None,
                "candidate_reached_baseline_final_quality": candidate_hit is not None,
                "beats_baseline_to_final_quality_epoch": candidate_hit is not None
                and candidate_hit["epoch"] < baseline_hit["epoch"],
                "beats_baseline_to_final_quality_time": candidate_hit is not None
                and candidate_hit["elapsed_sec"] < baseline_hit["elapsed_sec"],
                "strict_final_quality_speedup": candidate_hit is not None
                and candidate_hit["epoch"] < baseline_hit["epoch"]
                and candidate_hit["elapsed_sec"] < baseline_hit["elapsed_sec"],
            }
            detail_key = (
                f"{row['mode']}__baseline_final_target__seed_{seed:04d}__{domain_label}__{duration_label}"
            )
            duration_target_details[detail_key] = detail
            duration_target_groups.setdefault((row["mode"], domain_label, duration_label), []).append(detail)

    for (mode, domain_label, duration_label), rows in sorted(duration_target_groups.items()):
        duration_target_aggregates[f"{mode}__baseline_final_target__{domain_label}__{duration_label}"] = {
            "mode": mode,
            "domain_label": domain_label,
            "duration_label": duration_label,
            "paired_runs": len(rows),
            "candidate_reached_baseline_final_quality_count": sum(
                1 for row in rows if row["candidate_reached_baseline_final_quality"]
            ),
            "beats_baseline_to_final_quality_epoch_count": sum(
                1 for row in rows if row["beats_baseline_to_final_quality_epoch"]
            ),
            "beats_baseline_to_final_quality_time_count": sum(
                1 for row in rows if row["beats_baseline_to_final_quality_time"]
            ),
            "strict_final_quality_speedup_count": sum(
                1 for row in rows if row["strict_final_quality_speedup"]
            ),
        }

    write_json(
        output_root / "comparisons.json",
        {
            "per_seed": per_seed,
            "aggregates": aggregates,
            "milestone_aggregates": milestone_aggregates,
            "convergence": convergence,
            "paired_baseline": {
                "details": paired_baseline_details,
                "aggregates": paired_baseline_aggregates,
            },
            "paired_epoch_budget": {
                "details": epoch_budget_details,
                "aggregates": epoch_budget_aggregates,
            },
            "paired_time_budget": {
                "details": time_budget_details,
                "aggregates": time_budget_aggregates,
            },
            "paired_duration_target": {
                "details": duration_target_details,
                "aggregates": duration_target_aggregates,
            },
        },
    )


def run_all(
    specs: list[ValidationSpec],
    output_root: Path,
    max_workers: int,
    milestones: list[int] | None = None,
    target_solution_rmse: float | None = None,
    target_residual_rmse: float | None = None,
    lr_schedule: list[tuple[int, float]] | None = None,
):
    results_by_index: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                run_spec,
                spec,
                output_root,
                milestones,
                target_solution_rmse,
                target_residual_rmse,
                lr_schedule,
            ): idx
            for idx, spec in enumerate(specs)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            spec = specs[idx]
            try:
                results_by_index[idx] = future.result()
            except Exception as exc:
                results_by_index[idx] = {
                    "key": spec.key,
                    "mode": spec.mode,
                    "status": "failed",
                    "error": repr(exc),
                }
            partial_results = [results_by_index[i] for i in sorted(results_by_index)]
            write_summary(output_root, partial_results, target_solution_rmse, target_residual_rmse)
    return [results_by_index[idx] for idx in sorted(results_by_index)]


def main():
    args = parse_args()
    seeds = resolve_seeds(args)
    initial_sigma = parse_optional_float(args.initial_sigma)
    sigma_min = parse_optional_float(args.sigma_min)
    sigma_max = parse_optional_float(args.sigma_max)
    lr_schedule = parse_lr_schedule(args.lr_schedule)
    specs = choose_specs(
        build_specs(
            seeds,
            args.short_epochs,
            args.long_epochs,
            tuple(args.input_neurons),
            tuple(args.baseline_input_neurons) if args.baseline_input_neurons else None,
            args.localisation_family,
            resolve_activation_name(args.activation, args.localisation_family),
            initial_sigma,
            sigma_min,
            sigma_max,
            args.lr,
        ),
        args.experiments,
    )
    if args.list:
        for spec in specs:
            print(
                f"{spec.key:34s} seed={spec.seed} epochs={spec.epochs:<5d} "
                f"domain={spec.domain_label:<5s} mode={spec.mode}"
            )
        return 0

    if args.device not in {"cpu", "auto"}:
        raise SystemExit("Only CPU is supported in this validation runner.")

    output_root = ensure_output_root(args.output_root)
    manifest = {
        "seeds": seeds,
        "max_workers": max(1, args.max_workers),
        "short_epochs": args.short_epochs,
        "long_epochs": args.long_epochs,
        "input_neurons": list(args.input_neurons),
        "lr": args.lr,
        "initial_sigma": initial_sigma,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "milestones": args.milestones,
        "target_solution_rmse": args.target_solution_rmse,
        "target_residual_rmse": args.target_residual_rmse,
        "lr_schedule": lr_schedule,
        "experiments": [asdict(spec) for spec in specs],
    }
    write_json(output_root / "manifest.json", manifest)
    results = run_all(
        specs,
        output_root,
        max(1, args.max_workers),
        args.milestones,
        args.target_solution_rmse,
        args.target_residual_rmse,
        lr_schedule,
    )
    write_summary(output_root, results, args.target_solution_rmse, args.target_residual_rmse)
    failures = sum(1 for row in results if row.get("status") != "ok")
    print(f"output_root={output_root}")
    print(f"ok={len(results) - failures}")
    print(f"failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
