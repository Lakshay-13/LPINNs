#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cycler import cycler


ROOT = Path(__file__).resolve().parents[1]

TEXT_COLOR = "#404040"
PLOT_COLORS = [
    "#009E73",
    "#0072B2",
    "#E69F00",
    "#F0E442",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#666666",
]

plt.rcParams.update(
    {
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "figure.figsize": (6, 4),
        "axes.labelsize": 12,
        "axes.linewidth": 1.5,
        "xtick.labelsize": 12,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "ytick.labelsize": 12,
        "axes.edgecolor": TEXT_COLOR,
        "figure.titlesize": 12,
        "axes.prop_cycle": cycler(color=PLOT_COLORS),
        "axes.titlecolor": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


LOCALISER_FAMILIES = [
    "gaussian",
    "activated",
    "super_gaussian",
    "ricker",
    "boxcar",
    "laplace",
    "cauchy",
    "raised_cosine",
    "bump",
    "gabor",
    "morlet",
    "inverse_quadratic",
    "triangular",
]

DEFAULT_VARIANTS = ["fixed", "sigma", "mu_sigma"]

EXPERIMENT_ORDER = [
    "ho_function_screen_v1",
    "ho_same_network_v1",
    "ho_mixed_size_v1",
    "heat_function_screen_v1",
    "heat_same_network_v1",
    "heat_mixed_size_v1",
    "4d_function_screen_v1",
    "4d_same_network_v1",
    "4d_mixed_size_v1",
]


@dataclass(frozen=True)
class ProblemConfig:
    key: str
    label: str
    activation: str
    domains: dict[str, float]
    budgets: dict[str, int]
    screen_architecture: tuple[int, ...]
    same_network_architectures: list[tuple[int, ...]]
    mixed_size_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]]
    collocation_points: int
    eval_points: int
    lr: float


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    problem: str
    mode: str
    note: str


@dataclass(frozen=True)
class SeedRunSpec:
    experiment: str
    problem: str
    comparison_mode: str
    localiser_family: str
    variant_key: str
    variant_name: str
    localisation: bool
    mu_learnable: bool
    sigma_learnable: bool
    domain_label: str
    domain_value: float
    budget_label: str
    epochs: int
    architecture_label: str
    architecture: tuple[int, ...]
    baseline_architecture_label: str | None
    localised_architecture_label: str | None
    pair_group_relpath: str | None
    group_relpath: str
    seed_relpath: str
    activation: str
    seed: int
    note: str = ""


PROBLEM_CONFIGS: dict[str, ProblemConfig] = {
    "ho": ProblemConfig(
        key="ho",
        label="Harmonic Oscillator",
        activation="sine",
        domains={
            "pi": math.pi,
            "2pi": 2 * math.pi,
            "3pi": 3 * math.pi,
            "4pi": 4 * math.pi,
        },
        budgets={
            "1k": 1_000,
            "3k": 3_000,
            "5k": 5_000,
            "10k": 10_000,
            "50k": 50_000,
            "100k": 100_000,
        },
        screen_architecture=(64, 64),
        same_network_architectures=[
            (16,),
            (32,),
            (64,),
            (128,),
            (256,),
            (16, 16),
            (32, 32),
            (64, 64),
            (128, 128),
        ],
        mixed_size_pairs=[
            ((32, 32), (16, 16)),
            ((64, 64), (32, 32)),
            ((128, 128), (64, 64)),
            ((256,), (128,)),
        ],
        collocation_points=256,
        eval_points=512,
        lr=2e-3,
    ),
    "heat": ProblemConfig(
        key="heat",
        label="Heat Equation",
        activation="tanh",
        domains={
            "pi": math.pi,
            "2pi": 2 * math.pi,
            "4pi": 4 * math.pi,
            "8pi": 8 * math.pi,
        },
        budgets={
            "1k": 1_000,
            "3k": 3_000,
            "5k": 5_000,
            "10k": 10_000,
            "50k": 50_000,
        },
        screen_architecture=(64, 64),
        same_network_architectures=[
            (32,),
            (64,),
            (128,),
            (32, 32),
            (64, 64),
            (128, 128),
        ],
        mixed_size_pairs=[
            ((64, 64), (32, 32)),
            ((128, 128), (64, 64)),
        ],
        collocation_points=512,
        eval_points=181,
        lr=1e-3,
    ),
    "4d": ProblemConfig(
        key="4d",
        label="4D Problem",
        activation="tanh",
        domains={
            "pi": math.pi,
            "2pi": 2 * math.pi,
            "4pi": 4 * math.pi,
        },
        budgets={
            "1k": 1_000,
            "3k": 3_000,
            "5k": 5_000,
            "10k": 10_000,
        },
        screen_architecture=(32, 32),
        same_network_architectures=[
            (16,),
            (32,),
            (64,),
            (16, 16),
            (32, 32),
            (64, 64),
        ],
        mixed_size_pairs=[
            ((32, 32), (16, 16)),
            ((64, 64), (32, 32)),
        ],
        collocation_points=256,
        eval_points=256,
        lr=1e-3,
    ),
}

EXPERIMENT_CONFIGS: dict[str, ExperimentConfig] = {
    name: ExperimentConfig(
        name=name,
        problem=name.split("_", 1)[0] if not name.startswith("4d_") else "4d",
        mode=name.split("_")[1],
        note=name,
    )
    for name in EXPERIMENT_ORDER
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the localisation campaign across HO, Heat, and 4D.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_ORDER,
        default=None,
        help="Experiment roots to run. Defaults to the full campaign order.",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=["ho", "heat", "4d"],
        default=None,
        help="Optional problem-level filter applied after experiment selection.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=LOCALISER_FAMILIES,
        default=None,
        help="Localiser families to include for localised runs.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=None,
        help="Localiser variants to include: fixed, sigma, mu_sigma.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Domain labels or values (e.g. pi 2pi 4pi).",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=None,
        help="Epoch budgets such as 20 1k 3k 10000.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=None,
        help="Architecture filters such as 64x64 or baseline_64x64__localised_32x32.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def format_architecture_label(architecture: tuple[int, ...]) -> str:
    return "x".join(str(width) for width in architecture)


def format_pair_label(
    baseline_architecture: tuple[int, ...], localised_architecture: tuple[int, ...]
) -> str:
    return (
        f"baseline_{format_architecture_label(baseline_architecture)}"
        f"__localised_{format_architecture_label(localised_architecture)}"
    )


def parse_budget_token(token: str) -> int:
    lowered = token.strip().lower()
    if lowered.endswith("k"):
        return int(float(lowered[:-1]) * 1_000)
    return int(lowered)


def format_budget_label(epochs: int) -> str:
    if epochs % 1000 == 0 and epochs >= 1000:
        return f"{epochs // 1000}k"
    return str(epochs)


def parse_domain_token(token: str) -> float:
    lowered = token.strip().lower()
    if lowered == "pi":
        return math.pi
    if lowered.endswith("pi"):
        factor = lowered[:-2]
        return float(factor) * math.pi
    return float(lowered)


def domain_label_for_value(defaults: dict[str, float], value: float) -> str:
    for label, candidate in defaults.items():
        if math.isclose(candidate, value, rel_tol=0.0, abs_tol=1e-9):
            return label
    ratio = value / math.pi
    if math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
        if round(ratio) == 1:
            return "pi"
        return f"{int(round(ratio))}pi"
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value))


class SineActivation(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(value)


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "sine": SineActivation,
}


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    threads = int(os.environ.get("LOCALISATION_TORCH_THREADS", "1"))
    torch.set_num_threads(max(1, threads))


def localiser_basis(family: str, z: torch.Tensor) -> torch.Tensor:
    abs_z = z.abs()
    if family == "gaussian":
        return torch.exp(-0.5 * z.pow(2))
    if family == "activated":
        return torch.sigmoid(6.0 * (1.0 - z.pow(2)))
    if family == "super_gaussian":
        return torch.exp(-0.5 * z.pow(8))
    if family == "ricker":
        return (1.0 - z.pow(2)) * torch.exp(-0.5 * z.pow(2))
    if family == "boxcar":
        return torch.sigmoid(10.0 * (z + 1.0)) - torch.sigmoid(10.0 * (z - 1.0))
    if family == "laplace":
        return torch.exp(-abs_z)
    if family == "cauchy":
        return 1.0 / (1.0 + z.pow(2))
    if family == "raised_cosine":
        inside = abs_z <= 1.0
        return torch.where(inside, 0.5 * (1.0 + torch.cos(math.pi * z)), torch.zeros_like(z))
    if family == "bump":
        inside = abs_z < 1.0
        interior = torch.clamp(1.0 - z.pow(2), min=1e-6)
        return torch.where(inside, torch.exp(-1.0 / interior), torch.zeros_like(z))
    if family == "gabor":
        return torch.exp(-0.5 * z.pow(2)) * torch.cos(2.5 * math.pi * z)
    if family == "morlet":
        correction = math.exp(-0.5 * 25.0)
        return torch.exp(-0.5 * z.pow(2)) * (torch.cos(5.0 * z) - correction)
    if family == "inverse_quadratic":
        return 1.0 / torch.sqrt(1.0 + z.pow(2))
    if family == "triangular":
        return torch.clamp(1.0 - abs_z, min=0.0)
    raise ValueError(f"Unsupported localiser family: {family}")


class LocalisedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...],
        activation: str,
        mins: tuple[float, ...],
        maxs: tuple[float, ...],
        localisation: bool,
        family: str,
        mu_learnable: bool,
        sigma_learnable: bool,
    ) -> None:
        super().__init__()
        if not hidden_layers:
            raise ValueError("At least one hidden layer is required.")
        self.localisation = localisation
        self.family = family
        self.input_dim = input_dim
        self.activation = ACTIVATIONS[activation]()

        mins_tensor = torch.tensor(mins, dtype=torch.float32).view(1, -1)
        scales_tensor = torch.tensor(maxs, dtype=torch.float32).view(1, -1) - mins_tensor
        self.register_buffer("input_mins", mins_tensor)
        self.register_buffer("input_scales", torch.clamp(scales_tensor, min=1e-6))

        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_layers[index], hidden_layers[index + 1])
            for index in range(len(hidden_layers) - 1)
        )
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

        if localisation:
            width = hidden_layers[0]
            mu_init = torch.stack(
                [torch.linspace(0.05, 0.95, width, dtype=torch.float32) for _ in range(input_dim)],
                dim=0,
            )
            sigma_value = max(0.08, 1.5 / max(width, 4))
            sigma_init = torch.full((input_dim, width), sigma_value, dtype=torch.float32)
            if mu_learnable:
                clamped = mu_init.clamp(1e-4, 1 - 1e-4)
                self.raw_mu = nn.Parameter(torch.log(clamped / (1 - clamped)))
            else:
                self.register_buffer("fixed_mu", mu_init)
            if sigma_learnable:
                self.raw_sigma = nn.Parameter(inverse_softplus(sigma_init))
            else:
                self.register_buffer("fixed_sigma", sigma_init)

    def current_mu(self) -> torch.Tensor:
        if hasattr(self, "raw_mu"):
            return torch.sigmoid(self.raw_mu)
        return self.fixed_mu

    def current_sigma(self) -> torch.Tensor:
        if hasattr(self, "raw_sigma"):
            return F.softplus(self.raw_sigma) + 1e-3
        return self.fixed_sigma

    def gate(self, x_norm: torch.Tensor) -> torch.Tensor:
        mu = self.current_mu()
        sigma = self.current_sigma()
        z = (x_norm[:, :, None] - mu[None, :, :]) / sigma[None, :, :]
        return localiser_basis(self.family, z).prod(dim=1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x_norm = (coords - self.input_mins) / self.input_scales
        hidden = self.activation(self.input_layer(x_norm))
        if self.localisation:
            hidden = hidden * self.gate(x_norm)
        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))
        return self.output_layer(hidden)


def gradient(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def numerical_ho_reference(domain_value: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, domain_value, steps, dtype=np.float64)
    values = np.zeros_like(xs)
    state = np.array([0.0, 1.0], dtype=np.float64)

    def dynamics(current: np.ndarray) -> np.ndarray:
        return np.array([current[1], -math.sin(float(current[0]))], dtype=np.float64)

    for index in range(1, steps):
        dt = xs[index] - xs[index - 1]
        k1 = dynamics(state)
        k2 = dynamics(state + 0.5 * dt * k1)
        k3 = dynamics(state + 0.5 * dt * k2)
        k4 = dynamics(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        values[index] = state[0]
    return xs, values


def ho_solution(model: nn.Module, coords: torch.Tensor) -> torch.Tensor:
    t = coords[:, :1]
    return t + t.pow(2) * model(coords)


def heat_solution(model: nn.Module, coords: torch.Tensor, domain_value: float) -> torch.Tensor:
    x = coords[:, :1]
    t = coords[:, 1:2]
    return torch.sin(x) + t * x * (domain_value - x) * model(coords)


def four_d_solution(model: nn.Module, coords: torch.Tensor) -> torch.Tensor:
    envelope = coords.prod(dim=1, keepdim=True)
    return envelope * model(coords)


def exact_heat(coords: torch.Tensor) -> torch.Tensor:
    x = coords[:, :1]
    t = coords[:, 1:2]
    return torch.sin(x) * torch.exp(-t)


def exact_four_d(coords: torch.Tensor) -> torch.Tensor:
    return 10.0 * coords.prod(dim=1, keepdim=True)


def ho_residual(model: nn.Module, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coords = coords.requires_grad_(True)
    prediction = ho_solution(model, coords)
    du_dt = gradient(prediction, coords)
    d2u_dt2 = gradient(du_dt[:, :1], coords)[:, :1]
    residual = d2u_dt2 + torch.sin(prediction)
    return prediction, residual


def heat_residual(model: nn.Module, coords: torch.Tensor, domain_value: float) -> tuple[torch.Tensor, torch.Tensor]:
    coords = coords.requires_grad_(True)
    prediction = heat_solution(model, coords, domain_value)
    grads = gradient(prediction, coords)
    u_x = grads[:, :1]
    u_t = grads[:, 1:2]
    u_xx = gradient(u_x, coords)[:, :1]
    residual = u_t - u_xx
    return prediction, residual


def four_d_residual(model: nn.Module, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coords = coords.requires_grad_(True)
    prediction = four_d_solution(model, coords)
    d1 = gradient(prediction, coords)[:, 0:1]
    d2 = gradient(d1, coords)[:, 1:2]
    d3 = gradient(d2, coords)[:, 2:3]
    d4 = gradient(d3, coords)[:, 3:4]
    residual = d4 - 10.0
    return prediction, residual


def sample_collocation(problem: str, count: int, domain_value: float) -> torch.Tensor:
    if problem == "ho":
        return torch.rand(count, 1, dtype=torch.float32) * domain_value
    if problem == "heat":
        x = torch.rand(count, 1, dtype=torch.float32) * domain_value
        t = torch.rand(count, 1, dtype=torch.float32)
        return torch.cat([x, t], dim=1)
    if problem == "4d":
        uniform_count = max(1, int(count * 0.7))
        focus_count = count - uniform_count
        uniform = torch.rand(uniform_count, 4, dtype=torch.float32) * domain_value
        focus = torch.clamp(0.8 + 0.05 * torch.randn(focus_count, 4, dtype=torch.float32), 0.0, 1.0)
        focus = focus * domain_value
        return torch.cat([uniform, focus], dim=0)
    raise ValueError(f"Unsupported problem: {problem}")


def build_model(spec: SeedRunSpec) -> nn.Module:
    problem = PROBLEM_CONFIGS[spec.problem]
    input_dims = {"ho": 1, "heat": 2, "4d": 4}
    mins = {"ho": (0.0,), "heat": (0.0, 0.0), "4d": (0.0, 0.0, 0.0, 0.0)}
    maxs = {
        "ho": (spec.domain_value,),
        "heat": (spec.domain_value, 1.0),
        "4d": (spec.domain_value, spec.domain_value, spec.domain_value, spec.domain_value),
    }
    return LocalisedMLP(
        input_dim=input_dims[spec.problem],
        hidden_layers=spec.architecture,
        activation=spec.activation or problem.activation,
        mins=mins[spec.problem],
        maxs=maxs[spec.problem],
        localisation=spec.localisation,
        family=spec.localiser_family,
        mu_learnable=spec.mu_learnable,
        sigma_learnable=spec.sigma_learnable,
    )


def loss_from_residual(residual: torch.Tensor) -> torch.Tensor:
    return residual.pow(2).mean()


def evaluate_ho(spec: SeedRunSpec, model: nn.Module) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    xs, reference = numerical_ho_reference(spec.domain_value, PROBLEM_CONFIGS["ho"].eval_points)
    coords = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
    prediction, residual = ho_residual(model, coords)
    pred_np = prediction.detach().cpu().numpy().reshape(-1)
    ref_np = reference.reshape(-1)
    residual_np = residual.detach().cpu().numpy().reshape(-1)
    metrics = {
        "solution_rmse": float(np.sqrt(np.mean((pred_np - ref_np) ** 2))),
        "solution_mae": float(np.mean(np.abs(pred_np - ref_np))),
        "residual_rmse": float(np.sqrt(np.mean(residual_np**2))),
    }
    bundle = {
        "curve_x": xs.astype(np.float64),
        "prediction_curve": pred_np.astype(np.float64),
        "reference_curve": ref_np.astype(np.float64),
        "residual_curve": residual_np.astype(np.float64),
        "curve_label": np.array(["t"]),
        "curve_title": np.array(["Oscillation Over Domain"]),
    }
    return metrics, bundle


def evaluate_heat(spec: SeedRunSpec, model: nn.Module) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    x_values = np.linspace(0.0, spec.domain_value, PROBLEM_CONFIGS["heat"].eval_points, dtype=np.float64)
    t_values = np.linspace(0.0, 1.0, 61, dtype=np.float64)
    xx, tt = np.meshgrid(x_values, t_values, indexing="xy")
    coords = torch.tensor(np.column_stack([xx.reshape(-1), tt.reshape(-1)]), dtype=torch.float32)
    prediction, residual = heat_residual(model, coords, spec.domain_value)
    reference = exact_heat(coords)
    pred_np = prediction.detach().cpu().numpy().reshape(tt.shape)
    ref_np = reference.detach().cpu().numpy().reshape(tt.shape)
    residual_np = residual.detach().cpu().numpy().reshape(tt.shape)
    slice_index = len(t_values) // 2
    metrics = {
        "solution_rmse": float(np.sqrt(np.mean((pred_np - ref_np) ** 2))),
        "solution_mae": float(np.mean(np.abs(pred_np - ref_np))),
        "residual_rmse": float(np.sqrt(np.mean(residual_np**2))),
    }
    bundle = {
        "curve_x": x_values.astype(np.float64),
        "prediction_curve": pred_np[slice_index].astype(np.float64),
        "reference_curve": ref_np[slice_index].astype(np.float64),
        "residual_curve": residual_np[slice_index].astype(np.float64),
        "curve_label": np.array(["x at t=0.5"]),
        "curve_title": np.array(["Mid-Time Slice"]),
        "grid_prediction": pred_np.astype(np.float64),
        "grid_reference": ref_np.astype(np.float64),
        "grid_x": x_values.astype(np.float64),
        "grid_t": t_values.astype(np.float64),
    }
    return metrics, bundle


def evaluate_four_d(spec: SeedRunSpec, model: nn.Module) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    eval_points = rng.uniform(0.0, spec.domain_value, size=(1024, 4)).astype(np.float32)
    coords = torch.tensor(eval_points, dtype=torch.float32)
    prediction, residual = four_d_residual(model, coords)
    reference = exact_four_d(coords)
    pred_np = prediction.detach().cpu().numpy().reshape(-1)
    ref_np = reference.detach().cpu().numpy().reshape(-1)
    residual_np = residual.detach().cpu().numpy().reshape(-1)

    line = np.linspace(0.0, spec.domain_value, PROBLEM_CONFIGS["4d"].eval_points, dtype=np.float64)
    slice_coords = np.column_stack([line, line, line, line]).astype(np.float32)
    slice_prediction = four_d_solution(model, torch.tensor(slice_coords)).detach().cpu().numpy().reshape(-1)
    slice_reference = 10.0 * line**4
    metrics = {
        "solution_rmse": float(np.sqrt(np.mean((pred_np - ref_np) ** 2))),
        "solution_mae": float(np.mean(np.abs(pred_np - ref_np))),
        "residual_rmse": float(np.sqrt(np.mean(residual_np**2))),
    }
    bundle = {
        "curve_x": line.astype(np.float64),
        "prediction_curve": slice_prediction.astype(np.float64),
        "reference_curve": slice_reference.astype(np.float64),
        "residual_curve": np.zeros_like(line, dtype=np.float64),
        "curve_label": np.array(["diagonal slice"]),
        "curve_title": np.array(["4D Diagonal Slice"]),
        "parity_reference": ref_np.astype(np.float64),
        "parity_prediction": pred_np.astype(np.float64),
    }
    return metrics, bundle


def evaluate_spec(spec: SeedRunSpec, model: nn.Module) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    if spec.problem == "ho":
        return evaluate_ho(spec, model)
    if spec.problem == "heat":
        return evaluate_heat(spec, model)
    if spec.problem == "4d":
        return evaluate_four_d(spec, model)
    raise ValueError(f"Unsupported problem: {spec.problem}")


def save_loss_plot(history: list[float], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.log10(np.maximum(np.array(history, dtype=np.float64), 1e-30)), linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log10(train_loss)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_solution_plot(spec: SeedRunSpec, bundle: dict[str, np.ndarray], out_path: Path) -> None:
    if spec.problem == "ho":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(bundle["curve_x"], bundle["reference_curve"], linewidth=2.0, label="reference")
        axes[0].plot(bundle["curve_x"], bundle["prediction_curve"], linewidth=1.6, label="prediction")
        axes[0].set_title("Solution")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("u(t)")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()
        axes[1].plot(bundle["curve_x"], bundle["residual_curve"], linewidth=1.2, color=PLOT_COLORS[4])
        axes[1].set_title("Residual")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("residual")
        axes[1].grid(True, alpha=0.25)
    elif spec.problem == "heat":
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        pred = axes[0].imshow(
            bundle["grid_prediction"],
            aspect="auto",
            origin="lower",
            extent=[0.0, spec.domain_value, 0.0, 1.0],
            cmap="viridis",
        )
        axes[0].set_title("Prediction")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("t")
        fig.colorbar(pred, ax=axes[0], shrink=0.85)
        ref = axes[1].imshow(
            bundle["grid_reference"],
            aspect="auto",
            origin="lower",
            extent=[0.0, spec.domain_value, 0.0, 1.0],
            cmap="viridis",
        )
        axes[1].set_title("Reference")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("t")
        fig.colorbar(ref, ax=axes[1], shrink=0.85)
        axes[2].plot(bundle["curve_x"], bundle["reference_curve"], linewidth=2.0, label="reference")
        axes[2].plot(bundle["curve_x"], bundle["prediction_curve"], linewidth=1.5, label="prediction")
        axes[2].set_title("Mid-Time Slice")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("u(x, 0.5)")
        axes[2].grid(True, alpha=0.25)
        axes[2].legend()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(bundle["curve_x"], bundle["reference_curve"], linewidth=2.0, label="reference")
        axes[0].plot(bundle["curve_x"], bundle["prediction_curve"], linewidth=1.5, label="prediction")
        axes[0].set_title("Diagonal Slice")
        axes[0].set_xlabel("x1 = x2 = x3 = x4")
        axes[0].set_ylabel("u")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()
        axes[1].scatter(
            bundle["parity_reference"],
            bundle["parity_prediction"],
            s=8,
            alpha=0.35,
            color=PLOT_COLORS[1],
        )
        lower = min(bundle["parity_reference"].min(), bundle["parity_prediction"].min())
        upper = max(bundle["parity_reference"].max(), bundle["parity_prediction"].max())
        axes[1].plot([lower, upper], [lower, upper], linestyle="--", color=PLOT_COLORS[4], linewidth=1.2)
        axes[1].set_title("Parity")
        axes[1].set_xlabel("reference")
        axes[1].set_ylabel("prediction")
        axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def train_step(spec: SeedRunSpec, model: nn.Module) -> tuple[torch.Tensor, float]:
    batch = sample_collocation(spec.problem, PROBLEM_CONFIGS[spec.problem].collocation_points, spec.domain_value)
    if spec.problem == "ho":
        _, residual = ho_residual(model, batch)
    elif spec.problem == "heat":
        _, residual = heat_residual(model, batch, spec.domain_value)
    else:
        _, residual = four_d_residual(model, batch)
    loss = loss_from_residual(residual)
    return loss, float(torch.sqrt(loss.detach()).cpu())


def run_seed_worker(spec_payload: dict[str, Any], batch_root_text: str, overwrite: bool) -> dict[str, Any]:
    spec = SeedRunSpec(**spec_payload)
    batch_root = Path(batch_root_text)
    seed_dir = batch_root / spec.seed_relpath
    group_dir = batch_root / spec.group_relpath
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = seed_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        try:
            existing = json.loads(metrics_path.read_text())
            if existing.get("status") in {"completed", "failed"}:
                return {
                    "status": "skipped",
                    "group_relpath": spec.group_relpath,
                    "seed": spec.seed,
                    "metrics_path": str(metrics_path),
                }
        except json.JSONDecodeError:
            pass

    log_path = seed_dir / "raw.log"
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    history: list[float] = []
    best_metrics: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None

    with log_path.open("w", encoding="utf-8") as handle:
        def log(message: str) -> None:
            handle.write(f"{message}\n")
            handle.flush()

        try:
            set_reproducibility(spec.seed)
            log(f"start={started_at}")
            log(f"experiment={spec.experiment}")
            log(f"problem={spec.problem}")
            log(f"family={spec.localiser_family}")
            log(f"variant={spec.variant_name}")
            log(f"epochs={spec.epochs}")
            model = build_model(spec)
            optimizer = torch.optim.Adam(model.parameters(), lr=PROBLEM_CONFIGS[spec.problem].lr)
            eval_interval = max(1, spec.epochs // 20)
            wall_start = time.perf_counter()

            for epoch in range(1, spec.epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, train_residual_rmse = train_step(spec, model)
                loss.backward()
                optimizer.step()
                history.append(float(loss.detach().cpu()))

                if epoch == 1 or epoch % eval_interval == 0 or epoch == spec.epochs:
                    metrics, _ = evaluate_spec(spec, model)
                    checkpoint = {
                        "epoch": epoch,
                        "train_loss": history[-1],
                        "train_residual_rmse": train_residual_rmse,
                        **metrics,
                    }
                    if best_metrics is None or checkpoint["residual_rmse"] < best_metrics["residual_rmse"]:
                        best_metrics = checkpoint
                        best_state = copy.deepcopy(model.state_dict())
                    log(
                        "epoch={epoch} train_loss={loss:.6e} residual_rmse={res:.6e} solution_rmse={sol:.6e}".format(
                            epoch=epoch,
                            loss=history[-1],
                            res=metrics["residual_rmse"],
                            sol=metrics["solution_rmse"],
                        )
                    )

            if best_state is not None:
                model.load_state_dict(best_state)

            final_metrics, bundle = evaluate_spec(spec, model)
            wall_clock_seconds = time.perf_counter() - wall_start
            np.save(seed_dir / "loss_history.npy", np.array(history, dtype=np.float64))
            np.savez(seed_dir / "solution_slice.npz", **bundle)
            save_loss_plot(history, seed_dir / "loss_plot.png", f"{spec.variant_name} seed {spec.seed:02d}")
            save_solution_plot(spec, bundle, seed_dir / "solution_plot.png")

            metrics_payload = {
                "status": "completed",
                "started_at": started_at,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "experiment": spec.experiment,
                "problem": spec.problem,
                "comparison_mode": spec.comparison_mode,
                "localiser_family": spec.localiser_family,
                "variant": spec.variant_name,
                "domain_label": spec.domain_label,
                "domain_value": spec.domain_value,
                "budget_label": spec.budget_label,
                "epochs": spec.epochs,
                "architecture_label": spec.architecture_label,
                "architecture": list(spec.architecture),
                "baseline_architecture_label": spec.baseline_architecture_label,
                "localised_architecture_label": spec.localised_architecture_label,
                "seed": spec.seed,
                "pair_group_relpath": spec.pair_group_relpath,
                "wall_clock_seconds": wall_clock_seconds,
                "final_train_loss": history[-1],
                "final_train_residual_rmse": float(math.sqrt(max(history[-1], 0.0))),
                **final_metrics,
                "best_checkpoint": best_metrics,
                "note": spec.note,
            }
            write_json(metrics_path, metrics_payload)
            return {
                "status": "completed",
                "group_relpath": spec.group_relpath,
                "seed": spec.seed,
                "metrics_path": str(metrics_path),
            }
        except Exception as exc:  # pragma: no cover - this is runtime protection.
            error_summary = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            stack = traceback.format_exc()
            log(stack)
            failure_payload = {
                "status": "failed",
                "started_at": started_at,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "experiment": spec.experiment,
                "problem": spec.problem,
                "comparison_mode": spec.comparison_mode,
                "localiser_family": spec.localiser_family,
                "variant": spec.variant_name,
                "domain_label": spec.domain_label,
                "domain_value": spec.domain_value,
                "budget_label": spec.budget_label,
                "epochs": spec.epochs,
                "architecture_label": spec.architecture_label,
                "architecture": list(spec.architecture),
                "seed": spec.seed,
                "pair_group_relpath": spec.pair_group_relpath,
                "error": error_summary,
                "traceback": stack,
                "note": spec.note,
            }
            write_json(metrics_path, failure_payload)
            return {
                "status": "failed",
                "group_relpath": spec.group_relpath,
                "seed": spec.seed,
                "metrics_path": str(metrics_path),
                "error": error_summary,
            }


def build_manifest_groups(seed_specs: list[SeedRunSpec]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for spec in seed_specs:
        if spec.group_relpath in groups:
            continue
        groups[spec.group_relpath] = {
            "experiment": spec.experiment,
            "problem": spec.problem,
            "comparison_mode": spec.comparison_mode,
            "localiser_family": spec.localiser_family,
            "variant": spec.variant_name,
            "domain_label": spec.domain_label,
            "domain_value": spec.domain_value,
            "budget_label": spec.budget_label,
            "epochs": spec.epochs,
            "architecture_label": spec.architecture_label,
            "architecture": list(spec.architecture),
            "baseline_architecture_label": spec.baseline_architecture_label,
            "localised_architecture_label": spec.localised_architecture_label,
            "pair_group_relpath": spec.pair_group_relpath,
            "group_relpath": spec.group_relpath,
            "note": spec.note,
            "seeds": [],
        }
    for spec in seed_specs:
        groups[spec.group_relpath]["seeds"].append(spec.seed)
    return [groups[key] for key in sorted(groups)]


def should_run_experiment(experiment: ExperimentConfig, args: argparse.Namespace) -> bool:
    if args.problems and experiment.problem not in args.problems:
        return False
    return True


def selected_domain_items(problem: ProblemConfig, args: argparse.Namespace) -> list[tuple[str, float]]:
    if args.domains is None:
        return list(problem.domains.items())
    selected: list[tuple[str, float]] = []
    for token in args.domains:
        value = parse_domain_token(token)
        selected.append((domain_label_for_value(problem.domains, value), value))
    return selected


def selected_budget_items(problem: ProblemConfig, args: argparse.Namespace) -> list[tuple[str, int]]:
    if args.budgets is None:
        return list(problem.budgets.items())
    selected: list[tuple[str, int]] = []
    for token in args.budgets:
        value = parse_budget_token(token)
        selected.append((format_budget_label(value), value))
    return selected


def selected_architecture_labels(
    experiment: ExperimentConfig, problem: ProblemConfig, args: argparse.Namespace
) -> list[str]:
    if experiment.mode == "function":
        candidates = [format_architecture_label(problem.screen_architecture)]
    elif experiment.mode == "same":
        candidates = [format_architecture_label(arch) for arch in problem.same_network_architectures]
    else:
        candidates = [format_pair_label(*pair) for pair in problem.mixed_size_pairs]
    if args.architectures is None:
        return candidates
    requested = set(args.architectures)
    return [label for label in candidates if label in requested]


def build_seed_specs(args: argparse.Namespace) -> list[SeedRunSpec]:
    experiments = args.experiments or EXPERIMENT_ORDER
    families = args.families or LOCALISER_FAMILIES
    variants = args.variants or DEFAULT_VARIANTS
    seeds = args.seeds or list(range(1, 11))

    if args.smoke_test:
        experiments = ["ho_function_screen_v1"]
        families = ["gaussian"]
        variants = ["fixed"]
        seeds = [1]
        args.domains = ["pi"]
        args.budgets = ["20"]
        args.architectures = None

    specs: list[SeedRunSpec] = []
    for experiment_name in experiments:
        experiment = EXPERIMENT_CONFIGS[experiment_name]
        if not should_run_experiment(experiment, args):
            continue
        problem = PROBLEM_CONFIGS[experiment.problem]
        domain_items = selected_domain_items(problem, args)
        budget_items = selected_budget_items(problem, args)
        architecture_labels = selected_architecture_labels(experiment, problem, args)

        if experiment.mode == "function":
            arch = problem.screen_architecture
            arch_label = format_architecture_label(arch)
            if architecture_labels and arch_label not in architecture_labels:
                continue
            for domain_label, domain_value in domain_items:
                for budget_label, epochs in budget_items:
                    baseline_group_relpath = (
                        f"{experiment.name}/baseline/baseline_fc/domain_{domain_label}/"
                        f"epochs_{budget_label}/arch_{arch_label}"
                    )
                    for seed in seeds:
                        baseline_seed_relpath = f"{baseline_group_relpath}/seed_{seed:02d}"
                        specs.append(
                            SeedRunSpec(
                                experiment=experiment.name,
                                problem=problem.key,
                                comparison_mode="function_screen",
                                localiser_family="baseline",
                                variant_key="baseline",
                                variant_name="baseline_fc",
                                localisation=False,
                                mu_learnable=False,
                                sigma_learnable=False,
                                domain_label=domain_label,
                                domain_value=domain_value,
                                budget_label=budget_label,
                                epochs=epochs,
                                architecture_label=arch_label,
                                architecture=arch,
                                baseline_architecture_label=arch_label,
                                localised_architecture_label=arch_label,
                                pair_group_relpath=None,
                                group_relpath=baseline_group_relpath,
                                seed_relpath=baseline_seed_relpath,
                                activation=problem.activation,
                                seed=seed,
                                note="Function-screen baseline reference.",
                            )
                        )
                    for family in families:
                        for variant in variants:
                            mu_learnable = variant == "mu_sigma"
                            sigma_learnable = variant in {"sigma", "mu_sigma"}
                            variant_name = f"{family}_{variant}"
                            group_relpath = (
                                f"{experiment.name}/{family}/{variant_name}/domain_{domain_label}/"
                                f"epochs_{budget_label}/arch_{arch_label}"
                            )
                            for seed in seeds:
                                specs.append(
                                    SeedRunSpec(
                                        experiment=experiment.name,
                                        problem=problem.key,
                                        comparison_mode="function_screen",
                                        localiser_family=family,
                                        variant_key=variant,
                                        variant_name=variant_name,
                                        localisation=True,
                                        mu_learnable=mu_learnable,
                                        sigma_learnable=sigma_learnable,
                                        domain_label=domain_label,
                                        domain_value=domain_value,
                                        budget_label=budget_label,
                                        epochs=epochs,
                                        architecture_label=arch_label,
                                        architecture=arch,
                                        baseline_architecture_label=arch_label,
                                        localised_architecture_label=arch_label,
                                        pair_group_relpath=baseline_group_relpath,
                                        group_relpath=group_relpath,
                                        seed_relpath=f"{group_relpath}/seed_{seed:02d}",
                                        activation=problem.activation,
                                        seed=seed,
                                        note="Function-screen localised run.",
                                    )
                                )
        elif experiment.mode == "same":
            for arch in problem.same_network_architectures:
                arch_label = format_architecture_label(arch)
                if architecture_labels and arch_label not in architecture_labels:
                    continue
                for domain_label, domain_value in domain_items:
                    for budget_label, epochs in budget_items:
                        baseline_group_relpath = (
                            f"{experiment.name}/baseline/baseline_fc/domain_{domain_label}/"
                            f"epochs_{budget_label}/arch_{arch_label}"
                        )
                        for seed in seeds:
                            specs.append(
                                SeedRunSpec(
                                    experiment=experiment.name,
                                    problem=problem.key,
                                    comparison_mode="same_network",
                                    localiser_family="baseline",
                                    variant_key="baseline",
                                    variant_name="baseline_fc",
                                    localisation=False,
                                    mu_learnable=False,
                                    sigma_learnable=False,
                                    domain_label=domain_label,
                                    domain_value=domain_value,
                                    budget_label=budget_label,
                                    epochs=epochs,
                                    architecture_label=arch_label,
                                    architecture=arch,
                                    baseline_architecture_label=arch_label,
                                    localised_architecture_label=arch_label,
                                    pair_group_relpath=None,
                                    group_relpath=baseline_group_relpath,
                                    seed_relpath=f"{baseline_group_relpath}/seed_{seed:02d}",
                                    activation=problem.activation,
                                    seed=seed,
                                    note="Same-network baseline reference.",
                                )
                            )
                        for family in families:
                            for variant in variants:
                                mu_learnable = variant == "mu_sigma"
                                sigma_learnable = variant in {"sigma", "mu_sigma"}
                                variant_name = f"{family}_{variant}"
                                group_relpath = (
                                    f"{experiment.name}/{family}/{variant_name}/domain_{domain_label}/"
                                    f"epochs_{budget_label}/arch_{arch_label}"
                                )
                                for seed in seeds:
                                    specs.append(
                                        SeedRunSpec(
                                            experiment=experiment.name,
                                            problem=problem.key,
                                            comparison_mode="same_network",
                                            localiser_family=family,
                                            variant_key=variant,
                                            variant_name=variant_name,
                                            localisation=True,
                                            mu_learnable=mu_learnable,
                                            sigma_learnable=sigma_learnable,
                                            domain_label=domain_label,
                                            domain_value=domain_value,
                                            budget_label=budget_label,
                                            epochs=epochs,
                                            architecture_label=arch_label,
                                            architecture=arch,
                                            baseline_architecture_label=arch_label,
                                            localised_architecture_label=arch_label,
                                            pair_group_relpath=baseline_group_relpath,
                                            group_relpath=group_relpath,
                                            seed_relpath=f"{group_relpath}/seed_{seed:02d}",
                                            activation=problem.activation,
                                            seed=seed,
                                            note="Same-network localised run.",
                                        )
                                    )
        else:
            for baseline_arch, localised_arch in problem.mixed_size_pairs:
                pair_label = format_pair_label(baseline_arch, localised_arch)
                if architecture_labels and pair_label not in architecture_labels:
                    continue
                for domain_label, domain_value in domain_items:
                    for budget_label, epochs in budget_items:
                        baseline_group_relpath = (
                            f"{experiment.name}/baseline/baseline_fc/domain_{domain_label}/"
                            f"epochs_{budget_label}/arch_{pair_label}"
                        )
                        for seed in seeds:
                            specs.append(
                                SeedRunSpec(
                                    experiment=experiment.name,
                                    problem=problem.key,
                                    comparison_mode="mixed_size",
                                    localiser_family="baseline",
                                    variant_key="baseline",
                                    variant_name="baseline_fc",
                                    localisation=False,
                                    mu_learnable=False,
                                    sigma_learnable=False,
                                    domain_label=domain_label,
                                    domain_value=domain_value,
                                    budget_label=budget_label,
                                    epochs=epochs,
                                    architecture_label=pair_label,
                                    architecture=baseline_arch,
                                    baseline_architecture_label=format_architecture_label(baseline_arch),
                                    localised_architecture_label=format_architecture_label(localised_arch),
                                    pair_group_relpath=None,
                                    group_relpath=baseline_group_relpath,
                                    seed_relpath=f"{baseline_group_relpath}/seed_{seed:02d}",
                                    activation=problem.activation,
                                    seed=seed,
                                    note="Mixed-size baseline reference.",
                                )
                            )
                        for family in families:
                            for variant in variants:
                                mu_learnable = variant == "mu_sigma"
                                sigma_learnable = variant in {"sigma", "mu_sigma"}
                                variant_name = f"{family}_{variant}"
                                group_relpath = (
                                    f"{experiment.name}/{family}/{variant_name}/domain_{domain_label}/"
                                    f"epochs_{budget_label}/arch_{pair_label}"
                                )
                                for seed in seeds:
                                    specs.append(
                                        SeedRunSpec(
                                            experiment=experiment.name,
                                            problem=problem.key,
                                            comparison_mode="mixed_size",
                                            localiser_family=family,
                                            variant_key=variant,
                                            variant_name=variant_name,
                                            localisation=True,
                                            mu_learnable=mu_learnable,
                                            sigma_learnable=sigma_learnable,
                                            domain_label=domain_label,
                                            domain_value=domain_value,
                                            budget_label=budget_label,
                                            epochs=epochs,
                                            architecture_label=pair_label,
                                            architecture=localised_arch,
                                            baseline_architecture_label=format_architecture_label(baseline_arch),
                                            localised_architecture_label=format_architecture_label(localised_arch),
                                            pair_group_relpath=baseline_group_relpath,
                                            group_relpath=group_relpath,
                                            seed_relpath=f"{group_relpath}/seed_{seed:02d}",
                                            activation=problem.activation,
                                            seed=seed,
                                            note="Mixed-size localised run.",
                                        )
                                    )
    return specs


def load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def metric_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "stdev": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def determine_decision(localised_metrics: dict[str, Any], baseline_metrics: dict[str, Any] | None) -> tuple[str, str, str]:
    if baseline_metrics is None:
        return "baseline reference", "n/a", "pending"
    local_solution = localised_metrics["stats"]["solution_rmse"]["mean"]
    base_solution = baseline_metrics["stats"]["solution_rmse"]["mean"]
    local_time = localised_metrics["stats"]["wall_clock_seconds"]["mean"]
    base_time = baseline_metrics["stats"]["wall_clock_seconds"]["mean"]
    if local_solution is None or base_solution is None or local_time is None or base_time is None:
        return "insufficient successful seeds", "unknown", "review"
    better_quality = local_solution < base_solution * 0.98
    worse_quality = local_solution > base_solution * 1.02
    faster = local_time < base_time * 0.98
    slower = local_time > base_time * 1.02
    if better_quality and faster:
        return "won on time and quality", "cleaner", "carry_forward"
    if better_quality and not slower:
        return "won on quality", "cleaner", "carry_forward"
    if worse_quality and slower:
        return "slower and worse", "worse", "drop"
    if worse_quality:
        return "worse on quality", "worse", "drop"
    if faster and not worse_quality:
        return "won on time", "similar", "review"
    return "mixed result", "similar", "review"


def aggregate_group(
    batch_root: Path,
    group_manifest: dict[str, Any],
    seed_specs: list[SeedRunSpec],
) -> dict[str, Any]:
    group_dir = batch_root / group_manifest["group_relpath"]
    group_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    histories: list[np.ndarray] = []
    prediction_curves: list[np.ndarray] = []
    curve_x: np.ndarray | None = None
    reference_curve: np.ndarray | None = None

    for spec in seed_specs:
        metrics = load_metrics(batch_root / spec.seed_relpath / "metrics.json") or {
            "status": "missing",
            "seed": spec.seed,
        }
        metrics_rows.append(metrics)
        if metrics.get("status") != "completed":
            continue
        success_rows.append(metrics)
        history_path = batch_root / spec.seed_relpath / "loss_history.npy"
        slice_path = batch_root / spec.seed_relpath / "solution_slice.npz"
        if history_path.exists():
            histories.append(np.load(history_path))
        if slice_path.exists():
            bundle = np.load(slice_path, allow_pickle=True)
            prediction_curves.append(bundle["prediction_curve"])
            if curve_x is None:
                curve_x = bundle["curve_x"]
                reference_curve = bundle["reference_curve"]

    aggregate_csv_path = group_dir / "aggregate.csv"
    aggregate_json_path = group_dir / "aggregate.json"
    summary_md_path = group_dir / "summary.md"

    fieldnames = [
        "seed",
        "status",
        "wall_clock_seconds",
        "final_train_loss",
        "solution_rmse",
        "solution_mae",
        "residual_rmse",
        "error",
    ]
    with aggregate_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(metrics_rows, key=lambda item: item.get("seed", 0)):
            writer.writerow({name: row.get(name) for name in fieldnames})

    aggregate_payload = {
        "group": group_manifest,
        "counts": {
            "requested_seeds": len(seed_specs),
            "completed_seeds": sum(1 for row in metrics_rows if row.get("status") == "completed"),
            "failed_seeds": sum(1 for row in metrics_rows if row.get("status") == "failed"),
            "missing_seeds": sum(1 for row in metrics_rows if row.get("status") not in {"completed", "failed"}),
        },
        "stats": {
            "wall_clock_seconds": metric_stats([float(row["wall_clock_seconds"]) for row in success_rows]),
            "final_train_loss": metric_stats([float(row["final_train_loss"]) for row in success_rows]),
            "solution_rmse": metric_stats([float(row["solution_rmse"]) for row in success_rows]),
            "solution_mae": metric_stats([float(row["solution_mae"]) for row in success_rows]),
            "residual_rmse": metric_stats([float(row["residual_rmse"]) for row in success_rows]),
        },
    }

    baseline_metrics: dict[str, Any] | None = None
    if group_manifest["pair_group_relpath"]:
        baseline_aggregate = load_metrics(batch_root / group_manifest["pair_group_relpath"] / "aggregate.json")
        if baseline_aggregate:
            baseline_metrics = baseline_aggregate
            wins = {"time": 0, "solution_rmse": 0, "residual_rmse": 0, "paired_seeds": 0}
            baseline_rows = {
                row["seed"]: row
                for row in baseline_aggregate.get("per_seed", [])
                if row.get("status") == "completed"
            }
            per_seed_rows = []
            for row in success_rows:
                baseline_row = baseline_rows.get(row["seed"])
                if not baseline_row:
                    per_seed_rows.append(row)
                    continue
                wins["paired_seeds"] += 1
                if float(row["wall_clock_seconds"]) < float(baseline_row["wall_clock_seconds"]):
                    wins["time"] += 1
                if float(row["solution_rmse"]) < float(baseline_row["solution_rmse"]):
                    wins["solution_rmse"] += 1
                if float(row["residual_rmse"]) < float(baseline_row["residual_rmse"]):
                    wins["residual_rmse"] += 1
                per_seed_rows.append(row)
            aggregate_payload["wins_against_baseline"] = wins
    aggregate_payload["per_seed"] = metrics_rows

    if histories:
        combined_loss_path = group_dir / "combined_loss_plot.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        stacked = np.stack(histories)
        for series in histories:
            ax.plot(np.log10(np.maximum(series, 1e-30)), linewidth=0.8, alpha=0.35, color=PLOT_COLORS[1])
        ax.plot(
            np.log10(np.maximum(stacked.mean(axis=0), 1e-30)),
            linewidth=2.0,
            color=PLOT_COLORS[4],
            label="mean",
        )
        ax.set_title("Combined Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("log10(train_loss)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(combined_loss_path, dpi=180)
        plt.close(fig)

    if prediction_curves and curve_x is not None and reference_curve is not None:
        combined_solution_path = group_dir / "combined_solution_plot.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        curves = np.stack(prediction_curves)
        for curve in prediction_curves:
            ax.plot(curve_x, curve, linewidth=0.8, alpha=0.35, color=PLOT_COLORS[0])
        ax.plot(curve_x, curves.mean(axis=0), linewidth=2.0, color=PLOT_COLORS[4], label="mean prediction")
        ax.plot(curve_x, reference_curve, linewidth=1.8, color=PLOT_COLORS[1], label="reference")
        ax.set_title("Combined Solution Slice")
        ax.set_xlabel("domain")
        ax.set_ylabel("solution")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(combined_solution_path, dpi=180)
        plt.close(fig)

    decision, geometry, carry = determine_decision(aggregate_payload, baseline_metrics)
    summary_lines = [
        f"# {group_manifest['group_relpath']}",
        "",
        f"Tested: {group_manifest['experiment']} | {group_manifest['localiser_family']} | {group_manifest['variant']}",
        f"Domain: {group_manifest['domain_label']} | Budget: {group_manifest['budget_label']} | Architecture: {group_manifest['architecture_label']}",
        (
            "Seeds: {done}/{requested} completed, {failed} failed".format(
                done=aggregate_payload["counts"]["completed_seeds"],
                requested=aggregate_payload["counts"]["requested_seeds"],
                failed=aggregate_payload["counts"]["failed_seeds"],
            )
        ),
        f"Outcome: {decision}",
        f"Geometry: {geometry}",
        f"Carry forward: {carry}",
    ]
    if "wins_against_baseline" in aggregate_payload:
        wins = aggregate_payload["wins_against_baseline"]
        summary_lines.append(
            "Wins vs paired baseline: time {time}/{paired}, solution {solution}/{paired}, residual {residual}/{paired}".format(
                time=wins["time"],
                solution=wins["solution_rmse"],
                residual=wins["residual_rmse"],
                paired=wins["paired_seeds"],
            )
        )
    write_text(summary_md_path, "\n".join(summary_lines) + "\n")
    write_json(aggregate_json_path, aggregate_payload)
    return {
        "decision": decision,
        "geometry": geometry,
        "carry": carry,
        "counts": aggregate_payload["counts"],
    }


def initialise_group_state(
    batch_root: Path,
    seed_specs: list[SeedRunSpec],
) -> tuple[dict[str, dict[str, Any]], list[SeedRunSpec], list[SeedRunSpec]]:
    group_to_specs: dict[str, list[SeedRunSpec]] = defaultdict(list)
    for spec in seed_specs:
        group_to_specs[spec.group_relpath].append(spec)

    state: dict[str, dict[str, Any]] = {}
    queued: list[SeedRunSpec] = []
    preexisting: list[SeedRunSpec] = []
    for group_relpath, specs in group_to_specs.items():
        state[group_relpath] = {
            "requested": len(specs),
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "decision": None,
        }
        for spec in specs:
            metrics = load_metrics(batch_root / spec.seed_relpath / "metrics.json")
            if metrics and metrics.get("status") in {"completed", "failed"}:
                preexisting.append(spec)
                if metrics["status"] == "completed":
                    state[group_relpath]["completed"] += 1
                else:
                    state[group_relpath]["failed"] += 1
            else:
                queued.append(spec)
    return state, queued, preexisting


def write_top_level_summary(
    batch_root: Path,
    group_state: dict[str, dict[str, Any]],
    grouped_specs: dict[str, list[SeedRunSpec]],
) -> None:
    total_requested = sum(item["requested"] for item in group_state.values())
    total_completed = sum(item["completed"] for item in group_state.values())
    total_failed = sum(item["failed"] for item in group_state.values())
    total_done = total_completed + total_failed
    completed_groups = sum(
        1 for key, item in group_state.items() if item["completed"] + item["failed"] >= item["requested"]
    )

    summary_json = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {
            "requested_seed_runs": total_requested,
            "completed_seed_runs": total_completed,
            "failed_seed_runs": total_failed,
            "finished_seed_runs": total_done,
            "total_groups": len(group_state),
            "completed_groups": completed_groups,
        },
        "groups": group_state,
    }
    write_json(batch_root / "summary.json", summary_json)

    completed_lines: list[str] = []
    pending_lines: list[str] = []
    for group_relpath in sorted(group_state):
        item = group_state[group_relpath]
        line = (
            f"- {group_relpath}: {item['completed']}/{item['requested']} complete, "
            f"{item['failed']} failed"
        )
        if item.get("decision"):
            line += f" | {item['decision']}"
        if item["completed"] + item["failed"] >= item["requested"]:
            completed_lines.append(line)
        else:
            pending_lines.append(line)

    lines = [
        "# Campaign Summary",
        "",
        f"Updated: {summary_json['updated_at']}",
        f"Batch root: {batch_root}",
        (
            "Seed runs: {done}/{requested} finished | completed={completed} failed={failed}".format(
                done=total_done,
                requested=total_requested,
                completed=total_completed,
                failed=total_failed,
            )
        ),
        f"Groups: {completed_groups}/{len(group_state)} complete",
        "",
        "## Completed groups",
    ]
    lines.extend(completed_lines[:50] or ["- none yet"])
    lines.extend(["", "## Pending groups"])
    lines.extend(pending_lines[:50] or ["- none"])
    write_text(batch_root / "summary.md", "\n".join(lines) + "\n")


def group_is_complete(group_item: dict[str, Any]) -> bool:
    return group_item["completed"] + group_item["failed"] >= group_item["requested"]


def build_aggregate_dependents(
    manifest: dict[str, Any],
    ) -> dict[str, list[str]]:
    dependents: dict[str, list[str]] = defaultdict(list)
    for group in manifest["groups"]:
        pair_group = group.get("pair_group_relpath")
        if pair_group:
            dependents[pair_group].append(group["group_relpath"])
    return {key: sorted(value) for key, value in dependents.items()}


def request_group_aggregate(
    group_relpath: str,
    aggregation_executor: ThreadPoolExecutor,
    aggregation_futures: dict[str, Any],
    pending_reaggregates: set[str],
    batch_root: Path,
    manifest_groups: dict[str, dict[str, Any]],
    grouped_specs: dict[str, list[SeedRunSpec]],
) -> None:
    if group_relpath in aggregation_futures:
        pending_reaggregates.add(group_relpath)
        return
    aggregation_futures[group_relpath] = aggregation_executor.submit(
        aggregate_group,
        batch_root,
        manifest_groups[group_relpath],
        grouped_specs[group_relpath],
    )


def drain_group_aggregates(
    aggregation_executor: ThreadPoolExecutor,
    aggregation_futures: dict[str, Any],
    pending_reaggregates: set[str],
    batch_root: Path,
    manifest_groups: dict[str, dict[str, Any]],
    grouped_specs: dict[str, list[SeedRunSpec]],
    group_state: dict[str, dict[str, Any]],
    aggregate_dependents: dict[str, list[str]],
) -> bool:
    updated = False
    for group_relpath, future in list(aggregation_futures.items()):
        if not future.done():
            continue
        del aggregation_futures[group_relpath]
        aggregate_result = future.result()
        group_state[group_relpath]["decision"] = aggregate_result["decision"]
        updated = True
        print(
            "aggregated group={group} decision={decision}".format(
                group=group_relpath,
                decision=aggregate_result["decision"],
            ),
            flush=True,
        )
        for dependent_group in aggregate_dependents.get(group_relpath, []):
            if group_is_complete(group_state[dependent_group]):
                request_group_aggregate(
                    dependent_group,
                    aggregation_executor,
                    aggregation_futures,
                    pending_reaggregates,
                    batch_root,
                    manifest_groups,
                    grouped_specs,
                )
        if group_relpath in pending_reaggregates:
            pending_reaggregates.discard(group_relpath)
            request_group_aggregate(
                group_relpath,
                aggregation_executor,
                aggregation_futures,
                pending_reaggregates,
                batch_root,
                manifest_groups,
                grouped_specs,
            )
    return updated


def progress_line(group_state: dict[str, dict[str, Any]]) -> str:
    total_requested = sum(item["requested"] for item in group_state.values())
    total_completed = sum(item["completed"] for item in group_state.values())
    total_failed = sum(item["failed"] for item in group_state.values())
    finished_groups = sum(
        1 for item in group_state.values() if item["completed"] + item["failed"] >= item["requested"]
    )
    return (
        "seed_runs={done}/{requested} completed={completed} failed={failed} "
        "groups={finished}/{total_groups}".format(
            done=total_completed + total_failed,
            requested=total_requested,
            completed=total_completed,
            failed=total_failed,
            finished=finished_groups,
            total_groups=len(group_state),
        )
    )


def main() -> int:
    args = parse_args()
    requested_max_workers = args.max_workers
    args.max_workers = max(1, min(args.max_workers, 10))
    output_root = args.output_root or (ROOT / "campaign_runs" / "manual" / time.strftime("%Y%m%d_%H%M%S"))
    output_root.mkdir(parents=True, exist_ok=True)

    seed_specs = build_seed_specs(args)
    if not seed_specs:
        raise SystemExit("No seed runs were selected.")

    grouped_specs: dict[str, list[SeedRunSpec]] = defaultdict(list)
    for spec in seed_specs:
        grouped_specs[spec.group_relpath].append(spec)

    manifest_groups = {
        group["group_relpath"]: group
        for group in build_manifest_groups(seed_specs)
    }
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": str(output_root),
        "smoke_test": args.smoke_test,
        "requested_max_workers": requested_max_workers,
        "effective_max_workers": args.max_workers,
        "experiments": args.experiments or EXPERIMENT_ORDER,
        "problems": args.problems,
        "families": args.families or LOCALISER_FAMILIES,
        "variants": args.variants or DEFAULT_VARIANTS,
        "domains": args.domains,
        "budgets": args.budgets,
        "architectures": args.architectures,
        "seeds": args.seeds or list(range(1, 11)),
        "groups": list(manifest_groups.values()),
        "seed_runs": [asdict(spec) for spec in seed_specs],
    }
    write_json(output_root / "run_manifest.json", manifest)
    print(
        "manifest groups={groups} seed_runs={seed_runs} output_root={root}".format(
            groups=len(manifest["groups"]),
            seed_runs=len(seed_specs),
            root=output_root,
        ),
        flush=True,
    )
    if requested_max_workers != args.max_workers:
        print(
            f"requested_max_workers={requested_max_workers} capped_to={args.max_workers}",
            flush=True,
        )

    group_state, queued_specs, _ = initialise_group_state(output_root, seed_specs)
    aggregate_dependents = build_aggregate_dependents(manifest)
    with ThreadPoolExecutor(max_workers=1) as aggregation_executor:
        aggregation_futures: dict[str, Any] = {}
        pending_reaggregates: set[str] = set()
        for group_relpath, state in group_state.items():
            if group_is_complete(state):
                request_group_aggregate(
                    group_relpath,
                    aggregation_executor,
                    aggregation_futures,
                    pending_reaggregates,
                    output_root,
                    manifest_groups,
                    grouped_specs,
                )
        drain_group_aggregates(
            aggregation_executor,
            aggregation_futures,
            pending_reaggregates,
            output_root,
            manifest_groups,
            grouped_specs,
            group_state,
            aggregate_dependents,
        )
        write_top_level_summary(output_root, group_state, grouped_specs)
        print(f"initial {progress_line(group_state)} queued={len(queued_specs)}", flush=True)

        if queued_specs:
            max_workers = max(1, min(args.max_workers, len(queued_specs)))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_seed_worker, asdict(spec), str(output_root), args.overwrite): spec
                    for spec in queued_specs
                }
                for future in as_completed(futures):
                    spec = futures[future]
                    result = future.result()
                    group_item = group_state[spec.group_relpath]
                    if result["status"] in {"completed", "skipped"}:
                        group_item["completed"] += 1
                    elif result["status"] == "failed":
                        group_item["failed"] += 1
                        if args.fail_fast:
                            raise RuntimeError(result.get("error", "seed run failed"))

                    if group_is_complete(group_item):
                        request_group_aggregate(
                            spec.group_relpath,
                            aggregation_executor,
                            aggregation_futures,
                            pending_reaggregates,
                            output_root,
                            manifest_groups,
                            grouped_specs,
                        )
                    drain_group_aggregates(
                        aggregation_executor,
                        aggregation_futures,
                        pending_reaggregates,
                        output_root,
                        manifest_groups,
                        grouped_specs,
                        group_state,
                        aggregate_dependents,
                    )
                    write_top_level_summary(output_root, group_state, grouped_specs)
                    print(
                        "{status} group={group} seed={seed} {progress}".format(
                            status=result["status"],
                            group=spec.group_relpath,
                            seed=spec.seed,
                            progress=progress_line(group_state),
                        ),
                        flush=True,
                    )

        while aggregation_futures:
            if not drain_group_aggregates(
                aggregation_executor,
                aggregation_futures,
                pending_reaggregates,
                output_root,
                manifest_groups,
                grouped_specs,
                group_state,
                aggregate_dependents,
            ):
                time.sleep(0.2)

    write_top_level_summary(output_root, group_state, grouped_specs)
    print(f"finished {progress_line(group_state)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
