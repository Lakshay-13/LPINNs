#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentSpec:
    key: str
    group: str
    kind: str
    module_path: str | None
    class_name: str | None
    epochs: int
    seed: int
    kwargs: dict[str, Any]
    note: str = ""
    allow_failure: bool = False


ARCHITECTURE_EXPERIMENTS = [
    ExperimentSpec(
        key="ho_gaussian_single_final",
        group="architectures",
        kind="ho",
        module_path="HO/gaussian/Single Layer/harmonic_oscillator.py",
        class_name="HO",
        epochs=5000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            mu=False,
            sigma=False,
            input_neurons=[16, 256],
            lr=1e-2,
            activation="sine",
            initial_sigma=0.1,
            sigma_min=None,
            sigma_max=None,
            localization=True,
            scaling=True,
        ),
        note="HO Gaussian single-layer final-run style config",
    ),
    ExperimentSpec(
        key="ho_gaussian_multilayer_deep",
        group="architectures",
        kind="ho",
        module_path="HO/gaussian/Multiple Layers/harmonic_oscillator.py",
        class_name="HO",
        epochs=5000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            input_neurons=[16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
            lr=1e-2,
            activation="sine",
            sigma=True,
            initial_sigma=0.06,
            localization=True,
            scaling=True,
        ),
        note="HO Gaussian multi-layer deep final_run notebook config",
    ),
    ExperimentSpec(
        key="ho_gaussian_multilayer_wide",
        group="architectures",
        kind="ho",
        module_path="HO/gaussian/Multiple Layers/harmonic_oscillator.py",
        class_name="HO",
        epochs=5000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            input_neurons=[64, 64],
            lr=1e-2,
            activation="sine",
            sigma=True,
            initial_sigma=0.06,
            localization=True,
            scaling=True,
        ),
        note="HO Gaussian multi-layer wide final_run notebook config",
    ),
    ExperimentSpec(
        key="ho_activated_256",
        group="architectures",
        kind="ho",
        module_path="HO/activated localization/harmonic_oscillator.py",
        class_name="HO",
        epochs=5000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            input_neurons=[256],
            lr=1e-2,
            activation="sine",
            sigma=False,
            initial_sigma=0.06,
            localization=True,
            scaling=True,
        ),
        note="HO activated-localisation final notebook width 256",
    ),
    ExperimentSpec(
        key="ho_activated_512",
        group="architectures",
        kind="ho",
        module_path="HO/activated localization/harmonic_oscillator.py",
        class_name="HO",
        epochs=5000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            input_neurons=[512],
            lr=1e-2,
            activation="sine",
            sigma=False,
            initial_sigma=0.06,
            localization=True,
            scaling=True,
        ),
        note="HO activated-localisation final notebook width 512",
    ),
    ExperimentSpec(
        key="ho_activated_1024",
        group="architectures",
        kind="ho",
        module_path="HO/activated localization/harmonic_oscillator.py",
        class_name="HO",
        epochs=5000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            input_neurons=[1024],
            lr=1e-2,
            activation="sine",
            sigma=False,
            initial_sigma=0.06,
            localization=True,
            scaling=True,
        ),
        note="HO activated-localisation final notebook width 1024",
    ),
]

FUNCTION_EXPERIMENTS = [
    ExperimentSpec(
        key="ho_boxcar_baseline",
        group="functions",
        kind="ho",
        module_path="HO/boxcar/harmonic_oscillator.py",
        class_name="HO",
        epochs=5000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            mu=False,
            sigma=False,
            input_neurons=[32, 32],
            lr=1e-2,
            activation="sine",
            initial_sigma=0.06,
            sigma_min=None,
            sigma_max=None,
            localization=False,
            scaling=True,
            amplitude=1.0,
        ),
        note="HO boxcar longer-run baseline without localisation",
    ),
    ExperimentSpec(
        key="ho_boxcar_localised",
        group="functions",
        kind="ho",
        module_path="HO/boxcar/harmonic_oscillator.py",
        class_name="HO",
        epochs=5000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            mu=False,
            sigma=False,
            input_neurons=[32, 32],
            lr=1e-2,
            activation="sine",
            initial_sigma=0.06,
            sigma_min=None,
            sigma_max=None,
            localization=True,
            scaling=True,
            amplitude=1.0,
        ),
        note="HO boxcar longer-run localised branch",
    ),
    ExperimentSpec(
        key="ho_ricker_8_128",
        group="functions",
        kind="ho",
        module_path="HO/ricker wavelet/harmonic_oscillator.py",
        class_name="HO",
        epochs=50000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            sigma=True,
            input_neurons=[8, 128],
            lr=1e-2,
            activation="sine",
            initial_sigma=0.3,
            sigma_min=None,
            sigma_max=None,
            localization=True,
            scaling=True,
        ),
        note="HO ricker-wavelet initial_run notebook, [8, 128], 50k total epochs",
    ),
    ExperimentSpec(
        key="ho_ricker_16_256",
        group="functions",
        kind="ho",
        module_path="HO/ricker wavelet/harmonic_oscillator.py",
        class_name="HO",
        epochs=5000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            sigma=True,
            input_neurons=[16, 256],
            lr=1e-2,
            activation="sine",
            initial_sigma=0.3,
            sigma_min=None,
            sigma_max=None,
            localization=True,
            scaling=True,
        ),
        note="HO ricker-wavelet initial_run notebook, [16, 256], 5k total epochs",
    ),
    ExperimentSpec(
        key="ho_super_gaussian_1024x2",
        group="functions",
        kind="ho",
        module_path="HO/super gaussian/singe layer/harmonic_oscillator.py",
        class_name="HO",
        epochs=50000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            mu=False,
            sigma=False,
            input_neurons=[1024, 1024],
            lr=1e-2,
            activation="sine",
            initial_sigma=0.1,
            sigma_min=None,
            sigma_max=None,
            localization=True,
            scaling=True,
        ),
        note="HO super-Gaussian longer-run notebook, [1024, 1024]",
    ),
    ExperimentSpec(
        key="ho_super_gaussian_1024x3",
        group="functions",
        kind="ho",
        module_path="HO/super gaussian/singe layer/harmonic_oscillator.py",
        class_name="HO",
        epochs=50000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            mu=False,
            sigma=False,
            input_neurons=[1024, 1024, 1024],
            lr=1e-2,
            activation="sine",
            initial_sigma=0.1,
            sigma_min=None,
            sigma_max=None,
            localization=True,
            scaling=True,
        ),
        note="HO super-Gaussian longer-run notebook, [1024, 1024, 1024]",
    ),
    ExperimentSpec(
        key="ho_super_gaussian_16x3",
        group="functions",
        kind="ho",
        module_path="HO/super gaussian/singe layer/harmonic_oscillator.py",
        class_name="HO",
        epochs=50000,
        seed=3,
        kwargs=dict(
            t_0=0.0,
            u_0=0.0,
            u_0_prime=1.0,
            t_min=0.0,
            t_max=4 * math.pi,
            train_batch_size=256,
            valid_batch_size=64,
            mu=False,
            sigma=False,
            input_neurons=[16, 16, 16],
            lr=1e-2,
            activation="sine",
            initial_sigma=0.1,
            sigma_min=None,
            sigma_max=None,
            localization=True,
            scaling=True,
        ),
        note="HO super-Gaussian localization using the small-network longer-run config",
    ),
]

EQUATION_EXPERIMENTS = [
    ExperimentSpec(
        key="eq_heat_gaussian_128_localised",
        group="equations",
        kind="he",
        module_path="Heat Equation/Gaussian/heat_equation.py",
        class_name="HE",
        epochs=300000,
        seed=3,
        kwargs=dict(
            input_neurons=[128, 128],
            lr=1e-2,
            activation="tanh",
            sigma=True,
            initial_sigma=0.1,
            localization=True,
        ),
        note="Heat-equation Gaussian final notebook, [128, 128], localised",
    ),
    ExperimentSpec(
        key="eq_heat_gaussian_128_baseline",
        group="equations",
        kind="he",
        module_path="Heat Equation/Gaussian/heat_equation.py",
        class_name="HE",
        epochs=300000,
        seed=3,
        kwargs=dict(
            input_neurons=[128, 128],
            lr=1e-2,
            activation="tanh",
            sigma=True,
            initial_sigma=0.1,
            localization=False,
        ),
        note="Heat-equation Gaussian final notebook, [128, 128], no localisation",
    ),
    ExperimentSpec(
        key="eq_heat_gaussian_512_localised",
        group="equations",
        kind="he",
        module_path="Heat Equation/Gaussian/heat_equation.py",
        class_name="HE",
        epochs=300000,
        seed=3,
        kwargs=dict(
            input_neurons=[512, 512],
            lr=1e-2,
            activation="tanh",
            sigma=True,
            initial_sigma=0.1,
            localization=True,
        ),
        note="Heat-equation Gaussian final notebook, [512, 512], localised",
    ),
    ExperimentSpec(
        key="eq_heat_gaussian_512_baseline",
        group="equations",
        kind="he",
        module_path="Heat Equation/Gaussian/heat_equation.py",
        class_name="HE",
        epochs=300000,
        seed=3,
        kwargs=dict(
            input_neurons=[512, 512],
            lr=1e-2,
            activation="tanh",
            sigma=True,
            initial_sigma=0.1,
            localization=False,
        ),
        note="Heat-equation Gaussian final notebook, [512, 512], no localisation",
    ),
    ExperimentSpec(
        key="eq_heat_activated_trial",
        group="equations",
        kind="he",
        module_path="Heat Equation/Activated localization/heat_equation.py",
        class_name="HE",
        epochs=3000,
        seed=3,
        kwargs=dict(
            input_neurons=[16, 32],
            lr=1e-2,
            activation="tanh",
            mu=True,
            sigma=True,
            initial_sigma=0.1,
            localization=True,
        ),
        note="Activated heat-equation saved trial; known broken shape-mismatch branch",
        allow_failure=True,
    ),
    ExperimentSpec(
        key="eq_4d_run3",
        group="equations",
        kind="4d",
        module_path=None,
        class_name=None,
        epochs=150000,
        seed=3,
        kwargs={},
        note="4D Run 3 recreation with target-biased sampling",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Recreate localisation notebook experiments.")
    parser.add_argument("--group", choices=["architectures", "functions", "equations", "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--epoch-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--experiments", nargs="*", default=None, help="Optional subset of experiment keys.")
    parser.add_argument("--include-4d", action="store_true")
    parser.add_argument("--include-blocked", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--zip-at-end", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def select_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_torch(device: str, seed: int):
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    thread_count = os.environ.get("LOCALISATION_TORCH_THREADS")
    if thread_count:
        try:
            torch.set_num_threads(int(thread_count))
        except Exception:
            pass
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_default_device(device)
    except Exception:
        pass


def load_module(relative_path: str):
    module_path = ROOT / relative_path
    module_name = "loc_" + "_".join(module_path.parts[-4:]).replace(" ", "_").replace("-", "_").replace(".", "_")
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
        root = ROOT / "repro_runs" / stamp
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    return root


def scaled_epochs(spec: ExperimentSpec, scale: float) -> int:
    return max(1, int(round(spec.epochs * scale)))


def write_json(path: Path, payload: dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def resolve_kwargs(raw_kwargs: dict[str, Any]):
    import torch
    import torch.nn as nn

    class SineActivation(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    kwargs = dict(raw_kwargs)
    activation = kwargs.get("activation")
    if activation == "sine":
        kwargs["activation"] = SineActivation
    elif activation == "tanh":
        kwargs["activation"] = nn.Tanh
    return kwargs


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


def save_ho_artifacts(module, solver, exp_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    t_vals = np.linspace(solver.t_min, solver.t_max, 1000)
    pred = solver.get_solution()(t_vals, to_numpy=True)
    residuals = solver.get_residuals(t_vals, to_numpy=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_vals, pred)
    ax.set_title("HO solution")
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(exp_dir / "solution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_vals, residuals)
    ax.set_title("HO residuals")
    ax.set_xlabel("t")
    ax.set_ylabel("Residual")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(exp_dir / "residuals.png", dpi=180)
    plt.close(fig)

    return {
        "residual_abs_max": float(np.max(np.abs(residuals))),
        "residual_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
    }


def save_he_artifacts(module, solver, exp_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    x, y = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
    pred = solver.get_solution(best=True)(x, y, to_numpy=True)
    truth = module.he(x, y)
    residuals = solver.get_residuals(x, y, to_numpy=True)

    for name, data in [("prediction.png", pred), ("truth.png", truth), ("residuals.png", residuals)]:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(name.replace(".png", "").replace("_", " ").title())
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(exp_dir / name, dpi=180)
        plt.close(fig)

    return {
        "prediction_mse": float(np.mean(np.square(pred - truth))),
        "residual_abs_max": float(np.max(np.abs(residuals))),
        "residual_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
    }


def build_4d_solver():
    from copy import deepcopy

    import numpy as np
    import torch
    from neurodiffeq import diff
    from neurodiffeq.generators import BaseGenerator
    from neurodiffeq.solvers import BaseSolution, BaseSolver
    from neurodiffeq_conditions.conditions import ComposedCondition, ConditionComponent
    from scipy.stats import qmc, truncnorm

    lnn_module = load_module("4D/lnn.py")
    CustomNN = lnn_module.CustomNN

    class BundleSolution4D(BaseSolution):
        def _compute_u(self, net, condition, xs1, xs2, xs3, ys1, *ts):
            return condition.enforce(net, xs1, xs2, xs3, ys1, *ts)

    class BundleIntSolver4D(BaseSolver):
        def __init__(
            self,
            pde_system,
            conditions,
            xy_min,
            xy_max,
            theta_min=None,
            theta_max=None,
            eq_param_index=(),
            nets=None,
            train_generator=None,
            valid_generator=None,
            analytic_solutions=None,
            optimizer=None,
            loss_fn=None,
            n_batches_train=1,
            n_batches_valid=4,
            metrics=None,
            n_output_units=1,
            batch_size=None,
            shuffle=None,
        ):
            if isinstance(theta_min, (float, int)):
                theta_min = (theta_min,)
            elif theta_min is None:
                theta_min = ()

            if isinstance(theta_max, (float, int)):
                theta_max = (theta_max,)
            elif theta_max is None:
                theta_max = ()

            r_min = tuple(xy_min) + tuple(theta_min)
            r_max = tuple(xy_max) + tuple(theta_max)
            self.r_min = r_min
            self.r_max = r_max
            self.a_min = torch.tensor([])
            self.a_max = torch.tensor([])
            n_functions = len(conditions)
            n_coords = 4
            eq_param_index = tuple(n_functions + n_coords + idx for idx in eq_param_index)
            self.eq_param_index = eq_param_index

            def _diff_eqs_wrapper(*variables):
                funcs_and_coords = variables[: n_functions + n_coords]
                eq_params = tuple(variables[idx] for idx in eq_param_index)
                return pde_system(*funcs_and_coords, *eq_params)

            super().__init__(
                diff_eqs=_diff_eqs_wrapper,
                conditions=conditions,
                nets=nets,
                train_generator=train_generator,
                valid_generator=valid_generator,
                analytic_solutions=analytic_solutions,
                optimizer=optimizer,
                loss_fn=loss_fn,
                n_batches_train=n_batches_train,
                n_batches_valid=n_batches_valid,
                metrics=metrics,
                n_input_units=len(r_min),
                n_output_units=n_output_units,
                shuffle=shuffle,
                batch_size=batch_size,
            )

        def additional_loss(self, residual, funcs, coords):
            u = self.get_solution(best=False, copy=False)
            x1, x2, x3, y1 = coords[0], coords[1], coords[2], coords[3]
            in_val_x2 = torch.full((x2.size()[0],), self.r_min[1], device=x2.device)
            te_min = [param * torch.ones(x1.size()[0], device=x1.device) for param in self.r_min[4:]]
            x2_term = u(x1, in_val_x2, x3, y1, *te_min) ** 2
            dux1 = diff(x2_term.view(-1, 1), x1) ** 2
            d2ux2 = diff(dux1.view(-1, 1), x2) ** 2
            d3ux3 = diff(d2ux2.view(-1, 1), x3) ** 2
            return torch.mean(dux1 + d2ux2 + d3ux3)

        def get_solution(self, copy=True, best=True):
            nets = self.best_nets if best else self.nets
            conditions = self.conditions
            if copy:
                nets = deepcopy(nets)
                conditions = deepcopy(conditions)
            return BundleSolution4D(nets, conditions)

    class LatinHypercubeGeneratorNDbisp(BaseGenerator):
        def __init__(
            self,
            nsamp=2048,
            ndim=2,
            r_min=(0.0, 0.0),
            r_max=(1.0, 1.0),
            target_point=(0.5, 0.5),
            target_point_std=(0.1, 0.1),
            target_point_frac=0.0,
        ):
            super().__init__()
            self.size = nsamp
            sampler = qmc.LatinHypercube(d=ndim)
            self.getter = lambda: tuple(
                torch.cat(
                    (
                        torch.tensor(
                            r_min[i]
                            + sampler.random(n=round(nsamp * (1.0 - target_point_frac)))[:, i] * (r_max[i] - r_min[i]),
                            requires_grad=True,
                        ),
                        torch.tensor(
                            truncnorm.rvs(
                                a=(r_min[i] - target_point[i]) / target_point_std[i],
                                b=(r_max[i] - target_point[i]) / target_point_std[i],
                                loc=target_point[i],
                                scale=target_point_std[i],
                                size=round(nsamp * target_point_frac),
                            ),
                            requires_grad=True,
                        ),
                    )
                )
                for i in range(ndim)
            )

        def get_examples(self):
            return self.getter()

    order = 4
    mins = (0.0, 0.0, 0.0, 0.0)
    maxs = (1.0, 1.0, 1.0, 1.0)
    target_point = (0.8, 0.8, 0.8, 0.8)
    target_point_std = (0.01, 0.01, 0.01, 0.01)
    nets = [CustomNN(n_input_units=order, n_output_units=1, hidden_units=[32, 16, 16, 8, 8], first=False)]

    def zero_condition(*coords):
        return 0

    components = [ConditionComponent(w=-1e-6 * i, f_dirichlet=zero_condition, coord_index=i) for i in range(order)]
    composed_condition = ComposedCondition(components=components)
    equation = lambda u, kpa, kpb, kpc, kppa: [diff(diff(diff(diff(u, kpa), kpb), kpc), kppa) - 10.0]

    genhyp = LatinHypercubeGeneratorNDbisp(
        nsamp=4096,
        ndim=order,
        r_min=mins,
        r_max=maxs,
        target_point=target_point,
        target_point_std=target_point_std,
        target_point_frac=0.3,
    )

    solver = BundleIntSolver4D(
        pde_system=equation,
        conditions=[composed_condition],
        xy_min=mins,
        xy_max=maxs,
        nets=nets,
        train_generator=genhyp,
        valid_generator=genhyp,
    )
    return solver


def save_4d_artifacts(solver, exp_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.linspace(0.0, 1.0, 8)
    mesh = np.stack(np.meshgrid(grid, grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 4)
    pred = solver.get_solution(best=True)(
        mesh[:, 0],
        mesh[:, 1],
        mesh[:, 2],
        mesh[:, 3],
        to_numpy=True,
    )
    truth = 10.0 * mesh[:, 0] * mesh[:, 1] * mesh[:, 2] * mesh[:, 3]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(truth, pred, s=8, alpha=0.6)
    ax.set_title("4D predicted vs analytic")
    ax.set_xlabel("Analytic")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(exp_dir / "pred_vs_truth.png", dpi=180)
    plt.close(fig)

    return {
        "sample_mse": float(np.mean(np.square(pred - truth))),
        "sample_mae": float(np.mean(np.abs(pred - truth))),
    }


def run_experiment(spec: ExperimentSpec, output_root: Path, device: str, epoch_scale: float):
    import matplotlib.pyplot as plt

    exp_dir = output_root / "artifacts" / spec.key
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "logs" / f"{spec.key}.log"
    epochs = scaled_epochs(spec, epoch_scale)

    summary = {
        "key": spec.key,
        "group": spec.group,
        "kind": spec.kind,
        "epochs_requested": spec.epochs,
        "epochs_run": epochs,
        "seed": spec.seed,
        "device": device,
        "note": spec.note,
        "allow_failure": spec.allow_failure,
        "status": "failed",
        "log_path": str(log_path.relative_to(output_root)),
        "artifact_dir": str(exp_dir.relative_to(output_root)),
    }

    start = time.time()
    with log_path.open("w") as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
        print(f"=== {spec.key} ===")
        print(json.dumps({"config": spec.kwargs, "epochs": epochs, "device": device}, default=str, indent=2))
        try:
            configure_torch(device, spec.seed)
            plt.ioff()

            if spec.kind == "4d":
                solver = build_4d_solver()
                solver.fit(max_epochs=epochs)
                extra_metrics = save_4d_artifacts(solver, exp_dir)
            else:
                module = load_module(spec.module_path)
                cls = getattr(module, spec.class_name)
                runner = cls(**resolve_kwargs(spec.kwargs))
                runner.solver.fit(max_epochs=epochs)
                solver = runner.solver
                if spec.kind == "ho":
                    extra_metrics = save_ho_artifacts(module, solver, exp_dir)
                else:
                    extra_metrics = save_he_artifacts(module, solver, exp_dir)

            train_loss = [float(x) for x in solver.metrics_history["train_loss"]]
            save_loss_plot(train_loss, exp_dir / "loss.png", spec.key)
            summary.update(
                {
                    "status": "ok",
                    "elapsed_sec": time.time() - start,
                    "final_train_loss": float(train_loss[-1]),
                    "min_train_loss": float(min(train_loss)),
                }
            )
            summary.update(extra_metrics)
        except Exception as exc:
            traceback.print_exc()
            summary.update(
                {
                    "status": "failed",
                    "elapsed_sec": time.time() - start,
                    "error": repr(exc),
                }
            )

    write_json(exp_dir / "summary.json", summary)
    return summary


def choose_experiments(group: str, selected: list[str] | None, include_4d: bool, include_blocked: bool):
    experiments = []
    if group in {"architectures", "all"}:
        experiments.extend(ARCHITECTURE_EXPERIMENTS)
    if group in {"functions", "all"}:
        experiments.extend(FUNCTION_EXPERIMENTS)
    if group in {"equations", "all"}:
        for exp in EQUATION_EXPERIMENTS:
            if exp.key == "eq_4d_run3" and not include_4d:
                continue
            if exp.allow_failure and not include_blocked:
                continue
            experiments.append(exp)

    if selected:
        selected_set = set(selected)
        experiments = [exp for exp in experiments if exp.key in selected_set]
    return experiments


def summarize_results(results: list[dict[str, Any]]):
    return {
        "total": len(results),
        "ok": sum(1 for row in results if row.get("status") == "ok"),
        "failures": sum(1 for row in results if row.get("status") != "ok" and not row.get("allow_failure", False)),
        "allowed_failures": sum(1 for row in results if row.get("status") != "ok" and row.get("allow_failure", False)),
    }


def write_summary(outputs_root: Path, results: list[dict[str, Any]]):
    write_json(outputs_root / "summary.json", {"counts": summarize_results(results), "results": results})
    csv_path = outputs_root / "summary.csv"
    keys = sorted({key for row in results for key in row.keys()})
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def make_zip(output_root: Path):
    zip_path = output_root.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(output_root.parent))
    return zip_path


def execute_experiments(
    experiments: list[ExperimentSpec],
    output_root: Path,
    device: str,
    epoch_scale: float,
    max_workers: int,
    fail_fast: bool,
):
    if max_workers <= 1:
        results = []
        failures = 0
        for spec in experiments:
            result = run_experiment(spec, output_root, device, epoch_scale)
            results.append(result)
            if result["status"] != "ok" and not result.get("allow_failure", False):
                failures += 1
                if fail_fast:
                    break
        return results, failures

    results_by_index: dict[int, dict[str, Any]] = {}
    failures = 0
    stop_requested = False

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(run_experiment, spec, output_root, device, epoch_scale): idx
            for idx, spec in enumerate(experiments)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            spec = experiments[idx]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "key": spec.key,
                    "group": spec.group,
                    "kind": spec.kind,
                    "epochs_requested": spec.epochs,
                    "epochs_run": scaled_epochs(spec, epoch_scale),
                    "seed": spec.seed,
                    "device": device,
                    "note": spec.note,
                    "allow_failure": spec.allow_failure,
                    "status": "failed",
                    "error": repr(exc),
                    "artifact_dir": str((output_root / "artifacts" / spec.key).relative_to(output_root)),
                    "log_path": str((output_root / "logs" / f"{spec.key}.log").relative_to(output_root)),
                }

            results_by_index[idx] = result
            if result["status"] != "ok" and not result.get("allow_failure", False):
                failures += 1
                if fail_fast and not stop_requested:
                    stop_requested = True
                    for pending in future_to_index:
                        if pending is not future:
                            pending.cancel()

    ordered_results = [results_by_index[idx] for idx in sorted(results_by_index)]
    return ordered_results, failures


def main():
    args = parse_args()
    experiments = choose_experiments(args.group, args.experiments, args.include_4d, args.include_blocked)
    if args.list:
        for spec in experiments:
            flags = " allow-failure" if spec.allow_failure else ""
            print(f"{spec.group:14s} {spec.key:32s} epochs={spec.epochs:<7d} kind={spec.kind}{flags}")
        return 0

    if not experiments:
        print("No experiments selected.", file=sys.stderr)
        return 1

    device = select_device(args.device)
    output_root = ensure_output_root(args.output_root)
    max_workers = max(1, args.max_workers)

    manifest = {
        "device": device,
        "epoch_scale": args.epoch_scale,
        "max_workers": max_workers,
        "include_4d": args.include_4d,
        "include_blocked": args.include_blocked,
        "experiments": [asdict(spec) for spec in experiments],
    }
    write_json(output_root / "manifest.json", manifest)

    results, failures = execute_experiments(
        experiments=experiments,
        output_root=output_root,
        device=device,
        epoch_scale=args.epoch_scale,
        max_workers=max_workers,
        fail_fast=args.fail_fast,
    )
    write_summary(output_root, results)
    counts = summarize_results(results)

    if args.zip_at_end:
        zip_path = make_zip(output_root)
        print(f"archive={zip_path}")

    print(f"output_root={output_root}")
    print(f"ok={counts['ok']}")
    print(f"failures={counts['failures']}")
    print(f"allowed_failures={counts['allowed_failures']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
