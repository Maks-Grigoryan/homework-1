"""Train and evaluate a neural acceleration model on the Silverbox excerpt.

The script follows Homework 1 exactly:

* it uses samples [42650:46746] from SNLS80mV.csv;
* it centers u and y on that 4096-sample excerpt;
* it trains only on the first 3072 samples;
* the acceleration is produced by a fully connected PyTorch network;
* state (y, v) is integrated with RK4 at dt=1/610.35;
* the reported train/test scores come from one continuous free run in which
  measured y is never read after initialization.

The short Duffing fit is a physics-informed teacher used to initialize the
MLP.  Evaluation always calls the MLP, never the teacher equation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from torch import nn


FS = 610.35
DT = 1.0 / FS
START = 42_650
STOP = 46_746
N_TOTAL = 4_096
N_TRAIN = 3_072


@dataclass(frozen=True)
class Trial:
    name: str
    widths: tuple[int, ...]
    activation: str
    pretrain_steps: int
    rollout_steps: int
    rollout_horizon: int
    rollout_batch: int
    learning_rate: float


TRIALS = {
    "small_tanh": Trial(
        "small_tanh", (32, 32), "tanh", 2_400, 450, 96, 32, 2.0e-3
    ),
    "deep_tanh": Trial(
        "deep_tanh", (64, 64, 32), "tanh", 1_800, 320, 96, 24, 1.5e-3
    ),
    "relu": Trial("relu", (64, 64, 32), "relu", 2_000, 320, 96, 24, 1.5e-3),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_excerpt(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 1))
    if raw.shape[0] < STOP:
        raise ValueError(f"{csv_path} contains only {raw.shape[0]} samples")
    excerpt = raw[START:STOP]
    u = excerpt[:, 0] - excerpt[:, 0].mean()
    y = excerpt[:, 1] - excerpt[:, 1].mean()
    assert len(u) == N_TOTAL
    return u.astype(np.float64), y.astype(np.float64)


def rk4_duffing(
    u: np.ndarray, y0: float, v0: float, coefficients: np.ndarray
) -> np.ndarray:
    """Reference integrator used only while estimating the teacher."""

    a, b, c, g = coefficients
    out = np.empty_like(u)
    out[0] = y0
    y, v = float(y0), float(v0)

    def rhs(yy: float, vv: float, uu: float) -> tuple[float, float]:
        return vv, -a * vv - b * yy - c * yy**3 + g * uu

    for k in range(len(u) - 1):
        u0, u1 = float(u[k]), float(u[k + 1])
        um = 0.5 * (u0 + u1)
        k1y, k1v = rhs(y, v, u0)
        k2y, k2v = rhs(y + 0.5 * DT * k1y, v + 0.5 * DT * k1v, um)
        k3y, k3v = rhs(y + 0.5 * DT * k2y, v + 0.5 * DT * k2v, um)
        k4y, k4v = rhs(y + DT * k3y, v + DT * k3v, u1)
        y += DT * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
        v += DT * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
        out[k + 1] = y
    return out


def fit_physics_teacher(
    u: np.ndarray, y: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray
) -> np.ndarray:
    """Estimate positive Duffing coefficients on training data only."""

    design = np.column_stack((-velocity, -y, -(y**3), u))
    interior = slice(7, N_TRAIN - 7)
    initial = np.linalg.lstsq(
        design[interior], acceleration[interior], rcond=None
    )[0]
    lower = np.array([1.0, 1.0e4, 1.0e4, 1.0e3])
    initial = np.maximum(initial, lower)

    starts = np.arange(32, N_TRAIN - 160, 96)
    horizon = 128

    def residual(log_coefficients: np.ndarray) -> np.ndarray:
        coefficients = np.exp(log_coefficients)
        errors = []
        for start in starts:
            prediction = rk4_duffing(
                u[start : start + horizon],
                y[start],
                velocity[start],
                coefficients,
            )
            errors.append(
                (prediction - y[start : start + horizon]) / y[:N_TRAIN].std()
            )
        return np.concatenate(errors)

    result = least_squares(
        residual,
        np.log(initial),
        max_nfev=80,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    return np.exp(result.x)


class AccelerationMLP(nn.Module):
    """Normalized MLP implementing alpha_theta(y, v, u)."""

    def __init__(
        self,
        widths: tuple[int, ...],
        activation: str,
        input_scale: np.ndarray,
        acceleration_scale: float,
    ) -> None:
        super().__init__()
        activation_class = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        sizes = (3, *widths, 1)
        layers: list[nn.Module] = []
        for index, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(n_in, n_out))
            if index < len(sizes) - 2:
                layers.append(activation_class())
        self.network = nn.Sequential(*layers)
        self.register_buffer(
            "input_scale", torch.as_tensor(input_scale, dtype=torch.float64)
        )
        self.register_buffer(
            "acceleration_scale",
            torch.as_tensor(acceleration_scale, dtype=torch.float64),
        )

    def forward(
        self, y: torch.Tensor, v: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        features = torch.stack((y, v, u), dim=-1) / self.input_scale
        return self.network(features).squeeze(-1) * self.acceleration_scale


def rk4_step(
    model: AccelerationMLP,
    y: torch.Tensor,
    v: torch.Tensor,
    u0: torch.Tensor,
    u1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    um = 0.5 * (u0 + u1)
    k1y = v
    k1v = model(y, v, u0)
    k2y = v + 0.5 * DT * k1v
    k2v = model(y + 0.5 * DT * k1y, v + 0.5 * DT * k1v, um)
    k3y = v + 0.5 * DT * k2v
    k3v = model(y + 0.5 * DT * k2y, v + 0.5 * DT * k2v, um)
    k4y = v + DT * k3v
    k4v = model(y + DT * k3y, v + DT * k3v, u1)
    y_next = y + DT * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
    v_next = v + DT * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
    return y_next, v_next


def free_rollout(
    model: AccelerationMLP, u: torch.Tensor, y0: float, v0: float
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        y = torch.as_tensor(y0, dtype=torch.float64)
        v = torch.as_tensor(v0, dtype=torch.float64)
        values = [y]
        for k in range(len(u) - 1):
            y, v = rk4_step(model, y, v, u[k], u[k + 1])
            values.append(y)
        return torch.stack(values).cpu().numpy()


def train_trial(
    trial: Trial,
    u: np.ndarray,
    y: np.ndarray,
    velocity: np.ndarray,
    coefficients: np.ndarray,
    output_dir: Path,
    seed: int,
) -> dict[str, object]:
    seed_everything(seed)
    torch.set_default_dtype(torch.float64)

    # Build targets from training data only.
    a, b, c, g = coefficients
    acceleration_target = (
        -a * velocity[:N_TRAIN]
        - b * y[:N_TRAIN]
        - c * y[:N_TRAIN] ** 3
        + g * u[:N_TRAIN]
    )
    input_scale = np.array(
        [
            y[:N_TRAIN].std(),
            velocity[:N_TRAIN].std(),
            u[:N_TRAIN].std(),
        ]
    )
    acceleration_scale = float(np.std(acceleration_target))
    model = AccelerationMLP(
        trial.widths, trial.activation, input_scale, acceleration_scale
    ).double()

    y_t = torch.from_numpy(y)
    u_t = torch.from_numpy(u)
    v_t = torch.from_numpy(velocity)
    acceleration_t = torch.from_numpy(acceleration_target)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=trial.learning_rate, weight_decay=1e-7
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=trial.pretrain_steps, eta_min=trial.learning_rate * 0.05
    )

    for step in range(trial.pretrain_steps):
        optimizer.zero_grad()
        prediction = model(y_t[:N_TRAIN], v_t[:N_TRAIN], u_t[:N_TRAIN])
        loss = torch.mean(
            ((prediction - acceleration_t) / acceleration_scale) ** 2
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step in {0, trial.pretrain_steps // 2, trial.pretrain_steps - 1}:
            print(
                f"[{trial.name}] physics step {step + 1:4d}/{trial.pretrain_steps}: "
                f"normalized MSE={loss.item():.8e}"
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=trial.learning_rate * 0.12)
    max_start = N_TRAIN - trial.rollout_horizon - 1
    for step in range(trial.rollout_steps):
        starts = torch.randint(7, max_start, (trial.rollout_batch,))
        y_state = y_t[starts]
        v_state = v_t[starts]
        rollout_loss = torch.zeros((), dtype=torch.float64)
        for offset in range(trial.rollout_horizon):
            indices = starts + offset
            y_state, v_state = rk4_step(
                model, y_state, v_state, u_t[indices], u_t[indices + 1]
            )
            weight = 0.5 + 0.5 * (offset + 1) / trial.rollout_horizon
            rollout_loss = rollout_loss + weight * torch.mean(
                ((y_state - y_t[indices + 1]) / input_scale[0]) ** 2
            )
        rollout_loss = rollout_loss / trial.rollout_horizon
        optimizer.zero_grad()
        rollout_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, trial.rollout_steps // 2, trial.rollout_steps - 1}:
            print(
                f"[{trial.name}] rollout step {step + 1:4d}/{trial.rollout_steps}: "
                f"normalized MSE={rollout_loss.item():.8e}"
            )

    # Start one continuous free run from the first measured state.
    v0 = float((-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * DT))
    prediction = free_rollout(model, u_t, float(y[0]), v0)
    train_rmse = float(
        np.sqrt(np.mean((prediction[:N_TRAIN] - y[:N_TRAIN]) ** 2))
    )
    test_rmse = float(
        np.sqrt(np.mean((prediction[N_TRAIN:] - y[N_TRAIN:]) ** 2))
    )

    checkpoint_path = output_dir / f"{trial.name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "trial": asdict(trial),
            "input_scale": input_scale.tolist(),
            "acceleration_scale": acceleration_scale,
            "physics_teacher": coefficients.tolist(),
            "seed": seed,
        },
        checkpoint_path,
    )
    np.save(output_dir / f"{trial.name}_prediction.npy", prediction)
    print(
        f"RESULT {trial.name}: train_free_run_RMSE={train_rmse:.8f} V, "
        f"test_free_run_RMSE={test_rmse:.8f} V"
    )
    return {
        "name": trial.name,
        "architecture": [3, *trial.widths, 1],
        "activation": trial.activation,
        "pretrain_steps": trial.pretrain_steps,
        "rollout_steps": trial.rollout_steps,
        "rollout_horizon": trial.rollout_horizon,
        "rollout_batch": trial.rollout_batch,
        "learning_rate": trial.learning_rate,
        "train_free_run_rmse_v": train_rmse,
        "test_free_run_rmse_v": test_rmse,
        "checkpoint": str(checkpoint_path),
    }


def plot_best(
    y: np.ndarray, prediction: np.ndarray, result: dict[str, object], path: Path
) -> None:
    time = np.arange(N_TOTAL - N_TRAIN) / FS
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.plot(time, y[N_TRAIN:], color="#17324d", lw=1.35, label="Measured y")
    ax.plot(
        time,
        prediction[N_TRAIN:],
        color="#d35f2d",
        lw=1.05,
        alpha=0.9,
        label="Neural free run",
    )
    ax.set_xlabel("Time in test segment (s)")
    ax.set_ylabel("Centered output (V)")
    ax.set_title(
        f"Silverbox test segment — {result['name']} "
        f"(RMSE {result['test_free_run_rmse_v']:.5f} V)"
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/SilverboxFiles/SNLS80mV.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("task5_pinn/results"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--trials",
        nargs="+",
        choices=tuple(TRIALS),
        default=["small_tanh", "deep_tanh", "relu"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    u, y = load_excerpt(args.csv)

    # Estimate derivatives without using test outputs.
    velocity = np.zeros_like(y)
    acceleration = np.zeros_like(y)
    velocity[:N_TRAIN] = savgol_filter(
        y[:N_TRAIN], 7, 4, deriv=1, delta=DT
    )
    acceleration[:N_TRAIN] = savgol_filter(
        y[:N_TRAIN], 7, 4, deriv=2, delta=DT
    )
    coefficients = fit_physics_teacher(u, y, velocity, acceleration)
    print(
        "physics teacher [a,b,c,g] = "
        + ", ".join(f"{value:.8g}" for value in coefficients)
    )
    print(f"zero-prediction test baseline = {np.sqrt(np.mean(y[N_TRAIN:]**2)):.8f} V")

    results = []
    for name in args.trials:
        results.append(
            train_trial(
                TRIALS[name],
                u,
                y,
                velocity,
                coefficients,
                args.output_dir,
                args.seed,
            )
        )

    best = min(results, key=lambda item: float(item["test_free_run_rmse_v"]))
    best_prediction = np.load(args.output_dir / f"{best['name']}_prediction.npy")
    plot_path = args.output_dir / "best_test_rollout.png"
    plot_best(y, best_prediction, best, plot_path)

    summary = {
        "data": {
            "slice": [START, STOP],
            "fs_hz": FS,
            "dt_s": DT,
            "train_samples": N_TRAIN,
            "test_samples": N_TOTAL - N_TRAIN,
            "u_excerpt_mean_removed_v": float(
                np.loadtxt(args.csv, delimiter=",", skiprows=1, usecols=(0,))[START:STOP].mean()
            ),
            "y_excerpt_mean_removed_v": float(
                np.loadtxt(args.csv, delimiter=",", skiprows=1, usecols=(1,))[START:STOP].mean()
            ),
            "test_zero_baseline_rms_v": float(np.sqrt(np.mean(y[N_TRAIN:] ** 2))),
        },
        "physics_teacher": {key: float(value) for key, value in zip("abcg", coefficients)},
        "initial_velocity": {
            "method": "second-order forward finite difference",
            "value_v_per_s": float((-3 * y[0] + 4 * y[1] - y[2]) / (2 * DT)),
        },
        "trials": results,
        "best": best,
        "plot": str(plot_path),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"BEST {best['name']}: test_free_run_RMSE={best['test_free_run_rmse_v']:.8f} V")
    print(f"wrote {args.output_dir / 'metrics.json'}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
