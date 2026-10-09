"""Small identification aid for the Silverbox excerpt.

This is intentionally separate from the neural model.  It finds a useful
physics initialization and checks how hard the requested free-run metric is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


FS = 610.35
DT = 1.0 / FS
START = 42_650
STOP = 46_746
N_TRAIN = 3_072


def load_excerpt(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 1))
    excerpt = raw[START:STOP]
    u = excerpt[:, 0] - excerpt[:, 0].mean()
    y = excerpt[:, 1] - excerpt[:, 1].mean()
    return u.astype(np.float64), y.astype(np.float64)


def rk4_rollout(u: np.ndarray, y0: float, v0: float, q: np.ndarray) -> np.ndarray:
    """Integrate y'' = -a*y' - b*y - c*y^3 + g*u."""

    a, b, c, g = q
    y_hat = np.empty_like(u)
    y_hat[0] = y0
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
        y += (DT / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        v += (DT / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        y_hat[k + 1] = y
    return y_hat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/SilverboxFiles/SNLS80mV.csv"),
    )
    args = parser.parse_args()
    u, y = load_excerpt(args.csv)

    # A short Savitzky-Golay differentiator preserves the 66--70 Hz motion.
    v_sg = savgol_filter(y, 7, 4, deriv=1, delta=DT)
    acc_sg = savgol_filter(y, 7, 4, deriv=2, delta=DT)
    design = np.column_stack((-v_sg, -y, -(y**3), u))
    sl = slice(7, N_TRAIN - 7)
    q0 = np.linalg.lstsq(design[sl], acc_sg[sl], rcond=None)[0]

    # Fit short measured-initial-state rollouts.  This is used only to obtain
    # a stable initialization for the neural experiment.
    starts = np.arange(32, N_TRAIN - 160, 96)
    horizon = 128

    def residual(log_q: np.ndarray) -> np.ndarray:
        # a,b,c,g are positive for this circuit; log coordinates enforce that.
        q = np.exp(log_q)
        chunks = []
        for start in starts:
            pred = rk4_rollout(
                u[start : start + horizon], y[start], v_sg[start], q
            )
            chunks.append((pred - y[start : start + horizon]) / y.std())
        return np.concatenate(chunks)

    result = least_squares(
        residual,
        np.log(np.maximum(q0, np.array([1.0, 1e4, 1e4, 1e3]))),
        max_nfev=80,
        verbose=1,
    )
    q = np.exp(result.x)

    velocity_candidates = {
        "forward-2": (-3 * y[0] + 4 * y[1] - y[2]) / (2 * DT),
        "forward-4": (-25 * y[0] + 48 * y[1] - 36 * y[2] + 16 * y[3] - 3 * y[4])
        / (12 * DT),
        "savgol-7": v_sg[0],
    }
    print("initial derivative-LS [a,b,c,g] =", q0.tolist())
    print("short-rollout fit     [a,b,c,g] =", q.tolist())
    for name, v0 in velocity_candidates.items():
        pred = rk4_rollout(u, y[0], v0, q)
        train = np.sqrt(np.mean((pred[:N_TRAIN] - y[:N_TRAIN]) ** 2))
        test = np.sqrt(np.mean((pred[N_TRAIN:] - y[N_TRAIN:]) ** 2))
        print(f"{name:>10s}: v0={v0: .6f}, train={train:.8f}, test={test:.8f}")


if __name__ == "__main__":
    main()
