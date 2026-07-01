"""Compare PINN vs FVM for a trained case.

Loads the PINN checkpoint from media-out/{case_name}/{case_name}.pt, runs the
FVM with the same topography and initial condition, and produces:

    {case_name}_compare.png  -- Hovmoller side-by-side (2 rows x 3 cols)
    {case_name}_compare.gif  -- overlay animation: η_PINN vs η_FVM (--animate)

The Hovmoller shares colour scales per column so the comparison is honest.
The animation shows both free surfaces on a single panel with two colours,
which is the intended primary visual for the README.

Does NOT retrain. Fails with a clear message if the checkpoint is missing.

Usage:
    python -m scripts.compare --case 0
    python -m scripts.compare --case 2 --N 800 --animate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import DEVICE, DTYPE
from fvm import SimConfig, simulate
from model.sw_pinn import SW_PINN
from utils import CASES, display_label, parse_cases
from utils.topography import get_topography


NT_GRID = 256  # time steps for the Hovmoller diagram
NX_GRID = 256  # spatial evaluation points


def load_pinn(case_id: int, ckpt_path: Path) -> SW_PINN:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first with: "
            f"python main.py --case {case_id}"
        )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    width = ckpt["width"]
    depth = ckpt["depth"]
    case = ckpt["case"]

    model = SW_PINN(case_id=case_id, width=width, depth=depth).to(DEVICE)
    model.case = case  # preserve the T used during training
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def eval_pinn_grid(model: SW_PINN, nx: int, nt: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    case = model.case
    x_line = torch.linspace(0.0, 1.0, nx, device=DEVICE, dtype=DTYPE).view(-1)
    t_line = torch.linspace(0.0, case.T, nt, device=DEVICE, dtype=DTYPE).view(-1)
    tt, xx = torch.meshgrid(t_line, x_line, indexing="ij")
    x = xx.reshape(-1, 1)
    t = tt.reshape(-1, 1)

    h, u = model(x, t)
    h_map = h.reshape(nt, nx).cpu().numpy()
    u_map = u.reshape(nt, nx).cpu().numpy()

    z = get_topography(x_line.reshape(-1, 1), model.z_case).cpu().numpy().ravel()
    eta_map = h_map + z[None, :]
    x_np = x_line.cpu().numpy()
    t_np = t_line.cpu().numpy()
    return x_np, t_np, h_map, u_map, eta_map


def run_fvm_on_grid(
    case, N: int, nt: int, x_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run FVM and interpolate snapshots onto the PINN x-grid.

    Returns h_grid, u_grid, eta_grid, z_grid — all shaped (nt, nx) or (nx,).
    """
    z_case, s0_case, v0_case = parse_cases(case.name)
    cfg = SimConfig(
        L=1.0, N=N, T=case.T,
        z_case=z_case, s0_case=s0_case, v0_case=v0_case,
        v_case="zero", bc="reflective", n_snapshots=nt,
    )
    res = simulate(cfg)

    h_grid = np.empty((nt, x_grid.size))
    u_grid = np.empty((nt, x_grid.size))
    for i in range(nt):
        h_grid[i] = np.interp(x_grid, res.x, res.h[i])
        u_grid[i] = np.interp(x_grid, res.x, res.u[i])

    z_grid = np.interp(x_grid, res.x, res.z)
    eta_grid = h_grid + z_grid[None, :]
    return h_grid, u_grid, eta_grid, z_grid


def make_compare_plot(
    case_name: str,
    T: float,
    h_pinn: np.ndarray, u_pinn: np.ndarray, eta_pinn: np.ndarray,
    h_fvm: np.ndarray,  u_fvm: np.ndarray,  eta_fvm: np.ndarray,
    out_path: Path,
) -> None:
    fields = [
        ("h(x,t)",   h_pinn,   h_fvm,   "viridis"),
        ("u(x,t)",   u_pinn,   u_fvm,   "coolwarm"),
        ("η(x,t)", eta_pinn, eta_fvm, "plasma"),
    ]
    extent = [0.0, 1.0, 0.0, T]

    fig, axes = plt.subplots(2, 3, figsize=(17.0, 7.0), layout="constrained")

    for col, (title, pinn_d, fvm_d, cmap) in enumerate(fields):
        if cmap == "coolwarm":
            vmax = max(float(np.abs(pinn_d).max()), float(np.abs(fvm_d).max())) or 1.0
            vmin = -vmax
        else:
            vmin = min(float(pinn_d.min()), float(fvm_d.min()))
            vmax = max(float(pinn_d.max()), float(fvm_d.max()))

        for row, data in enumerate((pinn_d, fvm_d)):
            ax = axes[row, col]
            im = ax.imshow(
                data, origin="lower", aspect="auto", extent=extent,
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
            ax.set_xlabel("x")
            ax.set_yticks([0.0, T])
            if col == 0:
                ax.set_ylabel(f"{'PINN' if row == 0 else 'FVM'}\nt")
            if row == 0:
                ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(display_label(case_name))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_l2_errors(
    h_pinn: np.ndarray, u_pinn: np.ndarray, eta_pinn: np.ndarray,
    h_fvm: np.ndarray,  u_fvm: np.ndarray,  eta_fvm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Point-wise L² error over x at each time snapshot.

    Returns a dict with keys 'h', 'u', 'eta', each a 1D array of length nt.
    """
    return {
        "h":   np.sqrt(np.mean((h_pinn   - h_fvm)  ** 2, axis=1)),
        "u":   np.sqrt(np.mean((u_pinn   - u_fvm)  ** 2, axis=1)),
        "eta": np.sqrt(np.mean((eta_pinn - eta_fvm) ** 2, axis=1)),
    }


def plot_l2_errors(
    case_name: str,
    t_line: np.ndarray,
    errors: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Semilog plot of L²(Ω) error vs FVM over time for h, u, and η."""
    fig, ax = plt.subplots(figsize=(8.0, 4.5), layout="constrained")
    for field, label in [("eta", "η = h + z"), ("h", "h"), ("u", "u")]:
        ax.semilogy(t_line, errors[field], label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("L²(Ω) error  vs  FVM")
    ax.set_title(f"{display_label(case_name)}  —  PINN vs FVM pointwise L² error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def create_comparison_animation(
    case_name: str,
    x_grid: np.ndarray,
    t_line: np.ndarray,
    eta_pinn: np.ndarray,
    eta_fvm: np.ndarray,
    z_grid: np.ndarray,
    out_path: Path,
    frames: int = 60,
) -> Path:
    """Overlay animation: η_PINN (blue) and η_FVM (orange) on a single panel.

    Both free surfaces are drawn over the topography z(x) (dashed black),
    making it easy to see how the wave interacts with the bottom in each method.
    """
    n_snapshots = eta_pinn.shape[0]
    idx = [int(round(i * (n_snapshots - 1) / (frames - 1))) for i in range(frames)]

    all_eta = np.concatenate([eta_pinn[idx].ravel(), eta_fvm[idx].ravel(), z_grid])
    ymin, ymax = float(all_eta.min()), float(all_eta.max())
    pad = max(0.05, (ymax - ymin) * 0.12)

    fig, ax = plt.subplots(figsize=(9.0, 4.5), layout="constrained")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("η = h + z  (free surface)")
    ax.grid(True, alpha=0.25)

    ax.fill_between(x_grid, ymin - pad, z_grid, color="0.85", zorder=0)
    ax.plot(x_grid, z_grid, "k--", linewidth=1.0, label="z(x)  bottom", zorder=1)

    (line_pinn,) = ax.plot([], [], color="C0", linewidth=2.0, label="PINN", zorder=3)
    (line_fvm,) = ax.plot([], [], color="C1", linewidth=2.0, label="FVM (Berthon-Chalons)", zorder=2)
    ax.legend(loc="upper right", fontsize=9)

    title = ax.set_title("")

    def init():
        line_pinn.set_data([], [])
        line_fvm.set_data([], [])
        title.set_text("")
        return line_pinn, line_fvm, title

    def update(frame_idx: int):
        i = idx[frame_idx]
        line_pinn.set_data(x_grid, eta_pinn[i])
        line_fvm.set_data(x_grid, eta_fvm[i])
        title.set_text(f"{display_label(case_name)}  |  t = {t_line[i]:.3f}")
        return line_pinn, line_fvm, title

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    ani.save(out_path, writer="pillow", fps=15)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True, choices=list(CASES.keys()))
    parser.add_argument("--N", type=int, default=400,
                        help="Spatial resolution of the FVM grid (does not affect PINN)")
    parser.add_argument("--outdir", type=Path, default=Path("media-out"),
                        help="Output base directory (default: media-out)")
    parser.add_argument(
        "--animate", action=argparse.BooleanOptionalAction, default=True,
        help="Generate overlay comparison GIF (default: True). Use --no-animate to skip.",
    )
    args = parser.parse_args()

    case = CASES[args.case]
    case_dir = args.outdir / case.name
    ckpt_path = case_dir / f"{case.name}.pt"

    print(f"=== compare PINN vs FVM | {display_label(case.name)} ===")
    print(f"  ckpt: {ckpt_path}")

    model = load_pinn(args.case, ckpt_path)
    case = model.case

    x_grid, t_grid, h_p, u_p, eta_p = eval_pinn_grid(model, NX_GRID, NT_GRID)
    h_f, u_f, eta_f, z_grid = run_fvm_on_grid(case, args.N, NT_GRID, x_grid)

    case_dir.mkdir(parents=True, exist_ok=True)

    out_path = case_dir / f"{case.name}_compare.png"
    make_compare_plot(case.name, case.T, h_p, u_p, eta_p, h_f, u_f, eta_f, out_path)
    print(f"  saved: {out_path}")

    errors = compute_l2_errors(h_p, u_p, eta_p, h_f, u_f, eta_f)
    err_path = case_dir / f"{case.name}_l2_error.png"
    plot_l2_errors(case.name, t_grid, errors, err_path)
    print(f"  saved: {err_path}")
    print(f"  L2 error (η)  mean={errors['eta'].mean():.3e}  max={errors['eta'].max():.3e}  t=T: {errors['eta'][-1]:.3e}")
    print(f"  L2 error (h)  mean={errors['h'].mean():.3e}  max={errors['h'].max():.3e}  t=T: {errors['h'][-1]:.3e}")
    print(f"  L2 error (u)  mean={errors['u'].mean():.3e}  max={errors['u'].max():.3e}  t=T: {errors['u'][-1]:.3e}")

    if args.animate:
        gif_path = case_dir / f"{case.name}_compare.gif"
        create_comparison_animation(case.name, x_grid, t_grid, eta_p, eta_f, z_grid, gif_path)
        print(f"  saved: {gif_path}")


if __name__ == "__main__":
    main()
