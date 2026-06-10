from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from config import DEVICE, DTYPE
from utils.cases import Case, CASES
from utils.topography import get_topography

if TYPE_CHECKING:
    from model.sw_pinn import SW_PINN


def plot_training_history(history: list[dict[str, float]], case: Case, case_id: int, outdir: Path) -> Path:
    fig_path = outdir / f"sw_pinn_ansatz_case_{case_id}_{case.name}_loss.png"

    fig, ax = plt.subplots(figsize=(7.0, 4.0), layout="constrained")
    its = [row["it"] for row in history]
    for key in ("loss", "mass", "mom", "pos"):
        ax.semilogy(its, [row[key] for row in history], label=key)

    ax.set_title(f"case {case_id}: {case.name}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return fig_path


def plot_solution_snapshots(model: SW_PINN, outdir: Path, nx: int = 256) -> Path:
    case = model.case
    fig_path = outdir / f"sw_pinn_ansatz_case_{model.case_id}_{case.name}_snapshots.png"

    x = torch.linspace(0.0, 1.0, nx, device=DEVICE, dtype=DTYPE).view(-1, 1)
    times = torch.linspace(0.0, case.T, 5, device=DEVICE, dtype=DTYPE)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True, layout="constrained")

    with torch.no_grad():
        z = get_topography(x, model.z_case).cpu().numpy().ravel()
        x_np = x.cpu().numpy().ravel()

        for tau in times:
            t = torch.full_like(x, tau.item())
            h, u = model(x, t)
            h_np = h.cpu().numpy().ravel()
            u_np = u.cpu().numpy().ravel()
            eta_np = h_np + z
            label = f"t={tau.item():.3f}"

            axes[0].plot(x_np, h_np, label=label)
            axes[1].plot(x_np, u_np, label=label)
            axes[2].plot(x_np, eta_np, label=label)

        axes[2].plot(x_np, z, "k--", linewidth=1.0, label="z")

    axes[0].set_ylabel("h")
    axes[1].set_ylabel("u")
    axes[2].set_ylabel("eta = h + z")
    axes[2].set_xlabel("x")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncols=3, fontsize=8)

    fig.suptitle(f"case {model.case_id}: {case.name}")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return fig_path


def plot_solution_maps(model: SW_PINN, outdir: Path, nx: int = 160, nt: int = 120) -> Path:
    case = model.case
    fig_path = outdir / f"sw_pinn_ansatz_case_{model.case_id}_{case.name}_maps.png"

    x_line = torch.linspace(0.0, 1.0, nx, device=DEVICE, dtype=DTYPE)
    t_line = torch.linspace(0.0, case.T, nt, device=DEVICE, dtype=DTYPE)
    tt, xx = torch.meshgrid(t_line, x_line, indexing="ij")
    x = xx.reshape(-1, 1)
    t = tt.reshape(-1, 1)

    with torch.no_grad():
        h, u = model(x, t)
        h_map = h.reshape(nt, nx).cpu().numpy()
        u_map = u.reshape(nt, nx).cpu().numpy()

    from model.training import pde_residuals
    r_mass, r_mom, _ = pde_residuals(model, x, t)
    residual = torch.sqrt(r_mass.detach() ** 2 + r_mom.detach() ** 2)
    residual_map = residual.reshape(nt, nx).cpu().numpy()

    extent = [0.0, 1.0, 0.0, case.T]
    fields = [
        ("h(x,t)", h_map, "viridis"),
        ("u(x,t)", u_map, "coolwarm"),
        ("sqrt(r_mass^2 + r_mom^2)", residual_map, "magma"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), layout="constrained")
    for ax, (title, data, cmap) in zip(axes, fields, strict=True):
        im = ax.imshow(data, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"case {model.case_id}: {case.name}")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return fig_path


def _bounds_with_padding(series: list, extra: list | None = None, min_pad: float = 0.05) -> tuple[float, float]:
    values = list(series)
    if extra is not None:
        values.extend(extra)

    data_min = min(float(v.min()) for v in values)
    data_max = max(float(v.max()) for v in values)
    pad = max(min_pad, (data_max - data_min) * 0.1)
    return data_min - pad, data_max + pad


def create_animation(model: SW_PINN, outdir: Path, nx: int = 256, frames: int = 50) -> Path:
    case = model.case
    gif_path = outdir / f"sw_pinn_ansatz_case_{model.case_id}_{case.name}.gif"

    x = torch.linspace(0.0, 1.0, nx, device=DEVICE, dtype=DTYPE).view(-1, 1)
    times = torch.linspace(0.0, case.T, frames, device=DEVICE, dtype=DTYPE)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True, layout="constrained")
    
    # Pre-calculate to get bounds
    h_all, u_all, eta_all = [], [], []
    with torch.no_grad():
        z = get_topography(x, model.z_case).cpu().numpy().ravel()
        x_np = x.cpu().numpy().ravel()
        for tau in times:
            t = torch.full_like(x, tau.item())
            h, u = model(x, t)
            h_np = h.cpu().numpy().ravel()
            u_np = u.cpu().numpy().ravel()
            h_all.append(h_np)
            u_all.append(u_np)
            eta_all.append(h_np + z)
            
    line_h, = axes[0].plot([], [], label="h", color="C0")
    line_u, = axes[1].plot([], [], label="u", color="C1")
    line_eta, = axes[2].plot([], [], label="eta = h + z", color="C2")
    axes[2].plot(x_np, z, "k--", linewidth=1.0, label="z")

    axes[0].set_ylabel("h")
    axes[1].set_ylabel("u")
    axes[2].set_ylabel("eta = h + z")
    axes[2].set_xlabel("x")
    
    axes[0].set_ylim(*_bounds_with_padding(h_all))
    axes[1].set_ylim(*_bounds_with_padding(u_all))
    axes[2].set_ylim(*_bounds_with_padding(eta_all, extra=[z]))

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    title = fig.suptitle("")

    def init():
        line_h.set_data([], [])
        line_u.set_data([], [])
        line_eta.set_data([], [])
        title.set_text("")
        return line_h, line_u, line_eta, title

    def update(frame_idx):
        line_h.set_data(x_np, h_all[frame_idx])
        line_u.set_data(x_np, u_all[frame_idx])
        line_eta.set_data(x_np, eta_all[frame_idx])
        title.set_text(f"case {model.case_id}: {case.name} | t={times[frame_idx].item():.3f}")
        return line_h, line_u, line_eta, title

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    ani.save(gif_path, writer="pillow", fps=15)
    plt.close(fig)
    return gif_path


def make_plots(model: SW_PINN, history: list[dict[str, float]], outdir: Path, animate: bool = False) -> list[Path]:
    model.eval()
    paths = [
        plot_training_history(history, model.case, model.case_id, outdir),
        plot_solution_snapshots(model, outdir),
        plot_solution_maps(model, outdir),
    ]
    if animate:
        paths.append(create_animation(model, outdir))
    model.train()
    return paths
