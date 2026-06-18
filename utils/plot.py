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
from utils.cases import Case
from utils.topography import get_topography
from utils.domain import get_x, get_t
from utils.conservation import get_mass, get_energy

if TYPE_CHECKING:
    from model.sw_pinn import SW_PINN
    
def plot_training_history(history: list[dict[str, float]], case: Case, case_id: int, outdir: Path) -> Path:
    fig_path = outdir / f"{case.name}_loss.png"

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


def plot_solution_snapshots(model: SW_PINN, outdir: Path) -> Path:
    case = model.case
    fig_path = outdir / f"{case.name}.png"

    x = get_x()
    times = get_t(case.T, 5)

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


def plot_solution_hovmoller(model: SW_PINN, outdir: Path) -> Path:
    case = model.case
    fig_path = outdir / f"{case.name}_maps.png"

    nx = 256
    nt = 256

    x_line = get_x(nx=nx).reshape(-1)
    t_line = get_t(case.T, nt=nt).reshape(-1)
    tt, xx = torch.meshgrid(t_line, x_line, indexing="ij")
    x = xx.reshape(-1, 1)
    t = tt.reshape(-1, 1)

    with torch.no_grad():
        z = get_topography(x_line.reshape(-1, 1), model.z_case).cpu().numpy().ravel()
        h, u = model(x, t)
        h_map = h.reshape(nt, nx).cpu().numpy()
        u_map = u.reshape(nt, nx).cpu().numpy()
        eta_map = h_map + z[None, :]

    extent = [0.0, 1.0, 0.0, case.T]
    fields = [
        ("h(x,t)", h_map, "viridis"),
        ("u(x,t)", u_map, "coolwarm"),
        ("eta(x,t)", eta_map, "plasma"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 3.6), layout="constrained")
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


def create_animation(model: SW_PINN, outdir: Path, frames: int = 50) -> Path:
    case = model.case
    gif_path = outdir / f"{case.name}.gif"

    x = get_x()
    times = get_t(case.T, nt=frames)

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
        plot_solution_hovmoller(model, outdir),
        plot_conservation(model, outdir),
    ]
    if animate:
        paths.append(create_animation(model, outdir))
    model.train()
    return paths


def _conservation_series(model: SW_PINN) -> tuple[list[float], list[float], list[float]]:
    mass = get_mass(model)
    energy = get_energy(model)

    ts = sorted(mass.keys())
    M = [mass[t] for t in ts]
    E = [energy[t] for t in ts]
    M0 = M[0] if M[0] != 0.0 else 1.0
    E0 = E[0] if E[0] != 0.0 else 1.0

    M_rel = [m / M0 for m in M]
    E_rel = [e / E0 for e in E]
    return ts, M_rel, E_rel


def plot_conservation(model: SW_PINN, outdir: Path) -> Path:
    """
    Dos paneles lado a lado:
      - Izquierda: M(t)/M(0) y E(t)/E(0) con eje fijo en 1.0 ("se conserva").
      - Derecha:   |M(t)/M(0) - 1| y |E(t)/E(0) - 1| en escala log ("cuánto se desvía").
    """
    case = model.case
    fig_path = outdir / f"{case.name}_conservation.png"

    ts, M_rel, E_rel = _conservation_series(model)
    eps = 1e-16  # piso para evitar log(0)
    M_dev = [abs(m - 1.0) + eps for m in M_rel]
    E_dev = [abs(e - 1.0) + eps for e in E_rel]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), layout="constrained")

    ax_left, ax_right = axes

    ax_left.plot(ts, M_rel, "-", color="C0", label="M(t) / M(0)")
    ax_left.plot(ts, E_rel, "-", color="C3", label="E(t) / E(0)")
    ax_left.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_left.set_ylim(0.0, 1.1)
    ax_left.set_xlabel("t")
    ax_left.set_ylabel("valor normalizado")
    ax_left.set_title("conservaci\u00f3n")
    ax_left.grid(True, alpha=0.25)
    ax_left.legend(loc="lower right")

    ax_right.semilogy(ts, M_dev, "-", color="C0", label="|M(t)/M(0) - 1|")
    ax_right.semilogy(ts, E_dev, "-", color="C3", label="|E(t)/E(0) - 1|")
    ax_right.set_xlabel("t")
    ax_right.set_ylabel("desviaci\u00f3n relativa")
    ax_right.set_title("error de conservaci\u00f3n")
    ax_right.grid(True, which="both", alpha=0.25)
    ax_right.legend(loc="best")

    fig.suptitle(f"case {model.case_id}: {case.name}")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return fig_path

def plot_topography(z_case: str, outdir: Path) -> Path:
    print(f"Plotting topography {z_case}")

    fig_path = outdir / f"topography_{z_case}.png"

    x = get_x()

    fig, ax = plt.subplots(figsize=(8.0, 8.0), sharex=True, layout="constrained")

    with torch.no_grad():
        z = get_topography(x, z_case).cpu().numpy().ravel()
        x_np = x.cpu().numpy().ravel()
        ax.plot(x_np, z, "k--", linewidth=1.0, label="z")

    ax.set_ylabel("eta = h + z")
    ax.set_xlabel("x")

    fig.suptitle(f"z_case: {z_case}")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    return fig_path