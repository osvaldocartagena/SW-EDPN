#!/usr/bin/env python3
"""
PINN Shallow Water 1D con hard constraints simples:

    h(x,t) = h0(x) + t*x*(1-x)*NN_h(x,t)
    u(x,t) = u0(x) + t*x*(1-x)*NN_u(x,t)

Eso fuerza exactamente:

    h(x,0) = h0(x)
    u(x,0) = u0(x)

y tambien fuerza borde Dirichlet fijo:

    h(0,t) = h0(0),   u(0,t) = u0(0)
    h(1,t) = h0(1),   u(1,t) = u0(1)

La loss principal es solo el residual PDE.
Hay una penalizacion opcional suave para evitar h <= 0.

Uso:
    python swe_pinn_simple_ansatz.py --case 0 --steps 20000
    python swe_pinn_simple_ansatz.py --train-all --steps 12000
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

g = 9.81


@dataclass
class Case:
    name: str
    T: float


CASES = {
    0: Case("flat_small_sine_height", 0.20),
    1: Case("flat_small_sine_velocity", 0.20),
    2: Case("flat_gaussian_bump", 0.15),
    3: Case("quiet_flat", 0.20),
    4: Case("slope_lake_at_rest", 0.15),
    5: Case("bottom_bump_perturbation", 0.15),
}


# ----------------------------
# Topografia e IC
# ----------------------------

def z_fn(x: torch.Tensor, case_id: int) -> torch.Tensor:
    if case_id == 4:
        return 0.05 * x

    if case_id == 5:
        return 0.05 * torch.cos(2.0 * torch.pi * x)

    # 0*x conserva el grafo para poder hacer grad(z,x)
    else: #case_id == 0: # Rompeolas rectangular suavizado
        a = 0.4
        b = 0.6
        H = 1.2
        k = 10.0
        return (H/2) * (torch.tanh(k * (x - a)) - torch.tanh(k * (x - b)))
    
    return 0.0 * x


def initial_condition(x: torch.Tensor, case_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    if case_id == 0:
        h0 = 1.0 + 0.05 * torch.sin(torch.pi * x)
        u0 = 0.0 * x

    elif case_id == 1:
        h0 = 1.0 + 0.03 * torch.sin(torch.pi * x)
        u0 = 0.05 * torch.sin(torch.pi * x)

    elif case_id == 2:
        h0 = 1.0 + 0.2 * torch.exp(-120.0 * (x - 0.2) ** 2)
        u0 = 0.0 * x

    elif case_id == 3:
        h0 = torch.ones_like(x)
        u0 = 0.0 * x

    elif case_id == 4:
        # Lake at rest: eta = h + z constante.
        eta0 = 1.0
        h0 = eta0
        u0 = 0.0 * x

    elif case_id == 5:
        h0 = 1.0  + 0.02 * torch.sin(torch.pi * x)
        u0 = 0.0 * x

    else:
        raise ValueError(f"case_id invalido: {case_id}")

    return h0 - z_fn(x, case_id), u0


# ----------------------------
# Red y ansatz
# ----------------------------

class MLP(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        X = torch.cat([2.0 * x - 1.0, 2.0 * t - 1.0], dim=1)
        return self.net(X)


class SWE_PINN(nn.Module):
    def __init__(self, case_id: int, width: int = 64, depth: int = 4):
        super().__init__()
        self.case_id = case_id
        self.case = CASES[case_id]
        self.net = MLP(width=width, depth=depth)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0, u0 = initial_condition(x, self.case_id)

        raw = self.net(x, t / self.case.T)
        dh = raw[:, 0:1]
        du = raw[:, 1:2]

        envelope = t * x * (1.0 - x)

        h = h0 + envelope * dh
        u = u0 + envelope * du

        return h, u


# ----------------------------
# Autograd PDE
# ----------------------------

def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def pde_residuals(model: SWE_PINN, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    h, u = model(x, t)

    q = h * u
    flux = h * u**2 + 0.5 * g * h**2

    # h_t + (hu)_x = 0
    r_mass = grad(h, t) + grad(q, x)

    # (hu)_t + (hu^2 + gh^2/2)_x = -g h z_x
    z = z_fn(x, model.case_id)
    z_x = grad(z, x)

    r_mom = grad(q, t) + grad(flux, x) + g * h * z_x

    return r_mass, r_mom, h


def loss_fn(model: SWE_PINN, x: torch.Tensor, t: torch.Tensor, positivity_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    r_mass, r_mom, h = pde_residuals(model, x, t)

    loss_mass = (r_mass**2).mean()
    loss_mom = (r_mom**2).mean()

    # Opcional: evita soluciones no fisicas h <= 0 sin cambiar el ansatz pedido.
    loss_pos = F.relu(1e-4 - h).pow(2).mean()

    loss = loss_mass + loss_mom + positivity_weight * loss_pos

    logs = {
        "mass": loss_mass.item(),
        "mom": loss_mom.item(),
        "pos": loss_pos.item(),
        "h_min": h.min().item(),
    }

    return loss, logs


def plot_training_history(history: list[dict[str, float]], case: Case, case_id: int, outdir: Path) -> Path:
    fig_path = outdir / f"swe_pinn_ansatz_case_{case_id}_{case.name}_loss.png"

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


def plot_solution_snapshots(model: SWE_PINN, outdir: Path, nx: int = 256) -> Path:
    case = model.case
    fig_path = outdir / f"swe_pinn_ansatz_case_{model.case_id}_{case.name}_snapshots.png"

    x = torch.linspace(0.0, 1.0, nx, device=DEVICE, dtype=DTYPE).view(-1, 1)
    times = torch.linspace(0.0, case.T, 5, device=DEVICE, dtype=DTYPE)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True, layout="constrained")

    with torch.no_grad():
        z = z_fn(x, model.case_id).cpu().numpy().ravel()
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


def plot_solution_maps(model: SWE_PINN, outdir: Path, nx: int = 160, nt: int = 120) -> Path:
    case = model.case
    fig_path = outdir / f"swe_pinn_ansatz_case_{model.case_id}_{case.name}_maps.png"

    x_line = torch.linspace(0.0, 1.0, nx, device=DEVICE, dtype=DTYPE)
    t_line = torch.linspace(0.0, case.T, nt, device=DEVICE, dtype=DTYPE)
    tt, xx = torch.meshgrid(t_line, x_line, indexing="ij")
    x = xx.reshape(-1, 1)
    t = tt.reshape(-1, 1)

    with torch.no_grad():
        h, u = model(x, t)
        h_map = h.reshape(nt, nx).cpu().numpy()
        u_map = u.reshape(nt, nx).cpu().numpy()

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


def create_animation(model: SWE_PINN, outdir: Path, nx: int = 256, frames: int = 50) -> Path:
    case = model.case
    gif_path = outdir / f"swe_pinn_ansatz_case_{model.case_id}_{case.name}.gif"

    x = torch.linspace(0.0, 1.0, nx, device=DEVICE, dtype=DTYPE).view(-1, 1)
    times = torch.linspace(0.0, case.T, frames, device=DEVICE, dtype=DTYPE)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True, layout="constrained")
    
    # Pre-calculate to get bounds
    h_all, u_all, eta_all = [], [], []
    with torch.no_grad():
        z = z_fn(x, model.case_id).cpu().numpy().ravel()
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


def make_plots(model: SWE_PINN, history: list[dict[str, float]], outdir: Path, animate: bool = False) -> list[Path]:
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


@torch.no_grad()
def check_hard_constraints(model: SWE_PINN, n: int = 256) -> tuple[float, float]:
    x = torch.linspace(0.0, 1.0, n, device=DEVICE, dtype=DTYPE).view(-1, 1)
    t0 = torch.zeros_like(x)

    h, u = model(x, t0)
    h0, u0 = initial_condition(x, model.case_id)

    ic_err = ((h - h0) ** 2).mean() + ((u - u0) ** 2).mean()

    tb = torch.linspace(0.0, model.case.T, n, device=DEVICE, dtype=DTYPE).view(-1, 1)
    xl = torch.zeros_like(tb)
    xr = torch.ones_like(tb)

    hl, ul = model(xl, tb)
    hr, ur = model(xr, tb)

    h0l, u0l = initial_condition(xl, model.case_id)
    h0r, u0r = initial_condition(xr, model.case_id)

    bc_err = (
        ((hl - h0l) ** 2).mean()
        + ((ul - u0l) ** 2).mean()
        + ((hr - h0r) ** 2).mean()
        + ((ur - u0r) ** 2).mean()
    )

    return ic_err.item(), bc_err.item()


# ----------------------------
# Sampling y train
# ----------------------------

def rand(n: int, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, 1, device=DEVICE, dtype=DTYPE)


def sample_interior(n: int, T: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = rand(n, 0.0, 1.0)
    t = rand(n, 1e-6, T)
    return x, t


def train_one_case(
    case_id: int,
    steps: int,
    lr: float,
    n_int: int,
    width: int,
    depth: int,
    positivity_weight: float,
    outdir: Path,
    make_plot_files: bool,
    simulation_T: float | None = None,
    animate: bool = False,
) -> Path:
    torch.manual_seed(1234 + case_id)

    model = SWE_PINN(case_id=case_id, width=width, depth=depth).to(DEVICE)
    if simulation_T is not None:
        if simulation_T <= 0.0:
            raise ValueError("--T debe ser mayor que 0")
        model.case = Case(model.case.name, simulation_T)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    case = model.case
    outdir.mkdir(parents=True, exist_ok=True)
    save_path = outdir / f"swe_pinn_ansatz_case_{case_id}_{case.name}.pt"

    print(f"\n=== case {case_id}: {case.name} | T={case.T} | device={DEVICE} ===")
    print("Ansatz: h,u = inicial + t*x*(1-x)*NN")
    print("IC y borde Dirichlet fijo son exactos por construccion.")

    history: list[dict[str, float]] = []

    for it in range(1, steps + 1):
        x, t = sample_interior(n_int, case.T)

        loss, logs = loss_fn(model, x, t, positivity_weight)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        history.append({"it": it, "loss": loss.item(), **logs})

        if it == 1 or it % 500 == 0:
            ic_err, bc_err = check_hard_constraints(model)
            print(
                f"it={it:6d} "
                f"loss={loss.item():.3e} "
                f"mass={logs['mass']:.3e} "
                f"mom={logs['mom']:.3e} "
                f"pos={logs['pos']:.3e} "
                f"hmin={logs['h_min']:.3e} "
                f"ic={ic_err:.1e} "
                f"bc={bc_err:.1e}"
            )

    torch.save(
        {
            "case_id": case_id,
            "case": case,
            "width": width,
            "depth": depth,
            "state_dict": model.state_dict(),
        },
        save_path,
    )

    print(f"guardado: {save_path}")
    if make_plot_files:
        for fig_path in make_plots(model, history, outdir, animate=animate):
            print(f"plot: {fig_path}")

    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=0, choices=list(CASES.keys()))
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-int", type=int, default=2048)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--positivity-weight", type=float, default=100.0)
    parser.add_argument("--outdir", type=str, default="runs_swe_simple_ansatz")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--T", type=float, default=None, help="Simulation time horizon")
    parser.add_argument("--animate", action="store_true", help="Generate an animated gif of the solution")
    args = parser.parse_args()

    outdir = Path(args.outdir)

    if args.train_all:
        for case_id in CASES:
            train_one_case(
                case_id=case_id,
                steps=args.steps,
                lr=args.lr,
                n_int=args.n_int,
                width=args.width,
                depth=args.depth,
                positivity_weight=args.positivity_weight,
                outdir=outdir,
                make_plot_files=not args.no_plots,
                simulation_T=args.T,
                animate=args.animate,
            )
    else:
        train_one_case(
            case_id=args.case,
            steps=args.steps,
            lr=args.lr,
            n_int=args.n_int,
            width=args.width,
            depth=args.depth,
            positivity_weight=args.positivity_weight,
            outdir=outdir,
            make_plot_files=not args.no_plots,
            simulation_T=args.T,
            animate=args.animate,
        )


if __name__ == "__main__":
    main()
