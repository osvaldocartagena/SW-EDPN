from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from config import DEVICE, DTYPE
from utils import Case, get_topography
from model.sw_pinn import SW_PINN, initial_condition
from utils.plot import make_plots

G = 9.81

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

    model = SW_PINN(case_id=case_id, width=width, depth=depth).to(DEVICE)
    if simulation_T is not None:
        if simulation_T <= 0.0:
            raise ValueError("--T must be greater than 0")
        model.case = Case(model.case.name, simulation_T)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    case = model.case
    outdir.mkdir(parents=True, exist_ok=True)
    save_path = outdir / f"{case.name}.pt"

    print(f"\n=== case {case_id}: {case.name} | T={case.T} | device={DEVICE} ===")
    print("Ansatz: h = h0 + t*NN_h ; u = u0 + t*x*(1-x)*NN_u")
    print("IC (h, u) and wall BC (u) are exact by construction.")

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

    print(f"saved: {save_path}")
    if make_plot_files:
        for fig_path in make_plots(model, history, outdir, animate=animate):
            print(f"plot: {fig_path}")

    return save_path


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def pde_residuals(model: SW_PINN, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    h, u = model(x, t)

    q = h * u
    flux = h * u**2 + 0.5 * G * h**2

    # h_t + (hu)_x = 0
    r_mass = grad(h, t) + grad(q, x)

    # (hu)_t + (hu^2 + gh^2/2)_x = -g h z_x
    z = get_topography(x, model.z_case)
    z_x = grad(z, x)

    r_mom = grad(q, t) + grad(flux, x) + G * h * z_x

    return r_mass, r_mom, h


def loss_fn(model: SW_PINN, x: torch.Tensor, t: torch.Tensor, positivity_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    r_mass, r_mom, h = pde_residuals(model, x, t)

    loss_mass = (r_mass**2).mean()
    loss_mom = (r_mom**2).mean()

    # Optional: prevents unphysical h <= 0 without modifying the ansatz.
    loss_pos = F.relu(1e-4 - h).pow(2).mean()

    loss = loss_mass + loss_mom + positivity_weight * loss_pos

    logs = {
        "mass": loss_mass.item(),
        "mom": loss_mom.item(),
        "pos": loss_pos.item(),
        "h_min": h.min().item(),
    }

    return loss, logs

@torch.no_grad()
def check_hard_constraints(model: SW_PINN, n: int = 256) -> tuple[float, float]:
    x = torch.linspace(0.0, 1.0, n, device=DEVICE, dtype=DTYPE).view(-1, 1)
    t0 = torch.zeros_like(x)

    h, u = model(x, t0)
    h0, u0 = initial_condition(x, model.z_case, model.s0_case, model.v0_case)

    ic_err = ((h - h0) ** 2).mean() + ((u - u0) ** 2).mean()

    tb = torch.linspace(0.0, model.case.T, n, device=DEVICE, dtype=DTYPE).view(-1, 1)
    xl = torch.zeros_like(tb)
    xr = torch.ones_like(tb)

    hl, ul = model(xl, tb)
    hr, ur = model(xr, tb)

    h0l, u0l = initial_condition(xl, model.z_case, model.s0_case, model.v0_case)
    h0r, u0r = initial_condition(xr, model.z_case, model.s0_case, model.v0_case)

    bc_err = (
        + ((ul - u0l) ** 2).mean()
        + ((ur - u0r) ** 2).mean()
    )

    return ic_err.item(), bc_err.item()


def rand(n: int, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, 1, device=DEVICE, dtype=DTYPE)


def sample_interior(n: int, T: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = rand(n, 0.0, 1.0)
    t = rand(n, 1e-6, T)
    return x, t