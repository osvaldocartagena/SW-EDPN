"""Compara PINN vs FVM para un caso ya entrenado: 6 paneles (2 filas x 3 cols).

Carga el .pt del PINN en media-out/{case_name}/{case_name}.pt, corre el FVM
con la misma topografia / IC, y genera un Hovmoller comparativo:

    Fila 0 (PINN): h | u | eta
    Fila 1 (FVM):  h | u | eta

con escalas compartidas por columna (vmin/vmax iguales arriba y abajo)
para que la comparacion sea visualmente honesta.

Output:
    media-out/{case_name}/{case_name}_compare.png

NO reentrena el modelo. Si el .pt no existe, falla con un mensaje claro.

Uso:
    python -m scripts.compare --case 0
    python -m scripts.compare --case 2 --N 800
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import DEVICE, DTYPE
from fvm import SimConfig, simulate
from model.sw_pinn import SW_PINN
from utils import CASES, display_label, parse_cases
from utils.topography import get_topography


NT_GRID = 256  # numero de pasos temporales para el Hovmoller
NX_GRID = 256  # numero de puntos espaciales


def load_pinn(case_id: int, ckpt_path: Path) -> SW_PINN:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint {ckpt_path}. Entrena primero con: "
            f"python main.py --case {case_id}"
        )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    width = ckpt["width"]
    depth = ckpt["depth"]
    case = ckpt["case"]

    model = SW_PINN(case_id=case_id, width=width, depth=depth).to(DEVICE)
    model.case = case  # respeta el T con el que se entreno
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


def run_fvm_on_grid(case, N: int, nt: int, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corre FVM y reinterpola los snapshots a la grilla x del PINN."""
    z_case, s0_case, v0_case = parse_cases(case.name)
    cfg = SimConfig(
        L=1.0, N=N, T=case.T,
        z_case=z_case, s0_case=s0_case, v0_case=v0_case,
        v_case="zero", bc="reflective", n_snapshots=nt,
    )
    res = simulate(cfg)

    # Reinterpolar h y u de res.x (N celdas FVM) a x_grid (nx PINN)
    h_grid = np.empty((nt, x_grid.size))
    u_grid = np.empty((nt, x_grid.size))
    for i in range(nt):
        h_grid[i] = np.interp(x_grid, res.x, res.h[i])
        u_grid[i] = np.interp(x_grid, res.x, res.u[i])

    z_grid = np.interp(x_grid, res.x, res.z)
    eta_grid = h_grid + z_grid[None, :]
    return h_grid, u_grid, eta_grid


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
        ("eta(x,t)", eta_pinn, eta_fvm, "plasma"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True, choices=list(CASES.keys()))
    parser.add_argument("--N", type=int, default=400,
                        help="Resolucion espacial del FVM (no afecta al PINN)")
    parser.add_argument("--outdir", type=Path, default=Path("media-out"),
                        help="Directorio base (default media-out, en la raiz del PINN)")
    args = parser.parse_args()

    case = CASES[args.case]
    case_dir = args.outdir / case.name
    ckpt_path = case_dir / f"{case.name}.pt"

    print(f"=== compare PINN vs FVM | {display_label(case.name)} ===")
    print(f"  ckpt: {ckpt_path}")

    model = load_pinn(args.case, ckpt_path)
    case = model.case  # adopta el T con el que se entreno

    x_grid, _, h_p, u_p, eta_p = eval_pinn_grid(model, NX_GRID, NT_GRID)
    h_f, u_f, eta_f = run_fvm_on_grid(case, args.N, NT_GRID, x_grid)

    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / f"{case.name}_compare.png"
    make_compare_plot(case.name, case.T, h_p, u_p, eta_p, h_f, u_f, eta_f, out_path)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
