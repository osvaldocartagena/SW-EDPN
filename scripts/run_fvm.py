"""CLI: run one or all FVM cases and save plots to media-out/{case_name}/.

Uses the same case registry as the PINN (utils.cases.CASES) and parses
case names with utils.parse_cases.

Usage:
    python -m scripts.run_fvm --case 0
    python -m scripts.run_fvm --case all
    python -m scripts.run_fvm --case 2 --animate --N 800
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils import CASES, parse_cases, display_label
from fvm import SimConfig, simulate
from fvm.plot import make_plots


def run_one(
    case_id: int,
    N: int,
    cfl: float,
    n_snapshots: int,
    v0_case: str,
    bc: str,
    base_outdir: Path,
    animate: bool,
    simulation_T: float | None,
) -> None:
    case = CASES[case_id]
    z_case, s0_case, v0_case = parse_cases(case.name)

    cfg = SimConfig(
        T=case.T if simulation_T is None else simulation_T,
        N=N,
        cfl=cfl,
        n_snapshots=n_snapshots,
        z_case=z_case,
        s0_case=s0_case,
        v0_case=v0_case,
        bc=bc,
    )
    label = display_label(case.name)
    print(
        f"\n=== {label} (case {case_id}) | T={cfg.T} | N={cfg.N} | "
        f"z={cfg.z_case} s0={cfg.s0_case} v0={cfg.v0_case} v={cfg.v_case} bc={cfg.bc} ==="
    )

    res = simulate(cfg)
    print(f"  iters: {res.iters}  |  dx={cfg.dx:.4e}")
    print(f"  mass drift:   {abs(res.mass()[-1] / res.mass()[0] - 1):.3e}")
    print(f"  energy drift: {abs(res.energy()[-1] / res.energy()[0] - 1):.3e}")

    case_outdir = base_outdir / case.name
    case_outdir.mkdir(parents=True, exist_ok=True)

    for p in make_plots(res, case_outdir, case.name, label, animate=animate):
        print(f"  -> {p}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="0", help="ID de caso o 'all'")
    parser.add_argument("--N", type=int, default=400)
    parser.add_argument("--cfl", type=float, default=0.45)
    parser.add_argument("--snapshots", type=int, default=200)
    parser.add_argument("--v", type=str, default="zero",
                        choices=["zero", "constant", "sine", "triangular"],
                        help="Time-dependent body forcing h*v(t) (default zero)")
    parser.add_argument("--bc", type=str, default="reflective",
                        choices=["reflective", "neumann"])
    parser.add_argument("--outdir", type=Path, default=Path("media-out"),
                        help="Output base directory (default: media-out)")
    parser.add_argument("--animate", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Generate animated GIF (default True). Use --no-animate to skip.")
    parser.add_argument("--T", type=float, default=None,
                        help="Override the case simulation time T")
    args = parser.parse_args()

    case_ids = list(CASES.keys()) if args.case == "all" else [int(args.case)]
    for cid in case_ids:
        run_one(
            case_id=cid,
            N=args.N,
            cfl=args.cfl,
            n_snapshots=args.snapshots,
            v0_case=args.v,
            bc=args.bc,
            base_outdir=args.outdir,
            animate=args.animate,
            simulation_T=args.T,
        )


if __name__ == "__main__":
    main()
