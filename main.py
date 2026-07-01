#!/usr/bin/env python3
"""
PINN for 1D Shallow Water with hard constraints:

    h(x,t) = h0(x) + t * NN_h(x,t)
    u(x,t) = u0(x) + t*x*(1-x) * NN_u(x,t)

This enforces exactly:

    h(x,0) = h0(x)       (initial condition on h)
    u(x,0) = u0(x)       (initial condition on u)
    u(0,t) = u0(0) = 0   (wall BC for u, enforced by x*(1-x) factor)
    u(1,t) = u0(1) = 0   (wall BC for u, enforced by x*(1-x) factor)

Note: the Neumann (zero-gradient) BC for h at the walls is NOT enforced
by construction. Implementing a hard Neumann constraint to match the FVM
boundary treatment is planned as future work.

The training loss is the PDE residual only.
An optional soft penalty prevents unphysical h <= 0.

Usage:
    python main.py --case 0 --steps 20000
    python main.py --train-all --steps 12000
    python main.py --animate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import CASES
from model import train_one_case


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
    parser.add_argument("--outdir", type=str, default="media-out")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--T", type=float, default=None, help="Simulation time horizon")
    parser.add_argument("--animate", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Generate animated GIF of the solution (default True). Use --no-animate to skip.")
    args = parser.parse_args()

    base_outdir = Path(args.outdir)

    if args.train_all:
        case_ids = list(CASES.keys())
    else:
        case_ids = [args.case]

    for case_id in case_ids:
        case = CASES[case_id]
        case_outdir = base_outdir / case.name
        train_one_case(
            case_id=case_id,
            steps=args.steps,
            lr=args.lr,
            n_int=args.n_int,
            width=args.width,
            depth=args.depth,
            positivity_weight=args.positivity_weight,
            outdir=case_outdir,
            make_plot_files=not args.no_plots,
            simulation_T=args.T,
            animate=args.animate,
        )


if __name__ == "__main__":
    main()
