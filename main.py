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
    python sw_pinn_simple_ansatz.py --case 0 --steps 20000
    python sw_pinn_simple_ansatz.py --train-all --steps 12000
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
