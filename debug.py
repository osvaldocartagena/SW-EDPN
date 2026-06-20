
from __future__ import annotations

from utils.plot import plot_initial_state

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Plotea la topografia z(x) junto con la superficie libre "
                    "inicial eta(x, t=0) para validar la condicion inicial."
    )
    parser.add_argument("--z", type=str, default="flat",
                        choices=["flat", "inclined", "wavebreaker",
                                 "twowavebreakers", "cos"],
                        help="Caso de topografia")
    parser.add_argument("--s", type=str, default="gauss",
                        choices=["one", "gauss", "sine"],
                        help="Caso de superficie libre inicial")
    parser.add_argument("--outdir", type=str, default="media-out")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out = plot_initial_state(args.z, args.s, outdir)
    print(f"saved: {out}")


main()
