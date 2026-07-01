"""Plot topography z(x) and initial free surface η(x, t=0) for a given case.

Useful for visually inspecting and validating initial condition configurations
before launching a full training run.

Usage:
    python -m scripts.debug_ic --z wavebreaker --s gauss
    python -m scripts.debug_ic --z flat --s one
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.plot import plot_initial_state


def main():
    parser = argparse.ArgumentParser(
        description="Plot topography z(x) and initial free surface η(x, t=0) "
                    "to inspect and validate initial condition configurations."
    )
    parser.add_argument("--z", type=str, default="flat",
                        choices=["flat", "inclined", "wavebreaker",
                                 "twowavebreakers", "cos"],
                        help="Topography case")
    parser.add_argument("--s", type=str, default="gauss",
                        choices=["one", "gauss", "sine"],
                        help="Initial free surface case")
    parser.add_argument("--outdir", type=str, default="media-out")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out = plot_initial_state(args.z, args.s, outdir)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
