
from __future__ import annotations

from utils.plot import plot_topography

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zcase", type=str, default="flat",)
    parser.add_argument("--outdir", type=str, default="media-out")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    z_case = args.zcase

    plot_topography(z_case, outdir)

main()