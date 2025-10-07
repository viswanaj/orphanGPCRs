#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from typing import Optional

from Bio import Phylo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render_newick(newick_path: str, out_png: Optional[str] = None, out_svg: Optional[str] = None, dpi: int = 300) -> None:
    tree = Phylo.read(newick_path, "newick")
    fig = plt.figure(figsize=(16, 24), dpi=dpi)
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, do_show=False, axes=axes)
    if out_png:
        Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches='tight')
    if out_svg:
        Path(os.path.dirname(out_svg)).mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg, bbox_inches='tight')


def main() -> int:
    ap = argparse.ArgumentParser(description="Render Newick tree to image")
    ap.add_argument("--in_newick", required=True)
    ap.add_argument("--out_png", default=None)
    ap.add_argument("--out_svg", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    render_newick(args.in_newick, args.out_png, args.out_svg, args.dpi)
    print("Rendered tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

