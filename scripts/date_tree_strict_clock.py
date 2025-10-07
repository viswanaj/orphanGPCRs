#!/usr/bin/env python3

import os
import argparse
from typing import Optional, Dict
from pathlib import Path

from Bio import Phylo


def get_tree_height(tree) -> float:
    """Return the maximum root-to-tip path length (tree height) from branch lengths."""
    # Ensure a rooted tree representation (Phylo treats unrooted similarly for path_length)
    root = tree.root
    max_height = 0.0
    for term in tree.get_terminals():
        dist = tree.distance(root, term)
        if dist is None:
            continue
        if dist > max_height:
            max_height = dist
    return max_height


def compute_node_ages(tree, years_per_substitution: float) -> Dict[str, float]:
    """Compute ages (in years) for all clades based on distance from root.

    Age is defined as time before present, assuming the present at the tips. Under a strict
    clock, time = branch_length * years_per_substitution.
    """
    ages: Dict[str, float] = {}

    # Build a mapping from clade to age using root-to-clade distance
    root = tree.root

    def label_for_clade(clade) -> str:
        # Prefer name; for terminals, fall back to the terminal's name
        if getattr(clade, "name", None):
            return clade.name
        # For internal nodes without names, synthesize a label
        return f"node_{id(clade)}"

    for clade in tree.find_clades(order="level"):
        dist = tree.distance(root, clade) or 0.0
        ages[label_for_clade(clade)] = dist * years_per_substitution
    return ages


def write_node_ages(ages: Dict[str, float], out_path: str) -> None:
    with open(out_path, "w") as f:
        f.write("label,age_years\n")
        for label, age in ages.items():
            f.write(f"{label},{age:.3f}\n")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Estimate node ages from a Newick tree under a strict molecular clock. "
            "Provide either a substitution rate (subs/site/year) OR an overall tree age (years)."
        )
    )
    ap.add_argument("--in_newick", required=True, help="Input Newick tree (branch lengths in substitutions per site)")
    ap.add_argument("--out_csv", default=None, help="Output CSV of node ages (label,age_years)")
    ap.add_argument("--rate", type=float, default=None, help="Substitution rate (subs/site/year)")
    ap.add_argument("--tree_age_years", type=float, default=None, help="Calibrate so root age equals this many years")
    args = ap.parse_args(argv)

    if (args.rate is None) == (args.tree_age_years is None):
        raise SystemExit("Provide exactly one of --rate or --tree_age_years")

    tree = Phylo.read(args.in_newick, "newick")

    # Determine scaling factor years_per_substitution
    if args.rate is not None:
        if args.rate <= 0:
            raise SystemExit("--rate must be positive")
        years_per_substitution = 1.0 / args.rate
    else:
        height = get_tree_height(tree)
        if height <= 0:
            raise SystemExit("Tree height is zero; cannot calibrate from --tree_age_years")
        years_per_substitution = args.tree_age_years / height

    ages = compute_node_ages(tree, years_per_substitution)

    out_csv = args.out_csv
    if not out_csv:
        # default next to the input file
        base = os.path.splitext(os.path.basename(args.in_newick))[0]
        out_csv = os.path.join(os.getcwd(), f"{base}_node_ages.csv")

    Path(os.path.dirname(out_csv) or ".").mkdir(parents=True, exist_ok=True)
    write_node_ages(ages, out_csv)

    print(f"Wrote node ages to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





