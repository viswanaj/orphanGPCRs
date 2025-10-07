#!/usr/bin/env python3

import os
import re
import argparse
from typing import Optional, Dict, List, Set, Tuple
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Bio import Phylo


LABEL_RE = re.compile(r"^(?:sp\|)?([^|]+)\|([^|]+?)(?:\|.*)?$")


def parse_label_to_accession_and_entry(label: str) -> Tuple[str, str]:
    """Extract UniProt accession and entry name from tree label like 'sp|Q8NGJ8|O51S1_HUMAN'.

    Returns (accession, entry). If not matching, returns (label, label).
    """
    m = LABEL_RE.match(label)
    if not m:
        return label, label
    acc = m.group(1)
    entry = m.group(2)
    return acc, entry


def entry_to_gene_symbol(entry: str) -> str:
    """Convert UniProt entry name like 'O51S1_HUMAN' to gene symbol 'O51S1'."""
    if "_" in entry:
        return entry.split("_")[0]
    return entry


def load_orphan_gene_set(path: Optional[str]) -> Set[str]:
    genes: Set[str] = set()
    if not path:
        return genes
    with open(path, "r") as f:
        for line in f:
            g = line.strip()
            if g:
                genes.add(g.upper())
    return genes


def infer_orphan_by_regex(gene: str) -> bool:
    """Heuristic orphan inference: genes starting with GPR followed by digits."""
    return re.match(r"^GPR\d+$", gene.upper()) is not None


def get_years_per_substitution(tree, rate: Optional[float], tree_age_years: Optional[float]) -> float:
    if (rate is None) == (tree_age_years is None):
        raise SystemExit("Provide exactly one of --rate or --tree_age_years")
    if rate is not None:
        if rate <= 0:
            raise SystemExit("--rate must be positive")
        return 1.0 / rate
    # else calibrate by total height
    root = tree.root
    max_height = 0.0
    for tip in tree.get_terminals():
        d = tree.distance(root, tip) or 0.0
        if d > max_height:
            max_height = d
    if max_height <= 0:
        raise SystemExit("Tree height is zero; cannot calibrate from --tree_age_years")
    return tree_age_years / max_height


def compute_tip_ages(tree, years_per_substitution: float) -> Dict[str, float]:
    root = tree.root
    ages: Dict[str, float] = {}
    for tip in tree.get_terminals():
        label = getattr(tip, "name", None) or ""
        dist = tree.distance(root, tip) or 0.0
        ages[label] = dist * years_per_substitution
    return ages


def plot_histogram_with_orphan_rug(
    tip_label_to_age_years: Dict[str, float],
    out_png: Optional[str],
    out_svg: Optional[str],
    orphan_genes: Set[str],
    use_regex_orphan: bool,
    bins: int,
    units: str,
) -> None:
    # Prepare data vectors
    ages_years: List[float] = []
    orphan_ages_years: List[float] = []

    for label, age in tip_label_to_age_years.items():
        acc, entry = parse_label_to_accession_and_entry(label)
        gene = entry_to_gene_symbol(entry)
        ages_years.append(age)

        is_orphan = False
        if orphan_genes and gene.upper() in orphan_genes:
            is_orphan = True
        elif use_regex_orphan and infer_orphan_by_regex(gene):
            is_orphan = True
        if is_orphan:
            orphan_ages_years.append(age)

    # Convert units if needed
    scale = 1.0
    xlabel = "Age (years)"
    if units == "myr":
        scale = 1e-6
        xlabel = "Age (million years)"
    elif units == "kyr":
        scale = 1e-3
        xlabel = "Age (thousand years)"

    ages = [a * scale for a in ages_years]
    orphan_ages = [a * scale for a in orphan_ages_years]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(ages, bins=bins, color="#5b8ef0", alpha=0.7, edgecolor="white")

    # Rug marks for orphan ages (red)
    if orphan_ages:
        ymin, ymax = ax.get_ylim()
        for x in orphan_ages:
            ax.vlines(x, ymin=ymin * 0.0, ymax=ymin + (ymax - ymin) * 0.05, colors="red", linewidth=1.2, alpha=0.9)

    # Younger to the right: invert x-axis so larger ages are on the left
    ax.invert_xaxis()

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title("GPCR Tip Age Distribution")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    if out_png:
        Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches='tight')
    if out_svg:
        Path(os.path.dirname(out_svg)).mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg, bbox_inches='tight')


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Plot histogram of GPCR tip ages with orphan rug marks.")
    ap.add_argument("--in_newick", required=True, help="Input Newick tree (tip labels like sp|ACC|ENTRY_HUMAN)")
    ap.add_argument("--rate", type=float, default=None, help="Substitution rate (subs/site/year)")
    ap.add_argument("--tree_age_years", type=float, default=None, help="Calibrate so root age equals this many years")
    ap.add_argument("--orphans_list", default=None, help="Optional file with one gene symbol per line to mark as orphan")
    ap.add_argument("--infer_orphans_by_regex", action="store_true", help="Flag genes matching ^GPR\\d+$ as orphans")
    ap.add_argument("--bins", type=int, default=30, help="Number of histogram bins")
    ap.add_argument("--units", choices=["years", "kyr", "myr"], default="myr", help="X-axis units")
    ap.add_argument("--out_png", default=os.path.join(os.getcwd(), "data", "classA_alignment", "gpcr_age_hist.png"))
    ap.add_argument("--out_svg", default=os.path.join(os.getcwd(), "data", "classA_alignment", "gpcr_age_hist.svg"))
    args = ap.parse_args(argv)

    tree = Phylo.read(args.in_newick, "newick")
    yps = get_years_per_substitution(tree, args.rate, args.tree_age_years)
    tip_ages = compute_tip_ages(tree, yps)
    orphan_genes = load_orphan_gene_set(args.orphans_list)

    plot_histogram_with_orphan_rug(
        tip_label_to_age_years=tip_ages,
        out_png=args.out_png,
        out_svg=args.out_svg,
        orphan_genes=orphan_genes,
        use_regex_orphan=args.infer_orphans_by_regex,
        bins=args.bins,
        units=args.units,
    )

    print(f"Wrote histogram to: {args.out_png} {args.out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


