#!/usr/bin/env python3
"""
Maximum-likelihood phylogenetic tree of human GPCRs.

Pipeline
--------
1. Load sequences from gpcr_sequences.db → FASTA
2. Multiple sequence alignment (MAFFT | MUSCLE | Clustal Omega | Kalign)
3. VeryFastTree (WAG + Gamma) → Newick tree
4. Biopython: midpoint-root
5. Circular cladogram layout → interactive plotly HTML
6. Colour leaves by consensus_status (orphan / cognate / unknown)

Outputs  (all in gpcr_sequence_db/)
-------
  gpcr.fasta                       raw sequences
  gpcr.msa / gpcr.{aligner}.msa    alignment
  gpcr.nwk / gpcr.{aligner}.nwk    ML Newick tree
  gpcr_tree.html / gpcr_tree.{aligner}.html   interactive cladogram

The default aligner is MAFFT and uses the unsuffixed filenames so the original
artefacts on disk are preserved. Pass --aligner muscle|clustalo|kalign to run
an alternative; outputs are written with the aligner name embedded.
"""

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio import Phylo
from Bio.Phylo.BaseTree import Tree
import plotly.graph_objects as go

sys.setrecursionlimit(10000)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUT_DIR    = REPO_ROOT / "gpcr_sequence_db"
DB_PATH    = OUT_DIR / "gpcr_sequences.db"
FASTA_PATH = OUT_DIR / "gpcr.fasta"

# ── Aligner registry ─────────────────────────────────────────────────────────
# Each entry: pretty name + builder for the subprocess command. The runner
# below feeds FASTA → stdout file in MSA_PATH.
ALIGNERS = {
    "mafft":    {"label": "MAFFT (--auto)"},
    "muscle":   {"label": "MUSCLE v5 (-super5)"},
    "clustalo": {"label": "Clustal Omega"},
    "kalign":   {"label": "Kalign 3"},
}

VERYFASTTREE = "VeryFastTree"

# ── Visual ────────────────────────────────────────────────────────────────────
STATUS_COLOR = {
    "orphan":  "#e74c3c",
    "cognate": "#2980b9",
    "unknown": "#bdc3c7",
}
STATUS_ORDER = ["orphan", "cognate", "unknown"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & write FASTA
# ─────────────────────────────────────────────────────────────────────────────

def load_and_write_fasta() -> dict[str, dict]:
    """
    Pull sequences from DB, write FASTA.
    Returns {accession: record_dict}.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT uniprot_accession, uniprot_id, gene_name, protein_name,
               sequence, consensus_status, cluster_id, iuphar_name
        FROM gpcrs ORDER BY uniprot_accession
    """).fetchall()
    conn.close()

    records = {r["uniprot_accession"]: dict(r) for r in rows}

    with open(FASTA_PATH, "w") as fh:
        for acc, rec in records.items():
            label = acc   # keep it simple — tree leaf names = accession
            fh.write(f">{label}\n{rec['sequence']}\n")

    print(f"  {len(records)} sequences written → {FASTA_PATH.name}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2. MSA — one runner per aligner. Each writes FASTA-format alignment to msa_path.
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], stdout_path: Path | None = None, label: str = "") -> str:
    """Run a subprocess; on failure print stderr and raise. Returns stderr text."""
    print(f"  Running: {' '.join(cmd)}")
    if stdout_path is not None:
        with open(stdout_path, "w") as out:
            result = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
    else:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label or cmd[0]} failed:\n{result.stderr}")
    return result.stderr


def run_aligner(aligner: str, fasta: Path, msa_path: Path) -> None:
    if aligner == "mafft":
        _run(
            ["mafft", "--auto", "--thread", "-1", "--quiet", str(fasta)],
            stdout_path=msa_path,
            label="MAFFT",
        )
    elif aligner == "muscle":
        # MUSCLE v5 uses subcommand syntax. -super5 scales to thousands of seqs;
        # output is written to the path given by -output.
        _run(
            ["muscle", "-super5", str(fasta), "-output", str(msa_path)],
            label="MUSCLE",
        )
    elif aligner == "clustalo":
        _run(
            [
                "clustalo",
                "-i", str(fasta),
                "-o", str(msa_path),
                "--outfmt", "fasta",
                "--threads", "0",   # 0 = auto
                "--force",
            ],
            label="Clustal Omega",
        )
    elif aligner == "kalign":
        _run(
            ["kalign", "-i", str(fasta), "-o", str(msa_path), "-f", "fasta"],
            label="Kalign",
        )
    else:
        raise ValueError(f"Unknown aligner: {aligner}")
    print(f"  MSA written → {msa_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. VeryFastTree
# ─────────────────────────────────────────────────────────────────────────────

def run_fasttree(msa_path: Path, nwk_path: Path) -> None:
    cmd = [VERYFASTTREE, "-wag", "-gamma", str(msa_path)]
    stderr = _run(cmd, stdout_path=nwk_path, label="VeryFastTree")
    for line in stderr.splitlines()[-6:]:
        if line.strip():
            print(f"  [VFT] {line}")
    print(f"  Newick tree written → {nwk_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5. Parse tree + circular cladogram layout
# ─────────────────────────────────────────────────────────────────────────────

def _build_parent_map(tree: Tree) -> dict:
    parent = {}
    for node in tree.get_nonterminals():
        for child in node.clades:
            parent[child] = node
    return parent


def cladogram_layout(tree: Tree) -> dict:
    """
    Circular cladogram: leaves evenly spaced by DFS order.
    Internal nodes placed at the angular midpoint of their descendants.
    Returns {clade: (x, y)}.
    """
    leaves   = list(tree.get_terminals())
    n        = len(leaves)
    leaf_idx = {leaf: i for i, leaf in enumerate(leaves)}

    # Node depth (edge count from root)
    depths: dict = {}
    def _depth(clade, d: int = 0):
        depths[clade] = d
        for child in clade.clades:
            _depth(child, d + 1)
    _depth(tree.root)
    max_depth = max(depths.values()) or 1

    # Angular span → midpoint angle for each node
    angles: dict = {}
    def _span(clade):
        if clade.is_terminal():
            a = 2 * np.pi * leaf_idx[clade] / n
            angles[clade] = a
            return a, a
        child_spans = [_span(c) for c in clade.clades]
        lo = min(s[0] for s in child_spans)
        hi = max(s[1] for s in child_spans)
        angles[clade] = (lo + hi) / 2
        return lo, hi
    _span(tree.root)

    positions = {}
    for node in depths:
        r = depths[node] / max_depth
        a = angles[node]
        positions[node] = (r * np.cos(a), r * np.sin(a))

    return positions


def build_plotly_figure(
    tree: Tree,
    records: dict[str, dict],
    aligner_label: str = "MAFFT (--auto)",
) -> go.Figure:
    """
    Circular cladogram coloured by consensus_status.
    Branch lines in one trace; leaves in per-status traces for the legend.
    """
    pos    = cladogram_layout(tree)
    parent = _build_parent_map(tree)

    # ── Branch lines (single trace, None-separated) ───────────────────────
    bx, by = [], []
    for child, par in parent.items():
        px, py = pos[par]
        cx, cy = pos[child]
        # Draw an "L" in polar coords: arc-step to child's angle at parent's
        # radius, then radial to child. Approximated with 2 line segments.
        par_r   = np.hypot(px, py)
        child_a = np.arctan2(cy, cx)
        arc_x   = par_r * np.cos(child_a)
        arc_y   = par_r * np.sin(child_a)
        bx += [px, arc_x, cx, None]
        by += [py, arc_y, cy, None]

    traces = [
        go.Scatter(
            x=bx, y=by,
            mode="lines",
            line=dict(color="#d0d0d0", width=0.4),
            hoverinfo="none",
            showlegend=False,
        )
    ]

    # ── Leaf dots per status ──────────────────────────────────────────────
    for status in STATUS_ORDER:
        nodes = [
            n for n in tree.get_terminals()
            if records.get(n.name, {}).get("consensus_status") == status
        ]
        if not nodes:
            continue
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]

        hover = []
        for n in nodes:
            rec = records.get(n.name, {})
            gene    = rec.get("gene_name") or rec.get("uniprot_id") or n.name
            pname   = (rec.get("protein_name") or "")[:55]
            iuphar  = rec.get("iuphar_name") or "—"
            cluster = rec.get("cluster_id")
            cluster_str = str(cluster) if cluster is not None and cluster >= 0 else "noise"
            hover.append(
                f"<b>{gene}</b> ({n.name})<br>"
                f"{pname}<br>"
                f"IUPHAR: {iuphar}<br>"
                f"Status: <b>{status}</b><br>"
                f"UMAP cluster: {cluster_str}"
            )

        traces.append(
            go.Scatter(
                x=xs, y=ys,
                mode="markers",
                name=status,
                marker=dict(
                    color=STATUS_COLOR[status],
                    size=5,
                    opacity=0.9,
                    line=dict(width=0.3, color="white"),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Leaves not in DB (shouldn't happen, but guard)
    orphaned = [
        n for n in tree.get_terminals() if n.name not in records
    ]
    if orphaned:
        xs = [pos[n][0] for n in orphaned]
        ys = [pos[n][1] for n in orphaned]
        traces.append(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(color="#999", size=4),
            name="not in DB", hoverinfo="text",
            text=[n.name for n in orphaned],
        ))

    n_leaves = tree.count_terminals()
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(
            text=(
                "Human GPCR Phylogenetic Tree — ML (WAG + Γ)<br>"
                f"<sup>{n_leaves} sequences · {aligner_label} + VeryFastTree · "
                "coloured by orphan / cognate status</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(visible=False, range=[-1.15, 1.15]),
        yaxis=dict(visible=False, scaleanchor="x", range=[-1.15, 1.15]),
        legend_title="Status",
        width=1100,
        height=1100,
        template="plotly_white",
        hoverlabel=dict(font_size=12),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def output_paths(aligner: str) -> tuple[Path, Path, Path]:
    """
    Return (msa, nwk, html) paths for an aligner. MAFFT keeps the original
    unsuffixed names so existing artefacts on disk are preserved; the others
    embed the aligner name.
    """
    if aligner == "mafft":
        return (
            OUT_DIR / "gpcr.msa",
            OUT_DIR / "gpcr.nwk",
            OUT_DIR / "gpcr_tree.html",
        )
    return (
        OUT_DIR / f"gpcr.{aligner}.msa",
        OUT_DIR / f"gpcr.{aligner}.nwk",
        OUT_DIR / f"gpcr_tree.{aligner}.html",
    )


def run_pipeline(aligner: str, records: dict[str, dict], force: bool = False) -> None:
    label = ALIGNERS[aligner]["label"]
    msa_path, nwk_path, html_path = output_paths(aligner)

    print(f"\n══ Aligner: {label} ══")

    # 2. MSA
    if msa_path.exists() and not force:
        print(f"  Using cached MSA ({msa_path.name})")
    else:
        run_aligner(aligner, FASTA_PATH, msa_path)

    # 3. Tree
    if nwk_path.exists() and not force:
        print(f"  Using cached tree ({nwk_path.name})")
    else:
        run_fasttree(msa_path, nwk_path)

    # 4. Parse + root
    tree = Phylo.read(str(nwk_path), "newick")
    tree.root_at_midpoint()
    n_leaves   = tree.count_terminals()
    n_internal = len(list(tree.get_nonterminals()))
    print(f"  Tree: {n_leaves} leaves, {n_internal} internal nodes")

    # 5. Layout + plot
    fig = build_plotly_figure(tree, records, aligner_label=label)
    fig.write_html(str(html_path))
    print(f"  Saved: {html_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--aligner",
        choices=["mafft", "muscle", "clustalo", "kalign", "all"],
        default="mafft",
        help="Which MSA tool to run. 'all' runs every aligner sequentially.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run alignment and tree-building even if cached outputs exist.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print("── Step 1: Writing FASTA ──")
    records = load_and_write_fasta()

    aligners = list(ALIGNERS.keys()) if args.aligner == "all" else [args.aligner]
    for aligner in aligners:
        run_pipeline(aligner, records, force=args.force)

    print("\n── Done ─────────────────────────────────────────────")
    for aligner in aligners:
        _, _, html = output_paths(aligner)
        print(f"  {aligner:<10} → {html.name}")
    print("─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
