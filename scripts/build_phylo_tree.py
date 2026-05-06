#!/usr/bin/env python3
"""
Maximum-likelihood phylogenetic tree of human GPCRs.

Pipeline
--------
1. Load sequences from gpcr_sequences.db → FASTA
2. MAFFT (--auto) multiple sequence alignment
3. VeryFastTree (WAG + Gamma) → Newick tree
4. Biopython: midpoint-root
5. Circular cladogram layout → interactive plotly HTML
6. Colour leaves by consensus_status (orphan / cognate / unknown)

Outputs  (all in gpcr_sequence_db/)
-------
  gpcr.fasta        raw sequences
  gpcr.msa          MAFFT alignment
  gpcr.nwk          ML Newick tree
  gpcr_tree.html    interactive circular cladogram
"""

import re
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
MSA_PATH   = OUT_DIR / "gpcr.msa"
NWK_PATH   = OUT_DIR / "gpcr.nwk"
HTML_PATH  = OUT_DIR / "gpcr_tree.html"

# ── Tools ─────────────────────────────────────────────────────────────────────
MAFFT       = "mafft"
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
# 2. MAFFT
# ─────────────────────────────────────────────────────────────────────────────

def run_mafft() -> None:
    cmd = [MAFFT, "--auto", "--thread", "-1", "--quiet", str(FASTA_PATH)]
    print(f"  Running: {' '.join(cmd)}")
    with open(MSA_PATH, "w") as out:
        result = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MAFFT failed:\n{result.stderr}")
    print(f"  MSA written → {MSA_PATH.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. VeryFastTree
# ─────────────────────────────────────────────────────────────────────────────

def run_fasttree() -> None:
    cmd = [VERYFASTTREE, "-wag", "-gamma", str(MSA_PATH)]
    print(f"  Running: {' '.join(cmd)}")
    with open(NWK_PATH, "w") as out:
        result = subprocess.run(
            cmd, stdout=out, stderr=subprocess.PIPE, text=True
        )
    if result.returncode != 0:
        raise RuntimeError(f"VeryFastTree failed:\n{result.stderr}")
    # Print a few summary lines from stderr (FastTree logs there)
    for line in result.stderr.splitlines()[-6:]:
        if line.strip():
            print(f"  [VFT] {line}")
    print(f"  Newick tree written → {NWK_PATH.name}")


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
                f"<sup>{n_leaves} sequences · MAFFT + VeryFastTree · "
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

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # 1. FASTA
    print("── Step 1: Writing FASTA ──")
    records = load_and_write_fasta()

    # 2. MSA
    if MSA_PATH.exists():
        print(f"\n── Step 2: Using cached MSA ({MSA_PATH.name}) ──")
    else:
        print("\n── Step 2: Running MAFFT ──")
        run_mafft()

    # 3. Tree
    if NWK_PATH.exists():
        print(f"\n── Step 3: Using cached tree ({NWK_PATH.name}) ──")
    else:
        print("\n── Step 3: Running VeryFastTree ──")
        run_fasttree()

    # 4. Parse + root
    print("\n── Step 4: Parsing and rooting tree ──")
    tree = Phylo.read(str(NWK_PATH), "newick")
    tree.root_at_midpoint()
    n_leaves  = tree.count_terminals()
    n_internal = len(list(tree.get_nonterminals()))
    print(f"  {n_leaves} leaves, {n_internal} internal nodes")

    # 5. Layout + plot
    print("\n── Step 5: Building circular cladogram ──")
    fig = build_plotly_figure(tree, records)
    fig.write_html(str(HTML_PATH))
    print(f"  Saved: {HTML_PATH}")

    print(f"""
── Done ─────────────────────────────────────────────
  {FASTA_PATH.name:<30} raw sequences
  {MSA_PATH.name:<30} MAFFT alignment
  {NWK_PATH.name:<30} ML Newick tree
  {HTML_PATH.name:<30} interactive plot
─────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
