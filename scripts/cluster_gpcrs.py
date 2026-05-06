#!/usr/bin/env python3
"""
Sequence similarity, clustering, and visualisation of human GPCRs.

Pipeline
--------
1. Load sequences from gpcr_sequences.db
2. All-vs-all pairwise Needleman-Wunsch (parasail) → similarity matrix
3. UMAP(metric='precomputed') on the distance matrix → 2-D embedding
4. HDBSCAN on the embedding → cluster labels
5. Interactive plotly HTML coloured by consensus_status
6. Write UMAP coords + cluster IDs back to gpcr_sequences.db

Outputs
-------
  gpcr_sequence_db/similarity_matrix.npy   float32 840×840
  gpcr_sequence_db/gpcr_clusters.html      interactive scatter plot
  gpcr_sequences.db updated with umap_x, umap_y, cluster_id columns
"""

import sqlite3
import time
from pathlib import Path

import hdbscan
import numpy as np
import parasail
import plotly.graph_objects as go
import umap
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUT_DIR    = REPO_ROOT / "gpcr_sequence_db"
DB_PATH    = OUT_DIR / "gpcr_sequences.db"
SIM_PATH   = OUT_DIR / "similarity_matrix.npy"
HTML_PATH  = OUT_DIR / "gpcr_clusters.html"

# ── Alignment settings ────────────────────────────────────────────────────────
GAP_OPEN   = 10
GAP_EXTEND = 1
MATRIX     = parasail.blosum62

# ── UMAP settings ─────────────────────────────────────────────────────────────
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST    = 0.1
UMAP_RANDOM_SEED = 42

# ── HDBSCAN settings ──────────────────────────────────────────────────────────
MIN_CLUSTER_SIZE = 5
MIN_SAMPLES      = 3

# ── Visual settings ───────────────────────────────────────────────────────────
STATUS_COLOR = {
    "orphan":  "#e74c3c",   # red
    "cognate": "#2980b9",   # blue
    "unknown": "#bdc3c7",   # light grey
}
STATUS_ORDER = ["orphan", "cognate", "unknown"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_sequences() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT uniprot_accession, uniprot_id, gene_name, protein_name,
               sequence, sequence_length, consensus_status,
               iuphar_name, iuphar_target_id
        FROM gpcrs
        ORDER BY uniprot_accession
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pairwise similarity matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity(sequences: list[str]) -> np.ndarray:
    """
    All-vs-all Needleman-Wunsch alignment using parasail.
    Identity = matches / alignment_length (nw_stats, no traceback needed).
    Returns float32 n×n matrix with 1.0 on the diagonal.
    """
    n     = len(sequences)
    sim   = np.eye(n, dtype=np.float32)
    total = n * (n - 1) // 2

    with tqdm(total=total, desc="Pairwise alignment", unit="pair") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                r = parasail.nw_stats_striped_sat(
                    sequences[i], sequences[j],
                    GAP_OPEN, GAP_EXTEND, MATRIX,
                )
                identity   = r.matches / max(r.length, 1)
                sim[i, j]  = identity
                sim[j, i]  = identity
                pbar.update(1)

    return sim


# ─────────────────────────────────────────────────────────────────────────────
# 3. UMAP
# ─────────────────────────────────────────────────────────────────────────────

def run_umap(distance_matrix: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_components  = 2,
        metric        = "precomputed",
        n_neighbors   = UMAP_N_NEIGHBORS,
        min_dist      = UMAP_MIN_DIST,
        random_state  = UMAP_RANDOM_SEED,
    )
    return reducer.fit_transform(distance_matrix)


# ─────────────────────────────────────────────────────────────────────────────
# 4. HDBSCAN
# ─────────────────────────────────────────────────────────────────────────────

def run_hdbscan(embedding: np.ndarray) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = MIN_CLUSTER_SIZE,
        min_samples      = MIN_SAMPLES,
    )
    return clusterer.fit_predict(embedding)   # -1 = noise


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plotly HTML
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(records: list[dict], embedding: np.ndarray, cluster_labels: np.ndarray) -> go.Figure:
    fig = go.Figure()

    for status in STATUS_ORDER:
        idx = [i for i, r in enumerate(records) if r["consensus_status"] == status]
        if not idx:
            continue

        x     = embedding[idx, 0]
        y     = embedding[idx, 1]
        clust = cluster_labels[idx]

        hover = [
            (
                f"<b>{records[i]['gene_name'] or records[i]['uniprot_id']}</b><br>"
                f"{records[i]['protein_name'][:60]}<br>"
                f"UniProt: {records[i]['uniprot_accession']}<br>"
                f"Status: <b>{records[i]['consensus_status']}</b><br>"
                f"Cluster: {int(c) if c >= 0 else 'noise'}<br>"
                f"Length: {records[i]['sequence_length']} aa"
            )
            for i, c in zip(idx, clust)
        ]

        fig.add_trace(go.Scatter(
            x           = x,
            y           = y,
            mode        = "markers",
            name        = status,
            marker      = dict(
                color   = STATUS_COLOR[status],
                size    = 7,
                opacity = 0.80,
                line    = dict(width=0.4, color="white"),
            ),
            text        = hover,
            hovertemplate = "%{text}<extra></extra>",
        ))

    n_clusters = len(set(cluster_labels[cluster_labels >= 0]))
    fig.update_layout(
        title = dict(
            text=(
                f"Human GPCR sequence similarity — UMAP + HDBSCAN<br>"
                f"<sup>840 sequences · {n_clusters} clusters · "
                f"coloured by consensus orphan/cognate status</sup>"
            ),
            x=0.5,
        ),
        xaxis_title = "UMAP 1",
        yaxis_title = "UMAP 2",
        legend_title = "Status",
        width  = 1000,
        height = 750,
        template = "plotly_white",
        hoverlabel = dict(font_size=12),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Write back to DB
# ─────────────────────────────────────────────────────────────────────────────

def update_db(records: list[dict], embedding: np.ndarray, cluster_labels: np.ndarray) -> None:
    conn = sqlite3.connect(DB_PATH)

    # Add columns if they don't exist yet
    for col, typ in [("umap_x", "REAL"), ("umap_y", "REAL"), ("cluster_id", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE gpcrs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass   # column already exists

    conn.executemany(
        "UPDATE gpcrs SET umap_x=?, umap_y=?, cluster_id=? WHERE uniprot_accession=?",
        [
            (float(embedding[i, 0]), float(embedding[i, 1]),
             int(cluster_labels[i]), records[i]["uniprot_accession"])
            for i in range(len(records))
        ],
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load
    print("── Step 1: Loading sequences from DB ──")
    records   = load_sequences()
    sequences = [r["sequence"] for r in records]
    print(f"  {len(records)} sequences loaded")

    # 2. Similarity matrix (or load cached)
    if SIM_PATH.exists():
        print(f"\n── Step 2: Loading cached similarity matrix ({SIM_PATH.name}) ──")
        sim = np.load(SIM_PATH)
        print(f"  Loaded {sim.shape[0]}×{sim.shape[1]} matrix")
    else:
        print("\n── Step 2: Computing all-vs-all pairwise similarity ──")
        t0  = time.time()
        sim = compute_similarity(sequences)
        np.save(SIM_PATH, sim)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s — saved to {SIM_PATH.name}")

    # 3. UMAP
    print("\n── Step 3: UMAP dimensionality reduction ──")
    distance  = (1.0 - sim).astype(np.float32)
    distance  = np.clip(distance, 0.0, 1.0)   # guard against float rounding
    embedding = run_umap(distance)
    print(f"  Embedding shape: {embedding.shape}")

    # 4. HDBSCAN
    print("\n── Step 4: HDBSCAN clustering ──")
    cluster_labels = run_hdbscan(embedding)
    n_clusters = len(set(cluster_labels[cluster_labels >= 0]))
    n_noise    = int((cluster_labels == -1).sum())
    print(f"  {n_clusters} clusters, {n_noise} noise points")

    # 5. Plot
    print("\n── Step 5: Building interactive plot ──")
    fig = make_plot(records, embedding, cluster_labels)
    fig.write_html(HTML_PATH)
    print(f"  Saved: {HTML_PATH}")

    # 6. DB update
    print("\n── Step 6: Writing UMAP coords + cluster IDs to DB ──")
    update_db(records, embedding, cluster_labels)
    print("  Done.")

    # Summary
    from collections import Counter
    status_counts = Counter(r["consensus_status"] for r in records)
    print(f"""
── Summary ───────────────────────────────────────────
  Sequences          : {len(records)}
  HDBSCAN clusters   : {n_clusters}
  Noise points       : {n_noise}
  Orphan             : {status_counts['orphan']}
  Cognate            : {status_counts['cognate']}
  Unknown            : {status_counts['unknown']}
──────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
