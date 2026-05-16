#!/usr/bin/env python3
"""
infer_orphan_ligands.py — Score orphan GPCRs against a curated drug-like
ligand library using the fine-tuned LoRA adapter.

Phase 1  Load orphan (and optionally "unknown") GPCR sequences from gpcr_sequences.db
Phase 2  Fetch or load a curated drug-like ligand library from ChEMBL
         (max_phase >= 2, Lipinski-filtered; PAINS-filtered if RDKit is installed)
Phase 3  Load google/txgemma-2b-predict + LoRA adapter
Phase 4  Score all (receptor, ligand) pairs, streaming results to CSV
Phase 5  Rank top-K per receptor; flag multi-scaffold hits; write reports

Outputs (gpcr_sequence_db/txgemma-finetune/orphan_inference/)
-------------------------------------------------------------
  ligand_library.csv         curated compound library (cached across runs)
  orphan_scores.csv          full (uniprot, molecule_id, smiles, score) table
  orphan_tophits.csv         top-K per receptor with Bemis-Murcko scaffold
  orphan_summary.html        interactive bubble chart per receptor (Plotly)
  orphan_multiscaffold.txt   receptors with chemically diverse predicted hits

Runtime note
------------
  Scoring N receptors × L ligands is N×L model calls.
  Example: 160 orphans × 5000 ligands = 800k pairs.
  On a single A100 with batch_size=16 and QLoRA: roughly 4–8 hours.
  Use --max-ligands 500 for a quick sanity check (~30 min on A100).

Dependencies (beyond requirements.txt)
---------------------------------------
  pip install peft bitsandbytes accelerate   # required for LoRA inference
  pip install rdkit                           # optional — enables PAINS + scaffold analysis

Usage
-----
  python scripts/infer_orphan_ligands.py
  python scripts/infer_orphan_ligands.py --include-unknown --top-k 30
  python scripts/infer_orphan_ligands.py --max-ligands 500 --no-qlora --device mps
  python scripts/infer_orphan_ligands.py --skip-scoring          # re-rank from saved CSV
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import requests
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from txgemma_ligand_prediction import (  # noqa: E402
    MAX_NEW_TOKENS,
    MODEL_NAME,
    _parse_float,
    format_prompt,
    get_hf_token,
)
from evaluate_txgemma_lora import load_model, predict_scores  # noqa: E402

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
DB_PATH     = REPO_ROOT / "gpcr_sequence_db" / "gpcr_sequences.db"
FT_ROOT     = REPO_ROOT / "gpcr_sequence_db" / "txgemma-finetune"
ADAPTER_DIR = FT_ROOT / "adapter" / "final"
OUT_DIR     = FT_ROOT / "orphan_inference"

LIGAND_LIB_CSV    = OUT_DIR / "ligand_library.csv"
ORPHAN_SCORES_CSV = OUT_DIR / "orphan_scores.csv"
ORPHAN_HITS_CSV   = OUT_DIR / "orphan_tophits.csv"
ORPHAN_HTML       = OUT_DIR / "orphan_summary.html"
MULTISCAFFOLD_TXT = OUT_DIR / "orphan_multiscaffold.txt"

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

SCORE_FIELDS = [
    "uniprot", "gene_name", "protein_name", "status", "cluster_id",
    "molecule_id", "smiles", "predicted_score",
]
HIT_FIELDS = SCORE_FIELDS + ["rank", "scaffold_smiles"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    g = p.add_argument_group("model")
    g.add_argument("--adapter-dir",   default=str(ADAPTER_DIR))
    g.add_argument("--model-name",    default=MODEL_NAME)
    g.add_argument("--no-qlora",      action="store_true",
                   help="Disable 4-bit QLoRA; load base in bf16.")
    g.add_argument("--device",        default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    g.add_argument("--batch-size",    type=int, default=8)
    g.add_argument("--max-input-len", type=int, default=1024)

    g = p.add_argument_group("receptors")
    g.add_argument("--include-unknown", action="store_true",
                   help="Also score receptors with consensus_status='unknown' (443 extra).")

    g = p.add_argument_group("ligand library")
    g.add_argument("--min-phase",  type=int, default=2,
                   help="Minimum ChEMBL max_phase (default 2 = clinical + approved).")
    g.add_argument("--max-ligands", type=int, default=0,
                   help="Cap library size (0 = no cap). Set 500–2000 for quick runs.")
    g.add_argument("--no-pains",   action="store_true",
                   help="Skip PAINS filter even when RDKit is installed.")

    g = p.add_argument_group("ranking")
    g.add_argument("--top-k",         type=int, default=20,
                   help="Top-K ligands to keep per receptor.")
    g.add_argument("--min-scaffolds", type=int, default=3,
                   help="Min distinct Bemis-Murcko scaffolds in top-K to flag a receptor.")

    p.add_argument("--skip-scoring", action="store_true",
                   help="Load existing orphan_scores.csv and skip model inference.")
    return p.parse_args()


# ── Phase 1: Sequences ────────────────────────────────────────────────────────

def load_orphan_sequences(include_unknown: bool) -> list[dict]:
    statuses = ("orphan", "unknown") if include_unknown else ("orphan",)
    placeholders = ",".join("?" * len(statuses))
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        f"""
        SELECT uniprot_accession, gene_name, protein_name, sequence,
               consensus_status, cluster_id
        FROM gpcrs
        WHERE consensus_status IN ({placeholders})
        ORDER BY consensus_status, uniprot_accession
        """,
        statuses,
    ).fetchall()
    conn.close()
    return [
        {
            "uniprot":      r[0],
            "gene_name":    r[1] or "",
            "protein_name": r[2] or "",
            "sequence":     r[3],
            "status":       r[4],
            "cluster_id":   r[5] if r[5] is not None else "",
        }
        for r in rows
    ]


# ── Phase 2: Ligand library ───────────────────────────────────────────────────

def _chembl_get(url: str, params: dict) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def _passes_lipinski(props: dict) -> bool:
    try:
        return (
            float(props.get("mw_freebase") or 999) <= 500
            and int(props.get("hbd") or 99) <= 5
            and int(props.get("hba") or 99) <= 10
            and float(props.get("alogp") or 99) <= 5
        )
    except (TypeError, ValueError):
        return False


def _build_pains_catalog():
    if not HAVE_RDKIT:
        return None
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def fetch_ligand_library(min_phase: int, max_ligands: int, apply_pains: bool) -> list[dict]:
    if LIGAND_LIB_CSV.exists():
        print(f"  Loading cached library: {LIGAND_LIB_CSV.name}")
        rows: list[dict] = []
        with LIGAND_LIB_CSV.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
        if max_ligands and len(rows) > max_ligands:
            rows = rows[:max_ligands]
        print(f"  {len(rows)} compounds")
        return rows

    print(f"  Fetching ChEMBL molecules (max_phase >= {min_phase}, Lipinski filter)…")
    pains_cat = _build_pains_catalog() if (apply_pains and HAVE_RDKIT) else None
    if pains_cat:
        print("  PAINS filter: active")
    elif apply_pains and not HAVE_RDKIT:
        print("  PAINS filter: skipped (install rdkit to enable)")

    compounds: list[dict] = []
    offset = 0
    page_size = 1000
    total_fetched = 0

    while True:
        data = _chembl_get(
            f"{CHEMBL_BASE}/molecule",
            {
                "max_phase__gte": min_phase,
                "format":         "json",
                "limit":          page_size,
                "offset":         offset,
                "only": (
                    "molecule_chembl_id,pref_name,"
                    "molecule_structures,molecule_properties"
                ),
            },
        )
        batch = data.get("molecules", [])
        if not batch:
            break
        total_fetched += len(batch)

        for mol in batch:
            structs = mol.get("molecule_structures") or {}
            smiles = structs.get("canonical_smiles", "")
            if not smiles:
                continue
            props = mol.get("molecule_properties") or {}
            if not _passes_lipinski(props):
                continue
            if pains_cat:
                rdmol = Chem.MolFromSmiles(smiles)
                if rdmol is None or pains_cat.HasMatch(rdmol):
                    continue
            compounds.append({
                "molecule_id": mol.get("molecule_chembl_id", ""),
                "pref_name":   mol.get("pref_name", "") or "",
                "smiles":      smiles,
                "mw":          props.get("mw_freebase", ""),
                "hbd":         props.get("hbd", ""),
                "hba":         props.get("hba", ""),
                "alogp":       props.get("alogp", ""),
            })

        print(
            f"  Fetched {total_fetched} raw | Passed filters: {len(compounds)}", end="\r",
        )

        if len(batch) < page_size:
            break
        if max_ligands and len(compounds) >= max_ligands:
            compounds = compounds[:max_ligands]
            break
        offset += page_size
        time.sleep(0.15)

    print(f"\n  Library: {len(compounds)} compounds")
    if not compounds:
        return []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LIGAND_LIB_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(compounds[0].keys()))
        w.writeheader()
        w.writerows(compounds)
    print(f"  Cached: {LIGAND_LIB_CSV.name}")
    return compounds


# ── Phase 4: Scoring ──────────────────────────────────────────────────────────

def score_and_stream(
    model,
    tokenizer,
    device: str,
    receptors: list[dict],
    ligands: list[dict],
    batch_size: int,
    max_input_len: int,
    out_csv: Path,
) -> list[dict]:
    """Score all (receptor, ligand) pairs, writing each receptor's results immediately."""
    total = len(receptors) * len(ligands)
    print(f"  {len(receptors)} receptors × {len(ligands)} ligands = {total:,} pairs")

    all_rows: list[dict] = []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for i, rec in enumerate(receptors, 1):
            print(
                f"  [{i:3d}/{len(receptors)}] {rec['uniprot']}  {rec['gene_name'] or '—'}",
                flush=True,
            )
            prompts = [format_prompt(lig["smiles"], rec["sequence"]) for lig in ligands]
            raw_scores = predict_scores(
                model, tokenizer, device, prompts, batch_size, max_input_len,
            )
            n_ok = 0
            for lig, score in zip(ligands, raw_scores):
                if score is None:
                    continue
                row = {
                    "uniprot":         rec["uniprot"],
                    "gene_name":       rec["gene_name"],
                    "protein_name":    rec["protein_name"],
                    "status":          rec["status"],
                    "cluster_id":      rec["cluster_id"],
                    "molecule_id":     lig["molecule_id"],
                    "smiles":          lig["smiles"],
                    "predicted_score": float(score),
                }
                writer.writerow(row)
                all_rows.append(row)
                n_ok += 1
            print(f"         {n_ok}/{len(ligands)} scored", flush=True)

    return all_rows


# ── Phase 5: Rank + scaffold analysis ────────────────────────────────────────

def bemis_murcko_scaffold(smiles: str) -> str:
    if not HAVE_RDKIT:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def top_k_per_receptor(rows: list[dict], k: int) -> list[dict]:
    bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        bucket[r["uniprot"]].append(r)

    result: list[dict] = []
    for recs in bucket.values():
        top = sorted(recs, key=lambda r: float(r["predicted_score"]))[:k]
        for rank, r in enumerate(top, start=1):
            result.append({**r, "rank": rank, "scaffold_smiles": bemis_murcko_scaffold(r["smiles"])})
    return result


def find_multiscaffold_hits(tophits: list[dict], min_scaffolds: int) -> dict[str, set[str]]:
    bucket: dict[str, set[str]] = defaultdict(set)
    for r in tophits:
        sc = r.get("scaffold_smiles", "")
        if sc:
            bucket[r["uniprot"]].add(sc)
    return {u: sc for u, sc in bucket.items() if len(sc) >= min_scaffolds}


def write_multiscaffold_report(
    tophits: list[dict],
    multiscaffold: dict[str, set[str]],
    k: int,
    min_scaffolds: int,
    out_path: Path,
) -> None:
    rec_info: dict[str, dict] = {}
    rec_top: dict[str, list[dict]] = defaultdict(list)
    for r in tophits:
        u = r["uniprot"]
        rec_info.setdefault(u, r)
        rec_top[u].append(r)

    lines = [
        f"# Orphan GPCR multi-scaffold hits",
        f"# Criteria: top-{k} ligands, ≥{min_scaffolds} distinct Bemis-Murcko scaffolds",
        f"# {len(multiscaffold)} receptors flagged",
        "",
    ]
    for u in sorted(multiscaffold, key=lambda u: -len(multiscaffold[u])):
        info = rec_info[u]
        top = rec_top[u]
        lines.append(f"## {u}  {info['gene_name']}  ({info['status']})")
        lines.append(f"   {info['protein_name']}")
        lines.append(
            f"   Distinct scaffolds in top-{k}: {len(multiscaffold[u])}"
            f"  |  Best score: {top[0]['predicted_score']:.1f}"
            " (0 = strongest binder)"
        )
        lines.append(f"   Top-3 hits:")
        for r in top[:3]:
            mol_label = r["molecule_id"] or r["smiles"][:20]
            lines.append(
                f"     #{r['rank']:2d}  score={float(r['predicted_score']):.1f}"
                f"  {mol_label}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Wrote: {out_path.relative_to(REPO_ROOT)}")


def build_summary_html(
    tophits: list[dict],
    multiscaffold: dict[str, set[str]],
    out_path: Path,
) -> None:
    agg: dict[str, dict] = {}
    for r in tophits:
        u = r["uniprot"]
        if u not in agg:
            agg[u] = {
                "uniprot":      u,
                "gene_name":    r["gene_name"],
                "protein_name": r["protein_name"],
                "status":       r["status"],
                "best_score":   float(r["predicted_score"]),
                "scaffolds":    set(),
                "top3":         [],
            }
        sc = r.get("scaffold_smiles", "")
        if sc:
            agg[u]["scaffolds"].add(sc)
        if len(agg[u]["top3"]) < 3:
            agg[u]["top3"].append(r)

    recs = list(agg.values())
    x      = [r["best_score"] for r in recs]
    y      = [len(r["scaffolds"]) for r in recs]
    labels = [r["gene_name"] or r["uniprot"] for r in recs]
    colors = ["#e74c3c" if r["uniprot"] in multiscaffold else "#3498db" for r in recs]
    hover  = [
        (
            f"<b>{r['gene_name'] or r['uniprot']}</b> ({r['uniprot']})<br>"
            f"{r['protein_name'][:70]}<br>"
            f"Status: {r['status']}<br>"
            f"Best score: {r['best_score']:.1f} (lower = stronger binder)<br>"
            f"Distinct scaffolds in top-K: {len(r['scaffolds'])}<br>"
            + "".join(
                f"<br>#{h['rank']} {h['molecule_id'] or '—'}  "
                f"score={float(h['predicted_score']):.1f}"
                for h in r["top3"]
            )
        )
        for r in recs
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            color=colors, size=10, opacity=0.75,
            line=dict(width=1, color="white"),
        ),
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=(
                "Orphan GPCR — Predicted Hit Landscape (LoRA fine-tune)<br>"
                "<sup>"
                "x = best predicted score (0–1000, lower = stronger binder)  ·  "
                "y = distinct Bemis-Murcko scaffolds in top-K  ·  "
                "red = multi-scaffold flagged"
                "</sup>"
            ),
            x=0.5,
        ),
        xaxis_title="Best predicted score (lower = stronger predicted binder)",
        yaxis_title="Distinct scaffolds in top-K hits",
        height=700,
        width=1300,
        template="plotly_white",
    )
    fig.write_html(str(out_path))
    print(f"  Report: {out_path.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not HAVE_RDKIT:
        print("NOTE: RDKit not installed — PAINS filter and scaffold diversity disabled.\n")

    # Phase 1
    print("── Phase 1: Loading orphan GPCR sequences ──")
    receptors = load_orphan_sequences(args.include_unknown)
    n_orphan  = sum(1 for r in receptors if r["status"] == "orphan")
    n_unknown = sum(1 for r in receptors if r["status"] == "unknown")
    label = f"{n_orphan} orphan"
    if args.include_unknown:
        label += f" + {n_unknown} unknown"
    print(f"  {len(receptors)} receptors  ({label})")

    # Phase 2
    print("\n── Phase 2: Ligand library ──")
    ligands = fetch_ligand_library(
        min_phase=args.min_phase,
        max_ligands=args.max_ligands,
        apply_pains=not args.no_pains,
    )
    if not ligands:
        sys.exit("ERROR: ligand library is empty.")

    # Phase 3 + 4
    if args.skip_scoring and ORPHAN_SCORES_CSV.exists():
        print(f"\n── Skipping inference: loading {ORPHAN_SCORES_CSV.name} ──")
        scored_rows: list[dict] = []
        with ORPHAN_SCORES_CSV.open() as f:
            for r in csv.DictReader(f):
                r["predicted_score"] = float(r["predicted_score"])
                scored_rows.append(r)
        print(f"  {len(scored_rows):,} rows")
    else:
        if args.skip_scoring:
            print(f"  --skip-scoring set but {ORPHAN_SCORES_CSV.name} not found; running inference.")

        print("\n── Phase 3: Loading model + LoRA adapter ──")
        hf_token = get_hf_token()
        model, tokenizer, device = load_model(args, hf_token)

        print("\n── Phase 4: Scoring pairs ──")
        scored_rows = score_and_stream(
            model, tokenizer, device,
            receptors, ligands,
            batch_size=args.batch_size,
            max_input_len=args.max_input_len,
            out_csv=ORPHAN_SCORES_CSV,
        )
        print(f"  Saved: {ORPHAN_SCORES_CSV.relative_to(REPO_ROOT)}")

    if not scored_rows:
        sys.exit("No scored rows to analyze.")

    # Phase 5
    print(f"\n── Phase 5: Ranking (top-{args.top_k}) + scaffold analysis ──")
    tophits = top_k_per_receptor(scored_rows, k=args.top_k)

    with ORPHAN_HITS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HIT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(tophits)
    print(f"  Saved: {ORPHAN_HITS_CSV.relative_to(REPO_ROOT)}")

    multiscaffold = find_multiscaffold_hits(tophits, args.min_scaffolds)
    print(f"  Multi-scaffold receptors (≥{args.min_scaffolds} distinct scaffolds): "
          f"{len(multiscaffold)}")

    if HAVE_RDKIT and multiscaffold:
        write_multiscaffold_report(
            tophits, multiscaffold, args.top_k, args.min_scaffolds, MULTISCAFFOLD_TXT,
        )
    elif not HAVE_RDKIT:
        print("  Scaffold report skipped (install rdkit to enable).")

    build_summary_html(tophits, multiscaffold, ORPHAN_HTML)
    print("\nDone.")


if __name__ == "__main__":
    main()
