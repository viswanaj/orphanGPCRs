#!/usr/bin/env python3
"""
evaluate_baseline.py — Baseline txGemma-2b-predict on the cold-target test
fold, BEFORE fine-tuning.

This is the reference point the fine-tuned LoRA adapter will be measured
against. The cold-target test fold (gpcr_sequence_db/txgemma-finetune/data/
target_split/test.csv) contains 23 receptors that are disjoint from the
training receptors and from the four held-out validation receptors.

Subsamples the 26k-row test fold stratified by (receptor, pChEMBL bucket)
to keep MPS inference tractable: per receptor we take the top actives,
bottom inactives, and a few middle-band compounds.

Outputs
-------
  gpcr_sequence_db/txgemma-finetune/reports/baseline_pretrain.csv
  gpcr_sequence_db/txgemma-finetune/reports/baseline_pretrain.txt
"""

import csv
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

# Reuse the validated inference path from the existing pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from txgemma_ligand_prediction import (
    ACTIVE_THRESHOLD,
    INACTIVE_THRESHOLD,
    batch_predict,
    get_hf_token,
    load_txgemma,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_CSV  = REPO_ROOT / "gpcr_sequence_db" / "txgemma-finetune" / "data" / "target_split" / "test.csv"
DB_PATH   = REPO_ROOT / "gpcr_sequence_db" / "gpcr_sequences.db"
OUT_DIR   = REPO_ROOT / "gpcr_sequence_db" / "txgemma-finetune" / "reports"
OUT_CSV   = OUT_DIR / "baseline_pretrain.csv"
OUT_TXT   = OUT_DIR / "baseline_pretrain.txt"

# Stratified subsample budget per receptor. 23 receptors × 20 ≈ 460 predictions
# at roughly 5–10 s each on MPS → ~30–60 min runtime.
PER_RECEPTOR_ACTIVE   = 8
PER_RECEPTOR_INACTIVE = 8
PER_RECEPTOR_MIDDLE   = 4
SAMPLE_SEED           = 42


def load_test_rows() -> list[dict]:
    rows: list[dict] = []
    with open(TEST_CSV) as f:
        for r in csv.DictReader(f):
            r["pchembl_value"] = float(r["pchembl_value"])
            rows.append(r)
    return rows


def load_sequences() -> dict[str, str]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT uniprot_accession, sequence FROM gpcrs").fetchall()
    conn.close()
    return {u: s for u, s in rows if u and s}


def stratified_sample(rows: list[dict]) -> list[dict]:
    by_target: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_target[r["uniprot"]].append(r)

    rng = random.Random(SAMPLE_SEED)
    sampled: list[dict] = []
    for _, group in by_target.items():
        actives = sorted(
            [r for r in group if r["pchembl_value"] >= ACTIVE_THRESHOLD],
            key=lambda r: -r["pchembl_value"],
        )[:PER_RECEPTOR_ACTIVE]
        inactives = sorted(
            [r for r in group if r["pchembl_value"] <= INACTIVE_THRESHOLD],
            key=lambda r: r["pchembl_value"],
        )[:PER_RECEPTOR_INACTIVE]
        middle = [r for r in group if INACTIVE_THRESHOLD < r["pchembl_value"] < ACTIVE_THRESHOLD]
        rng.shuffle(middle)
        sampled.extend(actives + inactives + middle[:PER_RECEPTOR_MIDDLE])
    return sampled


def _signed_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    """Returns (Pearson R, Spearman ρ, AUC). Score-direction follows TDC: low=strong."""
    R, _   = stats.pearsonr(actual, predicted)
    rho, _ = stats.spearmanr(actual, predicted)
    labels, scores = [], []
    for av, pv in zip(actual, predicted):
        if av >= ACTIVE_THRESHOLD:
            labels.append(1); scores.append(-pv)
        elif av <= INACTIVE_THRESHOLD:
            labels.append(0); scores.append(-pv)
    auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")
    return float(R), float(rho), float(auc)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("── Loading cold-target test fold ──")
    rows = load_test_rows()
    print(f"  {len(rows)} rows, {len({r['uniprot'] for r in rows})} receptors")

    rows = stratified_sample(rows)
    print(f"  Subsampled to {len(rows)} rows "
          f"({len({r['uniprot'] for r in rows})} receptors covered)")

    print("\n── Attaching sequences ──")
    sequences = load_sequences()
    rows = [r for r in rows if r["uniprot"] in sequences]
    for r in rows:
        r["sequence"] = sequences[r["uniprot"]]
    print(f"  {len(rows)} rows with sequences")

    print("\n── Loading txGemma ──")
    hf_token = get_hf_token()
    model, tokenizer, device = load_txgemma(hf_token)

    print("\n── Running inference ──")
    predictions = batch_predict(model, tokenizer, device, rows)

    results: list[dict] = []
    n_failed = 0
    for r, pred in zip(rows, predictions):
        if pred is None:
            n_failed += 1
            continue
        results.append({
            "uniprot":         r["uniprot"],
            "chembl_target":   r["chembl_target"],
            "molecule_id":     r["molecule_id"],
            "smiles":          r["smiles"],
            "pchembl_value":   r["pchembl_value"],
            "score":           int(r["score"]),
            "assay_type":      r["assay_type"],
            "predicted_score": pred,
        })

    print(f"  {len(results)} parseable predictions ({n_failed} failed)")
    if not results:
        sys.exit("ERROR: no parseable predictions.")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"  Wrote {OUT_CSV.relative_to(REPO_ROOT)}")

    # Overall metrics.
    a = np.array([r["pchembl_value"]   for r in results])
    p = np.array([r["predicted_score"] for r in results])
    R, rho, auc = _signed_metrics(a, p)

    lines: list[str] = []
    lines.append("# Baseline txGemma-2b-predict on cold-target test fold (PRE-fine-tune)")
    lines.append("")
    lines.append(f"Overall:  n={len(results)}  R={R:+.3f}  ρ={rho:+.3f}  AUC={auc:.3f}")
    lines.append(f"Score distribution: min={p.min():.0f}  median={np.median(p):.0f}  "
                 f"max={p.max():.0f}  std={p.std():.1f}  distinct={len(set(p))}/{len(p)}")
    lines.append("")
    lines.append(f"{'UniProt':10s}  {'n':>4}  {'R':>7}  {'ρ':>7}  {'AUC':>7}")
    by_rec: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_rec[r["uniprot"]].append(r)
    for u in sorted(by_rec):
        rs = by_rec[u]
        if len(rs) < 4:
            lines.append(f"{u:10s}  {len(rs):>4}  (too few samples)")
            continue
        ar = np.array([r["pchembl_value"]   for r in rs])
        pr = np.array([r["predicted_score"] for r in rs])
        try:
            Rr, rhor, aucr = _signed_metrics(ar, pr)
        except Exception:
            Rr = rhor = aucr = float("nan")
        lines.append(f"{u:10s}  {len(rs):>4}  {Rr:>+7.3f}  {rhor:>+7.3f}  {aucr:>7.3f}")

    OUT_TXT.write_text("\n".join(lines))
    print("\n" + lines[2])
    print(f"  Per-receptor: {OUT_TXT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
