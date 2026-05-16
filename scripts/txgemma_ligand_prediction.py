#!/usr/bin/env python3
"""
txGemma-2b-predict binding affinity validation on four well-studied GPCRs.

Validation receptors
--------------------
  CB1R   (Class A)  CHEMBL218   cannabinoid receptor 1
  HT2AR  (Class A)  CHEMBL224   serotonin 2A receptor
  DRD2   (Class A)  CHEMBL217   dopamine D2 receptor
  GLP1R  (Class B1) CHEMBL1784  glucagon-like peptide-1 receptor

Pipeline
--------
  Phase 1  Fetch top-N actives + top-N inactives per receptor from ChEMBL
  Phase 2  Load google/txgemma-2b-predict (MPS, bfloat16)
  Phase 3  Predict pKi for every (SMILES, sequence) pair in batches
  Phase 4  Evaluate (Pearson R, Spearman ρ, AUC-ROC) and write HTML report

Usage
-----
  export HF_TOKEN=hf_...
  python scripts/txgemma_ligand_prediction.py

Outputs (gpcr_sequence_db/txgemma/)
-------
  chembl_ligands.csv     raw ChEMBL data
  predictions.csv        predicted vs actual per compound
  validation_report.html  interactive plots
"""

import csv
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import requests
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
DB_PATH    = REPO_ROOT / "gpcr_sequence_db" / "gpcr_sequences.db"
OUT_DIR    = REPO_ROOT / "gpcr_sequence_db" / "txgemma"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIGANDS_CSV   = OUT_DIR / "chembl_ligands.csv"
PREDS_CSV     = OUT_DIR / "predictions.csv"
REPORT_HTML   = OUT_DIR / "validation_report.html"

# ── Validation receptors ──────────────────────────────────────────────────────
VALIDATION_TARGETS = {
    "CB1R":  {"chembl_id": "CHEMBL218",  "uniprot": "P21554", "class": "A"},
    "HT2AR": {"chembl_id": "CHEMBL224",  "uniprot": "P28223", "class": "A"},
    "DRD2":  {"chembl_id": "CHEMBL217",  "uniprot": "P14416", "class": "A"},
    "GLP1R": {"chembl_id": "CHEMBL1784", "uniprot": "P43220", "class": "B1"},
}

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "google/txgemma-2b-predict"
N_ACTIVE    = 75    # top actives per receptor (pChEMBL ≥ 6)
N_INACTIVE  = 75    # bottom inactives per receptor (pChEMBL ≤ 5)
BATCH_SIZE  = 1   # MPS batch generation bug: only item[0] generates; run one at a time
MAX_SEQ_LEN = 512   # truncate sequences longer than this
MAX_NEW_TOKENS = 16

ACTIVE_THRESHOLD   = 6.0   # pChEMBL ≥ 6  →  active  (Ki/IC50 ≤ 1 µM)
INACTIVE_THRESHOLD = 5.0   # pChEMBL ≤ 5  →  inactive

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — ChEMBL
# ─────────────────────────────────────────────────────────────────────────────

def _chembl_get(url: str, params: dict) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)


def fetch_chembl_activities(chembl_target_id: str, limit_per_page: int = 500) -> list[dict]:
    """Return all pChEMBL-valued activities for a target, deduplicated by SMILES."""
    activities = []
    offset = 0
    while True:
        data = _chembl_get(
            f"{CHEMBL_BASE}/activity",
            {
                "target_chembl_id": chembl_target_id,
                "pchembl_value__isnull": "false",
                "assay_type": "B",           # binding assays
                "target_organism": "Homo sapiens",
                "limit": limit_per_page,
                "offset": offset,
                "format": "json",
            },
        )
        batch = data.get("activities", [])
        activities.extend(batch)
        if len(batch) < limit_per_page:
            break
        offset += limit_per_page
        time.sleep(0.2)

    # Keep rows that have SMILES
    valid = [
        a for a in activities
        if a.get("canonical_smiles") and a.get("pchembl_value")
    ]

    # Deduplicate: keep one row per canonical SMILES (median pChEMBL)
    smiles_vals: dict[str, list[float]] = {}
    smiles_row: dict[str, dict] = {}
    for a in valid:
        s = a["canonical_smiles"]
        v = float(a["pchembl_value"])
        smiles_vals.setdefault(s, []).append(v)
        smiles_row[s] = a   # keep representative row

    deduped = []
    for s, vals in smiles_vals.items():
        row = dict(smiles_row[s])
        row["pchembl_value"] = float(np.median(vals))
        row["canonical_smiles"] = s
        deduped.append(row)

    return deduped


def select_compounds(activities: list[dict], n_active: int, n_inactive: int) -> list[dict]:
    """Select top actives + bottom inactives."""
    actives   = sorted(
        [a for a in activities if float(a["pchembl_value"]) >= ACTIVE_THRESHOLD],
        key=lambda x: -float(x["pchembl_value"])
    )[:n_active]

    inactives = sorted(
        [a for a in activities if float(a["pchembl_value"]) <= INACTIVE_THRESHOLD],
        key=lambda x: float(x["pchembl_value"])
    )[:n_inactive]

    return actives + inactives


def fetch_all_ligands(sequences: dict[str, str]) -> list[dict]:
    """Fetch and select compounds for all validation targets."""
    if LIGANDS_CSV.exists():
        print(f"  Using cached {LIGANDS_CSV.name}")
        rows = []
        with open(LIGANDS_CSV) as f:
            for row in csv.DictReader(f):
                row["pchembl_value"] = float(row["pchembl_value"])
                rows.append(row)
        return rows

    all_rows: list[dict] = []
    for name, info in VALIDATION_TARGETS.items():
        print(f"  Fetching {name} ({info['chembl_id']})…", end="", flush=True)
        acts = fetch_chembl_activities(info["chembl_id"])
        selected = select_compounds(acts, N_ACTIVE, N_INACTIVE)
        print(f"  {len(selected)} compounds ({len([s for s in selected if s['pchembl_value']>=ACTIVE_THRESHOLD])} active)")
        for a in selected:
            all_rows.append({
                "receptor":       name,
                "gpcr_class":     info["class"],
                "uniprot":        info["uniprot"],
                "chembl_target":  info["chembl_id"],
                "molecule_id":    a.get("molecule_chembl_id", ""),
                "smiles":         a["canonical_smiles"],
                "pchembl_value":  a["pchembl_value"],
                "assay_type":     a.get("standard_type", ""),
                "sequence":       sequences[info["uniprot"]],
            })

    with open(LIGANDS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Saved {LIGANDS_CSV.name}  ({len(all_rows)} total compounds)")
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — load txGemma
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_txgemma(hf_token: str) -> tuple:
    device = get_device()
    print(f"  Device: {device}")
    print(f"  Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

    print(f"  Loading model (bfloat16)…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    print(f"  Model loaded — {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")
    return model, tokenizer, device


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — predict
# ─────────────────────────────────────────────────────────────────────────────

_TDC_BINDINGDB_KI_PROMPT = (
    "Instructions: Answer the following question about drug target interactions.\n"
    "Context: Drug-target binding is the physical interaction between a drug "
    "and a specific biological molecule, such as a protein or enzyme. This "
    "interaction is essential for the drug to exert its pharmacological effect. "
    "The strength of the drug-target binding is determined by the binding "
    "affinity, which is a measure of how tightly the drug binds to the target. "
    "Ki is the equilibrium dissociation constant of an inhibitor. It is the "
    "concentration of inhibitor at which half of the target binding sites are "
    "occupied. A lower Ki value indicates a stronger binding affinity.\n"
    "Question: Given the target amino acid sequence and compound SMILES "
    "string, predict their normalized binding affinity Kd from 000 to 1000, "
    "where 000 is minimum Ki and 1000 is maximum Ki.\n"
    "Drug SMILES: {Drug SMILES}\n"
    "Target amino acid sequence: {Target amino acid sequence}\n"
    "Answer:"
)


def format_prompt(smiles: str, sequence: str) -> str:
    """
    Exact TDC BindingDB_ki prompt template that txGemma-predict was fine-tuned
    on (fetched from google/txgemma-2b-predict::tdc_prompts.json). The model
    emits a normalized 0–1000 score where 000 = minimum Ki (strongest binder)
    and 1000 = maximum Ki (weakest binder). The score is NOT pKi or nM — do
    not exponentiate or log-transform it; use rank-based metrics for evaluation.
    """
    seq = sequence[:MAX_SEQ_LEN]
    return (
        _TDC_BINDINGDB_KI_PROMPT
        .replace("{Drug SMILES}", smiles)
        .replace("{Target amino acid sequence}", seq)
    )


def _parse_float(text: str) -> float | None:
    """
    Extract the first non-negative number from generated text. TDC BindingDB_ki
    outputs are 000–1000 normalized scores (low = strong binder).
    """
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    matches = re.findall(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    for m in matches:
        val = float(m)
        if val >= 0:
            return val
    return None


def batch_predict(
    model,
    tokenizer,
    device: str,
    rows: list[dict],
) -> list[float | None]:
    """Run batched inference; returns predicted pKi per row."""
    prompts = [format_prompt(r["smiles"], r["sequence"]) for r in rows]
    predictions: list[float | None] = []

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Predicting", unit="batch"):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        for out in outputs:
            new_tokens = out[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            predictions.append(_parse_float(text))

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — evaluate + report
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(actual: list[float], predicted: list[float]) -> dict:
    """
    `actual` is pChEMBL (high = strong binder). `predicted` is the txGemma
    0–1000 normalized score (LOW = strong binder). So a well-calibrated model
    should produce NEGATIVE Pearson / Spearman with pChEMBL. We report signed
    values; |ρ| is the strength of monotone signal regardless of direction.
    For AUC we negate the score so high-score → active flips into the standard
    convention (label=1 means active, higher score → more active).
    """
    a = np.array(actual)
    p = np.array(predicted)
    pearson_r, pearson_p = stats.pearsonr(a, p)
    spearman_r, spearman_p = stats.spearmanr(a, p)

    labels, scores = [], []
    for av, pv in zip(actual, predicted):
        if av >= ACTIVE_THRESHOLD:
            labels.append(1); scores.append(-pv)   # negate: low pv = active
        elif av <= INACTIVE_THRESHOLD:
            labels.append(0); scores.append(-pv)

    auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")

    return {
        "pearson_r":  round(float(pearson_r), 3),
        "pearson_p":  round(float(pearson_p), 4),
        "spearman_r": round(float(spearman_r), 3),
        "spearman_p": round(float(spearman_p), 4),
        "auc_roc":    round(auc, 3),
        "n":          len(actual),
    }


def build_report(result_rows: list[dict]) -> go.Figure:
    """4-panel HTML report: scatter + ROC per receptor, grouped."""
    receptors = list(dict.fromkeys(r["receptor"] for r in result_rows))
    n = max(len(receptors), 1)

    fig = make_subplots(
        rows=2, cols=n,
        subplot_titles=[f"{rec} — scatter" for rec in receptors]
                      + [f"{rec} — ROC" for rec in receptors],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    STATUS_COLOR = {
        "active":   "#e74c3c",
        "inactive": "#2980b9",
        "middle":   "#bdc3c7",
    }

    for col, rec in enumerate(receptors, start=1):
        rows = [r for r in result_rows if r["receptor"] == rec]
        actual    = [r["pchembl_value"] for r in rows]
        predicted = [r["predicted_score"] for r in rows]
        colors    = [
            STATUS_COLOR["active"]   if a >= ACTIVE_THRESHOLD  else
            STATUS_COLOR["inactive"] if a <= INACTIVE_THRESHOLD else
            STATUS_COLOR["middle"]
            for a in actual
        ]

        m = compute_metrics(actual, predicted)

        hover = [
            f"{r['molecule_id']}<br>"
            f"Actual pChEMBL: {r['pchembl_value']:.2f}<br>"
            f"txGemma score (0–1000): {r['predicted_score']:.1f}<br>"
            f"SMILES: {r['smiles'][:40]}…"
            for r in rows
        ]

        # Scatter: pChEMBL vs txGemma score. A well-calibrated model produces
        # a downward-sloping line (high pChEMBL ↔ low score = strong binder).
        fig.add_trace(go.Scatter(
            x=actual, y=predicted, mode="markers",
            marker=dict(color=colors, size=6, opacity=0.7),
            text=hover, hovertemplate="%{text}<extra></extra>",
            name=rec, showlegend=False,
        ), row=1, col=col)

        suffix = "" if col == 1 else str(col)
        fig.add_annotation(
            xref=f"x{suffix} domain", yref=f"y{suffix} domain",
            x=0.05, y=0.95, xanchor="left", yanchor="top",
            text=(f"R={m['pearson_r']}  ρ={m['spearman_r']}<br>"
                  f"AUC={m['auc_roc']}  n={m['n']}"),
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
        )

        # ROC curve — negate score so higher = active (sklearn convention)
        labels, scores = [], []
        for r in rows:
            if r["pchembl_value"] >= ACTIVE_THRESHOLD:
                labels.append(1); scores.append(-r["predicted_score"])
            elif r["pchembl_value"] <= INACTIVE_THRESHOLD:
                labels.append(0); scores.append(-r["predicted_score"])

        if len(set(labels)) == 2:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(labels, scores)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color="#e74c3c", width=2),
                name=f"{rec} ROC", showlegend=False,
            ), row=2, col=col)

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#aaa", width=1),
            hoverinfo="none", showlegend=False,
        ), row=2, col=col)

    fig.update_layout(
        title=dict(
            text=(
                "txGemma-2b-predict — Binding Affinity Validation<br>"
                "<sup>TDC BindingDB_ki score (low=strong) vs ChEMBL pChEMBL · "
                "red = active (≥6) · blue = inactive (≤5)</sup>"
            ),
            x=0.5,
        ),
        height=800,
        width=1400,
        template="plotly_white",
    )
    for col in range(1, n + 1):
        fig.update_xaxes(title_text="Actual pChEMBL",        row=1, col=col)
        fig.update_yaxes(title_text="txGemma score (0–1000)", row=1, col=col)
        fig.update_xaxes(title_text="FPR", row=2, col=col)
        fig.update_yaxes(title_text="TPR", row=2, col=col)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_sequences_from_db() -> dict[str, str]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT uniprot_accession, sequence FROM gpcrs"
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def get_hf_token() -> str:
    """HF_TOKEN env > ~/.cache/huggingface/token > getpass."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists():
        tok = cached.read_text().strip()
        if tok:
            return tok
    import getpass
    return getpass.getpass("HuggingFace token: ")


def main() -> None:
    hf_token = get_hf_token()

    # ── Phase 1: ChEMBL ───────────────────────────────────────────────────
    print("── Phase 1: Fetching ChEMBL ligands ──")
    sequences = load_sequences_from_db()
    ligand_rows = fetch_all_ligands(sequences)
    print(f"  {len(ligand_rows)} compound–receptor pairs ready")

    # ── Phase 2: Load model ───────────────────────────────────────────────
    print("\n── Phase 2: Loading txGemma-2b-predict ──")
    model, tokenizer, device = load_txgemma(hf_token)

    # ── Phase 3: Predict ──────────────────────────────────────────────────
    print("\n── Phase 3: Running predictions ──")
    predictions = batch_predict(model, tokenizer, device, ligand_rows)

    result_rows: list[dict] = []
    n_failed = 0
    for row, pred in zip(ligand_rows, predictions):
        if pred is None:
            n_failed += 1
            continue
        result_rows.append({**row, "predicted_score": pred})

    print(f"  {len(result_rows)} predictions parsed  ({n_failed} unparseable)")
    if not result_rows:
        print("  ERROR: no parseable predictions. Check model output format.")
        return

    scores = np.array([r["predicted_score"] for r in result_rows])
    print(f"  Score range: min={scores.min():.1f}  median={np.median(scores):.1f}  "
          f"max={scores.max():.1f}  std={scores.std():.2f}  "
          f"distinct={len(set(scores))}/{len(scores)}")

    with open(PREDS_CSV, "w", newline="") as f:
        fields = [
            "receptor", "gpcr_class", "uniprot", "molecule_id",
            "smiles", "pchembl_value", "assay_type", "predicted_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"  Saved: {PREDS_CSV.name}")

    # ── Phase 4: Evaluate + report ────────────────────────────────────────
    print("\n── Phase 4: Evaluating ──")
    print(f"{'Receptor':10s}  {'R':>6}  {'ρ':>6}  {'AUC':>6}  n")
    for rec in VALIDATION_TARGETS:
        rows = [r for r in result_rows if r["receptor"] == rec]
        if not rows:
            continue
        m = compute_metrics(
            [r["pchembl_value"] for r in rows],
            [r["predicted_score"] for r in rows],
        )
        print(
            f"{rec:10s}  {m['pearson_r']:>+6.3f}  {m['spearman_r']:>+6.3f}  "
            f"{m['auc_roc']:>6.3f}  {m['n']}"
        )

    if not result_rows:
        print("  No results to plot.")
        return

    fig = build_report(result_rows)
    fig.write_html(str(REPORT_HTML))
    print(f"\n  Report: {REPORT_HTML}")


if __name__ == "__main__":
    main()
