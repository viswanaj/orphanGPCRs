#!/usr/bin/env python3
"""
evaluate_txgemma_lora.py — Score the fine-tuned LoRA adapter and write
side-by-side reports vs the pre-fine-tune baseline.

Two evals (PLAN.md §4)
----------------------
  1. Held-out validation receptors — CB1R / HT2AR / DRD2 / GLP1R.
     Same compounds, same prompts, same metrics as
     scripts/txgemma_ligand_prediction.py — only difference is the LoRA
     adapter loaded on top of google/txgemma-2b-predict. Reuses the cached
     ChEMBL ligand CSV produced by that script.

  2. Cold-target test fold —
     gpcr_sequence_db/txgemma-finetune/data/target_split/test.csv
     If reports/baseline_pretrain.csv exists, the eval is restricted to the
     same (uniprot, molecule_id) rows so the A/B comparison is exact. Else
     a per-target sub-sample is used (cap controlled by --max-rows-per-target).

Outputs (gpcr_sequence_db/txgemma-finetune/reports/)
----------------------------------------------------
  lora_holdout_predictions.csv      per-compound preds (4 receptors)
  lora_holdout_report.html          Plotly scatter + ROC per receptor
  holdout_comparison.txt            ascii summary, baseline vs LoRA
  lora_coldtarget_predictions.csv   per-compound preds (cold-target test)
  lora_coldtarget_summary.txt       per-UniProt + overall metrics
  coldtarget_comparison.txt         baseline vs LoRA per UniProt + overall

Dependencies (in addition to requirements.txt)
----------------------------------------------
  pip install peft bitsandbytes accelerate

Usage
-----
  python scripts/evaluate_txgemma_lora.py
  python scripts/evaluate_txgemma_lora.py --no-qlora --device mps --batch-size 1
  python scripts/evaluate_txgemma_lora.py --skip-coldtarget
  python scripts/evaluate_txgemma_lora.py --skip-holdout
  python scripts/evaluate_txgemma_lora.py --adapter-dir path/to/adapter
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Reuse helpers from the existing pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from txgemma_ligand_prediction import (  # noqa: E402
    MAX_NEW_TOKENS,
    MODEL_NAME,
    VALIDATION_TARGETS,
    _parse_float,
    build_report,
    compute_metrics,
    fetch_all_ligands,
    format_prompt,
    get_hf_token,
    load_sequences_from_db,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
FT_ROOT     = REPO_ROOT / "gpcr_sequence_db" / "txgemma-finetune"
ADAPTER_DIR = FT_ROOT / "adapter" / "final"
TEST_CSV    = FT_ROOT / "data" / "target_split" / "test.csv"
REPORTS_DIR = FT_ROOT / "reports"

BASELINE_HOLDOUT_CSV = REPO_ROOT / "gpcr_sequence_db" / "txgemma" / "predictions.csv"
BASELINE_COLD_CSV    = REPORTS_DIR / "baseline_pretrain.csv"

LORA_HOLDOUT_PREDS_CSV = REPORTS_DIR / "lora_holdout_predictions.csv"
LORA_HOLDOUT_HTML      = REPORTS_DIR / "lora_holdout_report.html"
HOLDOUT_COMPARE_TXT    = REPORTS_DIR / "holdout_comparison.txt"

LORA_COLD_PREDS_CSV    = REPORTS_DIR / "lora_coldtarget_predictions.csv"
LORA_COLD_SUMMARY_TXT  = REPORTS_DIR / "lora_coldtarget_summary.txt"
COLD_COMPARE_TXT       = REPORTS_DIR / "coldtarget_comparison.txt"


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--adapter-dir", default=str(ADAPTER_DIR),
                   help="Path to the trained LoRA adapter dir.")
    p.add_argument("--model-name",  default=MODEL_NAME,
                   help="Base model name on Hugging Face.")
    p.add_argument("--no-qlora",    action="store_true",
                   help="Disable 4-bit QLoRA inference; load base in bf16.")
    p.add_argument("--device",      default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--batch-size",  type=int, default=8,
                   help="Generation batch size (auto-forced to 1 on MPS).")
    p.add_argument("--max-input-len", type=int, default=1024)

    p.add_argument("--skip-holdout",    action="store_true")
    p.add_argument("--skip-coldtarget", action="store_true")

    p.add_argument("--coldtarget-test-csv",     default=str(TEST_CSV))
    p.add_argument("--baseline-holdout-csv",    default=str(BASELINE_HOLDOUT_CSV))
    p.add_argument("--baseline-coldtarget-csv", default=str(BASELINE_COLD_CSV))
    p.add_argument("--max-rows-per-target", type=int, default=20,
                   help="If baseline cold-target preds aren't present, cap this many "
                        "rows per UniProt (top + bottom by pChEMBL). Set 0 to use all.")
    return p.parse_args()


# ── Model loading ────────────────────────────────────────────────────────────

def resolve_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(args: argparse.Namespace, hf_token: str):
    device = resolve_device(args.device)
    print(f"  Device: {device}")

    print(f"  Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Prompt ends with "Answer:" — keep that tail when truncating.
    tokenizer.truncation_side = "left"
    # Decoder-only batched generation requires left padding.
    tokenizer.padding_side = "left"

    use_qlora = (not args.no_qlora) and device == "cuda"
    if not args.no_qlora and device != "cuda":
        print(f"  4-bit QLoRA requires CUDA; falling back to bf16 on {device}.")

    if use_qlora:
        print("  Loading base in 4-bit (QLoRA)…")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb,
            device_map="auto",
            token=hf_token,
        )
    else:
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        print(f"  Loading base in {dtype}…")
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=dtype, token=hf_token,
        )
        if device != "cpu":
            base = base.to(device)

    adapter_path = Path(args.adapter_dir)
    if not adapter_path.exists():
        sys.exit(
            f"ERROR: adapter dir not found at {adapter_path}.\n"
            f"       Run scripts/train_txgemma_lora.py first."
        )
    print(f"  Attaching LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    # MPS batched generation is buggy in current torch builds (only batch[0]
    # generates real tokens). Force batch_size=1 there.
    if device == "mps" and args.batch_size > 1:
        print(f"  WARN: MPS batch generation bug; forcing batch_size=1 "
              f"(was {args.batch_size}).")
        args.batch_size = 1

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Model ready — {n_params:.1f}B params (LoRA attached)")
    return model, tokenizer, device


# ── Inference ────────────────────────────────────────────────────────────────

def predict_scores(
    model,
    tokenizer,
    device: str,
    prompts: list[str],
    batch_size: int,
    max_input_len: int,
) -> list[float | None]:
    scores: list[float | None] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Predicting", unit="batch"):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        )
        if device != "cpu":
            enc = enc.to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for ids in out[:, input_len:]:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            scores.append(_parse_float(text))
    return scores


# ── Metric helpers ───────────────────────────────────────────────────────────

def _per_receptor_metrics(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rec in VALIDATION_TARGETS:
        sub = [r for r in rows if r.get("receptor") == rec]
        if not sub:
            continue
        out[rec] = compute_metrics(
            [float(r["pchembl_value"])   for r in sub],
            [float(r["predicted_score"]) for r in sub],
        )
    return out


def _per_uniprot_metrics(rows: list[dict]) -> tuple[dict[str, dict], dict]:
    bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        bucket[r["uniprot"]].append(r)
    per: dict[str, dict] = {}
    for u, sub in bucket.items():
        per[u] = compute_metrics(
            [float(r["pchembl_value"])   for r in sub],
            [float(r["predicted_score"]) for r in sub],
        )
    overall = compute_metrics(
        [float(r["pchembl_value"])   for r in rows],
        [float(r["predicted_score"]) for r in rows],
    )
    return per, overall


def _fnum(x, sign: bool = False, width: int = 7, prec: int = 3) -> str:
    """NaN-safe float formatter for ascii tables."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return f"{'nan':>{width}}"
    fmt = f"{{:>+{width}.{prec}f}}" if sign else f"{{:>{width}.{prec}f}}"
    return fmt.format(x)


# ── Holdout (4 receptors) ────────────────────────────────────────────────────

def run_holdout_eval(model, tokenizer, device: str, args: argparse.Namespace) -> None:
    print("\n── Holdout validation: CB1R / HT2AR / DRD2 / GLP1R ──")
    sequences = load_sequences_from_db()
    rows = fetch_all_ligands(sequences)
    print(f"  {len(rows)} compounds across {len(VALIDATION_TARGETS)} receptors")

    prompts = [format_prompt(r["smiles"], r["sequence"]) for r in rows]
    preds = predict_scores(
        model, tokenizer, device, prompts,
        batch_size=args.batch_size, max_input_len=args.max_input_len,
    )

    result_rows: list[dict] = []
    n_failed = 0
    for r, p in zip(rows, preds):
        if p is None:
            n_failed += 1
            continue
        rr = dict(r)
        rr["pchembl_value"]   = float(rr["pchembl_value"])
        rr["predicted_score"] = float(p)
        result_rows.append(rr)
    print(f"  parsed {len(result_rows)} / {len(rows)}  ({n_failed} unparseable)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "receptor", "gpcr_class", "uniprot", "molecule_id",
        "smiles", "pchembl_value", "assay_type", "predicted_score",
    ]
    with LORA_HOLDOUT_PREDS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(result_rows)
    print(f"  Saved: {LORA_HOLDOUT_PREDS_CSV.relative_to(REPO_ROOT)}")

    if result_rows:
        fig = build_report(result_rows)
        fig.write_html(str(LORA_HOLDOUT_HTML))
        print(f"  Report: {LORA_HOLDOUT_HTML.relative_to(REPO_ROOT)}")

    write_holdout_comparison(result_rows, Path(args.baseline_holdout_csv))


def write_holdout_comparison(lora_rows: list[dict], baseline_csv: Path) -> None:
    print("\n  ── Holdout: baseline vs LoRA ──")
    lora = _per_receptor_metrics(lora_rows)

    if not baseline_csv.exists():
        print(f"  Baseline preds not found at {baseline_csv}.")
        print(f"  Tip: run scripts/txgemma_ligand_prediction.py to generate them.")
        baseline: dict[str, dict] = {}
    else:
        baseline_rows: list[dict] = []
        with baseline_csv.open() as f:
            for r in csv.DictReader(f):
                r["pchembl_value"]   = float(r["pchembl_value"])
                r["predicted_score"] = float(r["predicted_score"])
                baseline_rows.append(r)
        baseline = _per_receptor_metrics(baseline_rows)
        print(f"  Baseline rows loaded from {baseline_csv.relative_to(REPO_ROOT)}")

    L: list[str] = []
    L.append("# Held-out validation: baseline vs LoRA fine-tune")
    L.append("# Receptors: CB1R, HT2AR, DRD2, GLP1R (never seen in training)")
    L.append("")
    header = (f"{'Receptor':<8} {'n':>5}  "
              f"{'R_base':>8} {'R_lora':>8} {'ΔR':>8}  "
              f"{'ρ_base':>8} {'ρ_lora':>8} {'Δρ':>8}  "
              f"{'AUC_b':>7} {'AUC_l':>7} {'ΔAUC':>7}")
    L.append(header)
    L.append("-" * len(header))
    for rec in VALIDATION_TARGETS:
        l = lora.get(rec)
        b = baseline.get(rec)
        if l is None and b is None:
            continue
        n  = (l or b)["n"]
        Rb = b["pearson_r"]  if b else float("nan")
        Rl = l["pearson_r"]  if l else float("nan")
        rb = b["spearman_r"] if b else float("nan")
        rl = l["spearman_r"] if l else float("nan")
        ab = b["auc_roc"]    if b else float("nan")
        al = l["auc_roc"]    if l else float("nan")
        L.append(
            f"{rec:<8} {n:>5}  "
            f"{_fnum(Rb, sign=True, width=8)} {_fnum(Rl, sign=True, width=8)} {_fnum(Rl-Rb, sign=True, width=8)}  "
            f"{_fnum(rb, sign=True, width=8)} {_fnum(rl, sign=True, width=8)} {_fnum(rl-rb, sign=True, width=8)}  "
            f"{_fnum(ab,            width=7)} {_fnum(al,            width=7)} {_fnum(al-ab, sign=True, width=7)}"
        )

    HOLDOUT_COMPARE_TXT.write_text("\n".join(L) + "\n")
    print(f"  Wrote: {HOLDOUT_COMPARE_TXT.relative_to(REPO_ROOT)}")
    for line in L:
        print(f"    {line}")


# ── Cold-target test fold ────────────────────────────────────────────────────

def _subsample_per_target(rows: list[dict], cap: int) -> list[dict]:
    """Keep top + bottom `cap` rows per UniProt by pChEMBL (mirrors how the
    baseline sub-sample is structured: a mix of strong binders and weak
    binders so AUC is well-defined per target)."""
    bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        bucket[r["uniprot"]].append(r)
    out: list[dict] = []
    for u, lst in bucket.items():
        if len(lst) <= cap:
            out.extend(lst); continue
        lst_sorted = sorted(lst, key=lambda r: -float(r["pchembl_value"]))
        n_top = cap // 2
        n_bot = cap - n_top
        top = lst_sorted[:n_top]
        bot = lst_sorted[-n_bot:]
        seen = {id(r) for r in top}
        out.extend(top)
        out.extend(r for r in bot if id(r) not in seen)
    return out


def run_coldtarget_eval(model, tokenizer, device: str, args: argparse.Namespace) -> None:
    print("\n── Cold-target test fold ──")
    test_csv = Path(args.coldtarget_test_csv)
    if not test_csv.exists():
        print(f"  Skip: {test_csv} not found.")
        return

    rows: list[dict] = []
    with test_csv.open() as f:
        for r in csv.DictReader(f):
            r["pchembl_value"] = float(r["pchembl_value"])
            if r.get("score"):
                r["score"] = int(r["score"])
            rows.append(r)
    print(f"  Test fold: {len(rows)} rows, {len({r['uniprot'] for r in rows})} targets")

    baseline_csv = Path(args.baseline_coldtarget_csv)
    baseline_by_key: dict[tuple[str, str], dict] = {}
    if baseline_csv.exists():
        with baseline_csv.open() as f:
            for r in csv.DictReader(f):
                r["pchembl_value"]   = float(r["pchembl_value"])
                r["predicted_score"] = float(r["predicted_score"])
                key = (r["uniprot"], r.get("molecule_id", ""))
                baseline_by_key[key] = r
        baseline_keys = set(baseline_by_key)
        print(f"  Baseline: {len(baseline_keys)} rows from "
              f"{baseline_csv.relative_to(REPO_ROOT)}")
        rows = [r for r in rows
                if (r["uniprot"], r.get("molecule_id", "")) in baseline_keys]
        print(f"  Restricted to baseline subset → {len(rows)} rows for direct A/B")
    elif args.max_rows_per_target and args.max_rows_per_target > 0:
        rows = _subsample_per_target(rows, args.max_rows_per_target)
        print(f"  Capped to {args.max_rows_per_target}/target → {len(rows)} rows")

    if not rows:
        print("  No rows to score; aborting cold-target eval.")
        return

    # Use the prebuilt prompt — same string the model was trained on.
    prompts = [r["prompt"] for r in rows]
    preds = predict_scores(
        model, tokenizer, device, prompts,
        batch_size=args.batch_size, max_input_len=args.max_input_len,
    )

    out_rows: list[dict] = []
    n_failed = 0
    for r, p in zip(rows, preds):
        if p is None:
            n_failed += 1
            continue
        out_rows.append({
            "uniprot":         r["uniprot"],
            "chembl_target":   r.get("chembl_target", ""),
            "molecule_id":     r.get("molecule_id", ""),
            "smiles":          r["smiles"],
            "pchembl_value":   r["pchembl_value"],
            "score":           r.get("score", ""),
            "assay_type":      r.get("assay_type", ""),
            "predicted_score": float(p),
        })
    print(f"  parsed {len(out_rows)} / {len(rows)}  ({n_failed} unparseable)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with LORA_COLD_PREDS_CSV.open("w", newline="") as f:
        fields = [
            "uniprot", "chembl_target", "molecule_id", "smiles",
            "pchembl_value", "score", "assay_type", "predicted_score",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"  Saved: {LORA_COLD_PREDS_CSV.relative_to(REPO_ROOT)}")

    if not out_rows:
        print("  Nothing to summarize.")
        return

    write_coldtarget_summary(out_rows)
    if baseline_by_key:
        write_coldtarget_comparison(out_rows, baseline_by_key)


def write_coldtarget_summary(rows: list[dict]) -> None:
    per, overall = _per_uniprot_metrics(rows)
    scores = np.array([r["predicted_score"] for r in rows])
    L: list[str] = []
    L.append("# LoRA txGemma-2b-predict on cold-target test fold")
    L.append("")
    L.append(f"Overall:  n={overall['n']}  "
             f"R={_fnum(overall['pearson_r'], sign=True).strip()}  "
             f"ρ={_fnum(overall['spearman_r'], sign=True).strip()}  "
             f"AUC={_fnum(overall['auc_roc']).strip()}")
    L.append(f"Score distribution: min={int(scores.min())}  "
             f"median={int(np.median(scores))}  max={int(scores.max())}  "
             f"std={scores.std():.1f}  distinct={len(set(scores))}/{len(scores)}")
    L.append("")
    L.append(f"{'UniProt':<8} {'n':>5}  {'R':>8}  {'ρ':>8}  {'AUC':>7}")
    L.append("-" * 44)
    for u in sorted(per):
        m = per[u]
        L.append(
            f"{u:<8} {m['n']:>5}  "
            f"{_fnum(m['pearson_r'],  sign=True, width=8)}  "
            f"{_fnum(m['spearman_r'], sign=True, width=8)}  "
            f"{_fnum(m['auc_roc'],               width=7)}"
        )
    LORA_COLD_SUMMARY_TXT.write_text("\n".join(L) + "\n")
    print(f"  Wrote: {LORA_COLD_SUMMARY_TXT.relative_to(REPO_ROOT)}")


def write_coldtarget_comparison(
    lora_rows: list[dict],
    baseline_by_key: dict[tuple[str, str], dict],
) -> None:
    paired_lora: list[dict] = []
    paired_base: list[dict] = []
    for r in lora_rows:
        key = (r["uniprot"], r.get("molecule_id", ""))
        b = baseline_by_key.get(key)
        if b is None:
            continue
        paired_lora.append(r)
        paired_base.append(b)
    if not paired_lora:
        print("  No overlapping rows between LoRA and baseline; skipping comparison.")
        return

    per_lora, overall_lora = _per_uniprot_metrics(paired_lora)
    per_base, overall_base = _per_uniprot_metrics(paired_base)

    def fmt_row(label: str, n: int, base: dict, lora: dict) -> str:
        return (
            f"{label:<10} {n:>5}  "
            f"{_fnum(base['pearson_r'],  sign=True, width=8)} "
            f"{_fnum(lora['pearson_r'],  sign=True, width=8)} "
            f"{_fnum(lora['pearson_r']  - base['pearson_r'],  sign=True, width=8)}  "
            f"{_fnum(base['spearman_r'], sign=True, width=8)} "
            f"{_fnum(lora['spearman_r'], sign=True, width=8)} "
            f"{_fnum(lora['spearman_r'] - base['spearman_r'], sign=True, width=8)}  "
            f"{_fnum(base['auc_roc'],               width=7)} "
            f"{_fnum(lora['auc_roc'],               width=7)} "
            f"{_fnum(lora['auc_roc']    - base['auc_roc'],    sign=True, width=7)}"
        )

    header = (f"{'UniProt':<10} {'n':>5}  "
              f"{'R_base':>8} {'R_lora':>8} {'ΔR':>8}  "
              f"{'ρ_base':>8} {'ρ_lora':>8} {'Δρ':>8}  "
              f"{'AUC_b':>7} {'AUC_l':>7} {'ΔAUC':>7}")
    L: list[str] = []
    L.append("# Cold-target test fold: baseline vs LoRA (matched rows only)")
    L.append(f"# Pairs: {len(paired_lora)}")
    L.append("")
    L.append(header)
    L.append("-" * len(header))
    L.append(fmt_row("OVERALL", overall_lora["n"], overall_base, overall_lora))
    L.append("-" * len(header))
    for u in sorted(per_lora):
        if u not in per_base:
            continue
        L.append(fmt_row(u, per_lora[u]["n"], per_base[u], per_lora[u]))

    COLD_COMPARE_TXT.write_text("\n".join(L) + "\n")
    print(f"  Wrote: {COLD_COMPARE_TXT.relative_to(REPO_ROOT)}")
    # Print just the header + OVERALL to stdout for a quick glance.
    for line in L[:7]:
        print(f"    {line}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.skip_holdout and args.skip_coldtarget:
        sys.exit("Nothing to do: both --skip-holdout and --skip-coldtarget set.")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    hf_token = get_hf_token()
    print("── Loading model + LoRA adapter ──")
    model, tokenizer, device = load_model(args, hf_token)

    if not args.skip_holdout:
        run_holdout_eval(model, tokenizer, device, args)
    if not args.skip_coldtarget:
        run_coldtarget_eval(model, tokenizer, device, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
