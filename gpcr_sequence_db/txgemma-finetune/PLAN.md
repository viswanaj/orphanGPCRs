# Fine-tuning txGemma for GPCR–ligand affinity

## Motivation

`scripts/txgemma_ligand_prediction.py` validates `google/txgemma-2b-predict` on
four well-characterized GPCRs (CB1R, HT2AR, DRD2, GLP1R) with ChEMBL ligands.
The off-the-shelf model lands near random:

| Receptor | n | Pearson R | Spearman ρ | AUC |
|---|---:|---:|---:|---:|
| CB1R | 150 | −0.16 | −0.09 | 0.58 |
| HT2AR | 150 | −0.02 | −0.09 | 0.56 |
| DRD2 | 150 | +0.08 | +0.14 | 0.47 |
| GLP1R | 83 | −0.23 | +0.08 | 0.58 |

Predicted scores collapse into a narrow band (IQR 455–564 of the 0–1000 TDC
output) — the model is not differentiating within a single target family.
Goal of fine-tuning: lift AUC from ~0.55 to ~0.75+ on held-out GPCRs and
recover signed Spearman ≤ −0.4. Keep the TDC BindingDB_ki prompt format
unchanged so we stay in distribution.

## Held-out validation receptors

**CB1R, HT2AR, DRD2, GLP1R are excluded entirely from training data** so the
existing validation harness (`scripts/txgemma_ligand_prediction.py`) remains a
true test set. This gives a clean before/after comparison: same compounds,
same prompt, same metrics — the only variable is the fine-tuned adapter.

## 1. Dataset

**Source.** ChEMBL is the right base — same activity schema already in use.
Filter to:

- Targets where `target_type = "SINGLE PROTEIN"` and the UniProt is in
  `gpcr_sequences.db` (~800 human GPCRs)
- Binding assays only (`assay_type = "B"`), `pchembl_value` not null,
  human only (drop noisy rodent/mixed-species rows)
- Dedupe by (target, canonical SMILES) keeping the median pChEMBL across
  replicates (same approach as `fetch_chembl_activities`)
- **Exclude by UniProt accession** (not ChEMBL target ID): P21554 (CB1R),
  P28223 (HT2AR), P14416 (DRD2), P43220 (GLP1R). Some ChEMBL targets bundle
  multiple receptors or list cross-referenced allosteric assays, so the
  UniProt list is the safe filter.

**Expected size.** Class A GPCRs alone yield 200k–400k pairs in ChEMBL;
orphans contribute nearly zero (the point). Sufficient for LoRA — full
fine-tune is overkill at this scale.

**Label.** Convert pChEMBL → TDC 0–1000 normalized score so the loss is on
the model's native output:

```
score = 1000 * (1 - (pChEMBL - 4) / 8)   clipped to [0, 1000]
```

pChEMBL 4 → 1000 (weakest), pChEMBL 12 → 0 (strongest). Format the target
token as a zero-padded 3-digit integer (`"452"`, `"008"`) matching what the
model already emits.

## 2. Splits — escalating difficulty

1. **Random split** (sanity check)
2. **Cold-ligand split** — Bemis–Murcko scaffold split, no scaffold seen
   in both train and test
3. **Cold-target split** — hold out entire receptors; this is the number
   that predicts orphan-receptor performance

Report all three. The cold-target number is the only one that matters for
the orphan use case; expect a meaningful gap vs. random.

Note: held-out CB1R/HT2AR/DRD2/GLP1R are a *fourth, separate* eval set —
not used for split selection, only for the final before/after comparison.

## 3. Training recipe

- **Base model:** `google/txgemma-2b-predict`. Optionally benchmark
  `txgemma-9b-predict` zero-shot first as a cheaper sanity check.
- **Method:** QLoRA — 4-bit base + LoRA adapters on
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`,
  rank 16, alpha 32, dropout 0.05.
- **Loss:** causal LM cross-entropy on answer tokens only (mask the prompt).
- **Hyperparameters:** lr 2e-4, cosine schedule, 3 epochs, effective batch 32,
  max_seq_len 1024, bf16.
- **Compute:** 4–8 hrs on a single A100/H100. Not feasible on MPS — use
  Colab Pro / Lambda / Modal.
- **Eval cadence:** every 500 steps, inference on cold-target validation,
  log Spearman + AUC. Stop when AUC plateaus.

## 4. Evaluation harness

Reuse `compute_metrics` and `build_report` from
`scripts/txgemma_ligand_prediction.py`. Add a per-class rank-correlation panel
(Class A vs B/C) — generalization is likely uneven.

**The headline comparison:** rerun the existing validation script with the
LoRA adapter loaded, produce a side-by-side report (baseline vs fine-tuned)
on the same 4 held-out receptors.

## 5. Orphan-receptor inference

Once the adapter is trained and the held-out comparison looks good:

1. Iterate over orphan UniProts in `gpcr_sequences.db`
2. Score against a curated ligand library — ChEMBL drug-like compounds
   (~10k, filtered by Lipinski + PAINS) as a starting point
3. Rank top-K predicted hits per orphan; flag receptors where multiple
   chemically diverse scaffolds score high (harder to fake than a
   single-scaffold hit)

## 6. Risks

- **Optimistic eval.** ChEMBL biases toward tractable targets; cold-target
  AUC will read better than reality for orphans whose true ligand chemistry
  may be unlike anything in ChEMBL.
- **Olfactory GPCRs.** Most of the "unknown" orphans — they bind small
  volatiles, not drug-like chemistry. Consider excluding from the
  orphan-inference pass or treating as a separate track.
- **0–1000 quantization** caps resolution at ~0.008 log units. Acceptable,
  but never claim sub-quantile precision.

## 7. Folder structure

Code lives in `scripts/` (alongside the existing pipelines); data artifacts
live in `gpcr_sequence_db/txgemma-finetune/`.

```
scripts/
├── prepare_finetune_data.py    # ChEMBL pull, dedup, split, hold-out filter
├── train_txgemma_lora.py       # QLoRA training loop
└── evaluate_txgemma_lora.py    # rerun validation with adapter, A/B report

gpcr_sequence_db/txgemma-finetune/
├── PLAN.md                     # this file
├── data/                       # train/val/test CSVs + split metadata
├── adapter/                    # LoRA weights + tokenizer config
└── reports/                    # before/after HTML comparisons
```

## 8. First deliverable

Before any GPU spend: `prepare_training_data.py` produces the three split
CSVs and prints class-balance / target-coverage stats. That alone often
reveals whether the dataset is rich enough — and is reusable regardless of
which base model wins.
