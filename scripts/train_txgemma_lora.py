#!/usr/bin/env python3
"""
train_txgemma_lora.py — QLoRA fine-tune txGemma-2b-predict on the cold-target
training fold.

Inputs
------
  gpcr_sequence_db/txgemma-finetune/data/target_split/train.csv
  gpcr_sequence_db/txgemma-finetune/data/target_split/val.csv

Both CSVs are produced by scripts/prepare_finetune_data.py. They already
contain the TDC BindingDB_ki prompt and the zero-padded 3-digit answer
string (`score_str`). Held-out receptors (CB1R/HT2AR/DRD2/GLP1R) are
absent by construction — see PLAN.md.

Outputs
-------
  gpcr_sequence_db/txgemma-finetune/adapter/
    config.json                 # run config + LoRA + tokenization details
    checkpoint-XXX/             # intermediate Trainer checkpoints
    final/                      # final LoRA weights + tokenizer
    runs/                       # TensorBoard event files

Dependencies (in addition to requirements.txt)
----------------------------------------------
  pip install peft bitsandbytes accelerate datasets tensorboard

Compute
-------
  Designed for a single CUDA GPU (A100/A10/T4/3090 etc.). Will not run on
  Apple MPS (bitsandbytes 4-bit kernels are CUDA-only). For a non-CUDA
  smoke test pass --no-qlora --cpu-debug, which loads the bf16 base on
  CPU; do NOT use that for a real run.

Loss masking
------------
  We tokenize prompt and answer separately and concatenate. Labels for
  prompt tokens are set to -100 so the cross-entropy loss is computed
  ONLY on the answer tokens (the digits + EOS). This matches how txGemma
  was originally fine-tuned on TDC tasks.

Usage
-----
  python scripts/train_txgemma_lora.py
  python scripts/train_txgemma_lora.py --epochs 5 --lr 1e-4
  python scripts/train_txgemma_lora.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Reuse helpers from the existing pipeline (HF token lookup).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from txgemma_ligand_prediction import get_hf_token  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "gpcr_sequence_db" / "txgemma-finetune" / "data"
TRAIN_CSV  = DATA_DIR / "target_split" / "train.csv"
VAL_CSV    = DATA_DIR / "target_split" / "val.csv"
ADAPTER_DIR = REPO_ROOT / "gpcr_sequence_db" / "txgemma-finetune" / "adapter"

MODEL_NAME = "google/txgemma-2b-predict"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Paths
    train_csv:  str = str(TRAIN_CSV)
    val_csv:    str = str(VAL_CSV)
    output_dir: str = str(ADAPTER_DIR)

    # Model
    model_name: str = MODEL_NAME
    use_qlora:  bool = True

    # LoRA
    lora_r:        int   = 16
    lora_alpha:    int   = 32
    lora_dropout:  float = 0.05
    lora_targets:  list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    epochs:           int   = 3
    lr:               float = 2e-4
    batch_size:       int   = 4
    grad_accum:       int   = 8        # effective batch = batch_size * grad_accum = 32
    max_seq_len:      int   = 1024
    warmup_ratio:     float = 0.03
    weight_decay:     float = 0.0
    lr_scheduler:     str   = "cosine"

    # Eval / save / log cadence
    eval_steps:       int   = 500
    save_steps:       int   = 500
    save_total_limit: int   = 3
    logging_steps:    int   = 50
    eval_max_rows:    int   = 1000     # cap val rows used per eval (LM loss only)

    # Misc
    seed:             int   = 42
    resume:           bool  = False
    cpu_debug:        bool  = False    # for syntax-level smoke tests only


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train-csv",     default=cfg.train_csv)
    p.add_argument("--val-csv",       default=cfg.val_csv)
    p.add_argument("--output-dir",    default=cfg.output_dir)
    p.add_argument("--model-name",    default=cfg.model_name)
    p.add_argument("--no-qlora",      action="store_true",
                   help="Disable 4-bit quantization; load base in bf16.")
    p.add_argument("--lora-r",        type=int,   default=cfg.lora_r)
    p.add_argument("--lora-alpha",    type=int,   default=cfg.lora_alpha)
    p.add_argument("--lora-dropout",  type=float, default=cfg.lora_dropout)
    p.add_argument("--epochs",        type=int,   default=cfg.epochs)
    p.add_argument("--lr",            type=float, default=cfg.lr)
    p.add_argument("--batch-size",    type=int,   default=cfg.batch_size)
    p.add_argument("--grad-accum",    type=int,   default=cfg.grad_accum)
    p.add_argument("--max-seq-len",   type=int,   default=cfg.max_seq_len)
    p.add_argument("--eval-steps",    type=int,   default=cfg.eval_steps)
    p.add_argument("--save-steps",    type=int,   default=cfg.save_steps)
    p.add_argument("--logging-steps", type=int,   default=cfg.logging_steps)
    p.add_argument("--eval-max-rows", type=int,   default=cfg.eval_max_rows)
    p.add_argument("--seed",          type=int,   default=cfg.seed)
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--cpu-debug",     action="store_true",
                   help="Load on CPU for a syntax/IO smoke test. NOT for real training.")
    a = p.parse_args()

    cfg.train_csv     = a.train_csv
    cfg.val_csv       = a.val_csv
    cfg.output_dir    = a.output_dir
    cfg.model_name    = a.model_name
    cfg.use_qlora     = not a.no_qlora
    cfg.lora_r        = a.lora_r
    cfg.lora_alpha    = a.lora_alpha
    cfg.lora_dropout  = a.lora_dropout
    cfg.epochs        = a.epochs
    cfg.lr            = a.lr
    cfg.batch_size    = a.batch_size
    cfg.grad_accum    = a.grad_accum
    cfg.max_seq_len   = a.max_seq_len
    cfg.eval_steps    = a.eval_steps
    cfg.save_steps    = a.save_steps
    cfg.logging_steps = a.logging_steps
    cfg.eval_max_rows = a.eval_max_rows
    cfg.seed          = a.seed
    cfg.resume        = a.resume
    cfg.cpu_debug     = a.cpu_debug
    return cfg


# ── Tokenization with answer-only loss masking ───────────────────────────────

def build_tokenize_fn(tokenizer, max_seq_len: int):
    """
    Tokenize `prompt` and `score_str` separately, concatenate, and emit
    `labels` where prompt tokens are -100 (no loss) and answer tokens are the
    actual ids (loss is computed). The answer is " <score_str>" + EOS — the
    leading space mirrors what naturally follows "Answer:" in the prompt.

    Truncation: prompts are already capped at MAX_SEQ_LEN=512 chars of protein
    sequence in prepare_finetune_data.py, so the joined ids almost always fit
    in max_seq_len=1024. If they don't, we left-truncate the prompt so the
    full answer is always present.
    """
    eos = tokenizer.eos_token_id
    assert eos is not None, "tokenizer must have an EOS token"

    def _tok(example: dict) -> dict:
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        answer_text = " " + example["score_str"]
        answer_ids  = tokenizer(answer_text, add_special_tokens=False)["input_ids"] + [eos]

        # Reserve room for the full answer; truncate the prompt from the left
        # (drop the start, keep the SMILES/sequence/Answer: tail) if needed.
        budget = max_seq_len - len(answer_ids)
        if budget < 1:
            # Pathological: answer alone exceeds budget. Skip by emitting empty.
            return {"input_ids": [], "labels": [], "attention_mask": []}
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]

        input_ids = prompt_ids + answer_ids
        labels    = [-100] * len(prompt_ids) + answer_ids
        attn      = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    return _tok


def build_dataset(csv_path: Path, tokenizer, max_seq_len: int) -> Dataset:
    ds = Dataset.from_csv(str(csv_path))
    # Defensive: every row must have score_str. score_str is written as a
    # string by prepare_finetune_data.py.
    ds = ds.filter(lambda r: r.get("score_str") is not None and r.get("prompt"))
    tok_fn = build_tokenize_fn(tokenizer, max_seq_len)
    ds = ds.map(tok_fn, remove_columns=ds.column_names, desc=f"Tokenizing {csv_path.name}")
    ds = ds.filter(lambda r: len(r["input_ids"]) > 0)
    return ds


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_model(cfg: TrainConfig, hf_token: str):
    print(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if cfg.cpu_debug:
        print("CPU debug mode — loading base in bf16 on CPU. Do NOT train this.")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, token=hf_token, dtype=torch.bfloat16,
        )
        return model, tokenizer

    if cfg.use_qlora:
        print("Loading base in 4-bit (QLoRA)")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb,
            device_map="auto",
            token=hf_token,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
        )
    else:
        print("Loading base in bf16 (no QLoRA)")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )
        model.gradient_checkpointing_enable()

    return model, tokenizer


def attach_lora(model, cfg: TrainConfig):
    print(f"Attaching LoRA: r={cfg.lora_r} α={cfg.lora_alpha} drop={cfg.lora_dropout}")
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora_targets),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Run config snapshot ──────────────────────────────────────────────────────

def save_run_config(cfg: TrainConfig, tokenizer, output_dir: Path) -> None:
    """Persist a reproducibility snapshot alongside the adapter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "config": asdict(cfg),
        "tokenizer": {
            "name_or_path":   tokenizer.name_or_path,
            "eos_token_id":   tokenizer.eos_token_id,
            "pad_token_id":   tokenizer.pad_token_id,
            "vocab_size":     tokenizer.vocab_size,
        },
        "torch_version":       torch.__version__,
        "cuda_available":      torch.cuda.is_available(),
        "device_name":         (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(snapshot, indent=2))


# ── Train ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(cfg.train_csv).exists():
        sys.exit(f"ERROR: train CSV not found at {cfg.train_csv}. "
                 f"Run scripts/prepare_finetune_data.py first.")
    if not Path(cfg.val_csv).exists():
        sys.exit(f"ERROR: val CSV not found at {cfg.val_csv}.")

    hf_token = get_hf_token()
    model, tokenizer = load_base_model(cfg, hf_token)
    model = attach_lora(model, cfg)

    save_run_config(cfg, tokenizer, output_dir)

    print("\n── Building datasets ──")
    train_ds = build_dataset(Path(cfg.train_csv), tokenizer, cfg.max_seq_len)
    val_ds   = build_dataset(Path(cfg.val_csv),   tokenizer, cfg.max_seq_len)
    print(f"  train: {len(train_ds)} examples")
    print(f"  val:   {len(val_ds)} examples (capping eval at {cfg.eval_max_rows})")
    val_eval = val_ds.shuffle(seed=cfg.seed).select(range(min(cfg.eval_max_rows, len(val_ds))))

    # DataCollatorForSeq2Seq pads input_ids and labels (with -100 for labels)
    # to the longest sequence in the batch. It's the right collator for our
    # explicit-label format even though we're doing causal LM training.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir                 = str(output_dir),
        num_train_epochs           = cfg.epochs,
        per_device_train_batch_size= cfg.batch_size,
        per_device_eval_batch_size = cfg.batch_size,
        gradient_accumulation_steps= cfg.grad_accum,
        learning_rate              = cfg.lr,
        lr_scheduler_type          = cfg.lr_scheduler,
        warmup_ratio               = cfg.warmup_ratio,
        weight_decay               = cfg.weight_decay,
        bf16                       = (not cfg.cpu_debug),
        logging_steps              = cfg.logging_steps,
        eval_strategy              = "steps",
        eval_steps                 = cfg.eval_steps,
        save_strategy              = "steps",
        save_steps                 = cfg.save_steps,
        save_total_limit           = cfg.save_total_limit,
        load_best_model_at_end     = True,
        metric_for_best_model      = "eval_loss",
        greater_is_better          = False,
        gradient_checkpointing     = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        optim                      = ("paged_adamw_8bit" if cfg.use_qlora else "adamw_torch"),
        report_to                  = ["tensorboard"],
        logging_dir                = str(output_dir / "runs"),
        seed                       = cfg.seed,
        dataloader_num_workers     = 2,
        remove_unused_columns      = False,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_eval,
        data_collator   = collator,
        tokenizer       = tokenizer,
    )

    print("\n── Training ──")
    trainer.train(resume_from_checkpoint=cfg.resume)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final adapter to {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Final eval on the capped val set (LM loss).
    metrics = trainer.evaluate()
    (output_dir / "final_eval.json").write_text(json.dumps(metrics, indent=2))
    print(f"  Final eval: {metrics}")
    print("\nDone. Next: scripts/evaluate_txgemma_lora.py to score the adapter.")


if __name__ == "__main__":
    main()
