#!/usr/bin/env python3
"""Use TxGemma-Chat to reason about novel scaffolds for orphan GPCRs.

Reads the orphan GPCR brain expression CSV produced by
query_allen_brain_expression.py, feeds each receptor's amino acid
sequence and brain expression context to TxGemma-Chat, and collects
scaffold reasoning and proposed modifications.

Prerequisites:
  - Hugging Face account with TxGemma access accepted
  - HF_TOKEN environment variable set (or pass --hf_token)
  - GPU with sufficient VRAM (9B-Chat ~18 GB fp16, ~6 GB 4-bit)
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

SYSTEM_PROMPT = (
    "You are an expert structural biologist and medicinal chemist specializing "
    "in GPCR pharmacology. You have deep knowledge of transmembrane receptor "
    "architecture, ligand-binding pockets, and scaffold-based drug design."
)


def load_expression_csv(csv_path: str) -> List[dict]:
    """Load orphan GPCR expression data from the Allen Brain Atlas CSV."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sequence", "").strip():
                rows.append(row)
    return rows


def build_analysis_prompt(gene: str, sequence: str, brain_regions: str,
                          z_scores: str) -> str:
    """Build the first turn: ask TxGemma to analyse the receptor."""
    return (
        f"I am studying the orphan GPCR **{gene}**, which has no known "
        f"endogenous ligand. Its amino acid sequence is:\n\n"
        f"```\n{sequence}\n```\n\n"
        f"This receptor shows uniquely enriched expression in the following "
        f"human brain regions (from Allen Human Brain Atlas microarray data):\n"
        f"  Regions: {brain_regions}\n"
        f"  Z-scores: {z_scores}\n\n"
        f"Please analyse this receptor's likely structural features:\n"
        f"1. Predict the approximate transmembrane domain boundaries.\n"
        f"2. Identify conserved GPCR motifs (DRY, NPxxY, CWxP, etc.).\n"
        f"3. Describe the probable ligand-binding pocket location and character "
        f"(hydrophobic, charged, etc.).\n"
        f"4. Note any unusual sequence features that distinguish this orphan "
        f"from well-characterised GPCRs."
    )


def build_scaffold_prompt(gene: str) -> str:
    """Build the second turn: ask TxGemma to propose novel scaffolds."""
    return (
        f"Based on your structural analysis of {gene}, please propose novel "
        f"molecular scaffolds that could serve as starting points for ligand "
        f"discovery:\n\n"
        f"1. Suggest 2-3 distinct chemical scaffold classes (described as "
        f"SMILES or common pharmacophore descriptions) that are likely to "
        f"interact with the predicted binding pocket.\n"
        f"2. For each scaffold, explain the rationale — which residues or "
        f"structural features it targets and what type of interaction "
        f"(agonist, antagonist, allosteric modulator) it might achieve.\n"
        f"3. Suggest any modifications to the receptor sequence itself "
        f"(point mutations or chimeric constructs) that could aid in "
        f"deorphanisation screening assays.\n"
        f"4. Prioritise the scaffolds by predicted tractability."
    )


def _best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, hf_token: Optional[str], quantize_4bit: bool):
    """Load TxGemma-Chat model and tokenizer from Hugging Face."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = _best_device()
    print(f"Using device: {device}")

    if quantize_4bit and device != "cuda":
        print("WARNING: 4-bit quantisation requires CUDA; falling back to bfloat16.")
        quantize_4bit = False

    kwargs: dict = {"token": hf_token, "dtype": torch.bfloat16}

    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        # device_map={"": device} loads all layers onto one device cleanly
        kwargs["device_map"] = {"": device}

    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    print(f"Loading model from {model_id} (4-bit={quantize_4bit})...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    return model, tokenizer


def chat_generate(model, tokenizer, messages: List[dict],
                  max_new_tokens: int = 1024) -> str:
    """Run multi-turn chat generation and return the assistant response."""
    import torch

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def process_gene(model, tokenizer, gene_data: dict,
                 max_new_tokens: int) -> dict:
    """Run the two-turn conversation for a single orphan GPCR."""
    gene = gene_data["gene_symbol"]
    sequence = gene_data["sequence"]
    brain_regions = gene_data.get("top_enriched_structures", "")
    z_scores = gene_data.get("z_scores", "")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Turn 1: structural analysis
    analysis_prompt = build_analysis_prompt(gene, sequence, brain_regions, z_scores)
    messages.append({"role": "user", "content": analysis_prompt})

    print(f"  Turn 1: structural analysis...")
    analysis_response = chat_generate(model, tokenizer, messages, max_new_tokens)
    messages.append({"role": "assistant", "content": analysis_response})

    # Turn 2: scaffold proposals
    scaffold_prompt = build_scaffold_prompt(gene)
    messages.append({"role": "user", "content": scaffold_prompt})

    print(f"  Turn 2: scaffold proposals...")
    scaffold_response = chat_generate(model, tokenizer, messages, max_new_tokens)

    return {
        "gene_symbol": gene,
        "uniprot_accession": gene_data.get("uniprot_accession", ""),
        "brain_regions": brain_regions,
        "z_scores": z_scores,
        "sequence_length": len(sequence),
        "structural_analysis": analysis_response,
        "scaffold_proposals": scaffold_response,
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate novel scaffold proposals for orphan GPCRs using TxGemma-Chat."
    )
    parser.add_argument(
        "--input_csv",
        default=os.path.join("data", "orphan_gpcr_brain_expression.csv"),
        help="CSV from query_allen_brain_expression.py",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("data", "txgemma_scaffold_proposals.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--model",
        default="google/txgemma-9b-chat",
        help="Hugging Face model ID for TxGemma-Chat",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--quantize_4bit",
        action="store_true",
        help="Load model in 4-bit quantisation (requires bitsandbytes, reduces VRAM to ~6 GB)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens per generation turn",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N genes (for testing)",
    )
    args = parser.parse_args(argv)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: No HF_TOKEN found. Model download may fail if gated.", file=sys.stderr)

    print("Loading expression data...")
    gene_rows = load_expression_csv(args.input_csv)
    if not gene_rows:
        print("No orphan GPCRs with sequences found in input CSV.", file=sys.stderr)
        return 1
    print(f"  {len(gene_rows)} orphan GPCRs with expression + sequence data")

    if args.limit:
        gene_rows = gene_rows[:args.limit]
        print(f"  Limited to first {args.limit}")

    model, tokenizer = load_model(args.model, hf_token, args.quantize_4bit)

    results = []
    for i, gene_data in enumerate(gene_rows, 1):
        gene = gene_data["gene_symbol"]
        print(f"\n[{i}/{len(gene_rows)}] Processing {gene}...")
        t0 = time.time()
        try:
            result = process_gene(model, tokenizer, gene_data, args.max_new_tokens)
            results.append(result)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} scaffold proposals to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
