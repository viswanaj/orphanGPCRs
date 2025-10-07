#!/usr/bin/env python3

import os
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

FASTA_HEADER_RE = re.compile(r"^>.* GN=([^\s]+)")

# Heuristic class mapping based on human GPCR gene symbols
# Strategy:
# - Explicit prefixes/symbols for distinct classes (F, B2/Adhesion, C, B1/Secretin)
# - Default remaining GPCRs to Class A (Rhodopsin-like)

CLASS_F_PREFIXES = ("FZD",)  # Frizzled receptors
CLASS_F_SYMBOLS = {"SMO"}    # Smoothened

CLASS_B2_PREFIXES = ("ADGR",)  # Adhesion GPCRs (ADGRx naming)

CLASS_C_PREFIXES = (
    "GRM",    # Metabotropic glutamate receptors
    "GABBR",  # GABA-B receptor subunits
    "TAS1R",  # Taste receptor type 1
    "CASR",   # Calcium-sensing receptor
    "GPRC6A", # Class C orphan
)

# Known Class B1 (Secretin-like) receptors (symbols and common prefixes)
CLASS_B1_SYMBOLS = {
    # Secretin-like hormone receptors
    "GLP1R", "GLP2R", "GIPR", "GCGR", "SCTR", "GHRHR", "CRHR1", "CRHR2",
    "PTH1R", "PTH2R", "CALCR", "CALCRL", "VIPR1", "VIPR2", "ADCYAP1R1",
    "PAC1",  # Alias; UniProt often uses ADCYAP1R1
    "CGRPR", # Sometimes used for CALCRL-RAMP complexes, included defensively
}
CLASS_B1_PREFIXES = (
    # Some use family-like prefixes
    "VIPR", "PTH", "CRHR", "GLP", "GIPR", "GCGR", "SCTR", "GHRHR", "CALCR",
)

CLASS_LABELS = {
    "A": "Class A (Rhodopsin-like)",
    "B1": "Class B1 (Secretin-like)",
    "B2": "Class B2 (Adhesion)",
    "C": "Class C (Glutamate)",
    "F": "Class F (Frizzled/Smoothened)",
}


def parse_fasta(path: str) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    header: str = ""
    seq_chunks: List[str] = []

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    entries.append((header, "".join(seq_chunks)))
                header = line
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header:
            entries.append((header, "".join(seq_chunks)))
    return entries


def extract_gene_symbol(header: str) -> str:
    m = FASTA_HEADER_RE.match(header)
    if m:
        return m.group(1)
    # Fallback: attempt to find gene symbol in the UniProt header tokens
    # e.g., ">sp|ACC|ENTRY_NAME ... GN=XYZ ..."
    for token in header.split():
        if token.startswith("GN="):
            return token[3:]
    return ""


def classify_gene_symbol(gene: str) -> str:
    if not gene:
        return "A"
    gene_upper = gene.upper()

    # Class F
    if gene_upper in CLASS_F_SYMBOLS:
        return "F"
    for p in CLASS_F_PREFIXES:
        if gene_upper.startswith(p):
            return "F"

    # Class B2 (Adhesion)
    for p in CLASS_B2_PREFIXES:
        if gene_upper.startswith(p):
            return "B2"

    # Class C
    for p in CLASS_C_PREFIXES:
        if gene_upper.startswith(p):
            return "C"

    # Class B1 (Secretin-like)
    if gene_upper in CLASS_B1_SYMBOLS:
        return "B1"
    for p in CLASS_B1_PREFIXES:
        if gene_upper.startswith(p):
            return "B1"

    # Default to Class A (Rhodopsin-like)
    return "A"


def write_fasta(entries: List[Tuple[str, str]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out:
        for header, seq in entries:
            out.write(f"{header}\n")
            # Wrap to 60 chars per line
            for i in range(0, len(seq), 60):
                out.write(seq[i:i+60] + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Group human reviewed GPCR FASTA by GPCR class using gene-symbol heuristics.")
    parser.add_argument(
        "--in",
        dest="input_fasta",
        required=True,
        help="Input FASTA file (human, reviewed GPCRs)",
    )
    parser.add_argument(
        "--out_dir",
        dest="out_dir",
        default=os.path.join(os.getcwd(), "data", "classes"),
        help="Output directory for per-class FASTA files",
    )
    args = parser.parse_args()

    entries = parse_fasta(args.input_fasta)
    class_to_entries: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for header, seq in entries:
        gene = extract_gene_symbol(header)
        cls = classify_gene_symbol(gene)
        class_to_entries[cls].append((header, seq))

    # Write per-class FASTAs
    for cls, label in CLASS_LABELS.items():
        out_fasta = os.path.join(args.out_dir, f"gpcrs_human_reviewed_{cls}.fasta")
        write_fasta(class_to_entries.get(cls, []), out_fasta)

    # Summary
    total = sum(len(v) for v in class_to_entries.values())
    print(f"Total entries: {total}")
    for cls, label in CLASS_LABELS.items():
        count = len(class_to_entries.get(cls, []))
        print(f"{label}: {count}")

    # Unknown bucket (should be empty since we default to A)
    unknown = len(class_to_entries.get("?", []))
    if unknown:
        print(f"Unclassified: {unknown}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
