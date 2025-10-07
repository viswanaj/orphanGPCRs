#!/usr/bin/env python3

import os
import re
import csv
import argparse
from typing import List, Tuple

FASTA_HEADER_RE = re.compile(r"^>sp\|([^|]+)\|([^\s]+).* GN=([^\s]+)")
GENE_FALLBACK_RE = re.compile(r" GN=([^\s]+)")

CLASS_F_PREFIXES = ("FZD",)
CLASS_F_SYMBOLS = {"SMO"}
CLASS_B2_PREFIXES = ("ADGR",)
CLASS_C_PREFIXES = ("GRM", "GABBR", "TAS1R", "CASR", "GPRC6A")
CLASS_B1_SYMBOLS = {
    "GLP1R", "GLP2R", "GIPR", "GCGR", "SCTR", "GHRHR", "CRHR1", "CRHR2",
    "PTH1R", "PTH2R", "CALCR", "CALCRL", "VIPR1", "VIPR2", "ADCYAP1R1", "PAC1",
}
CLASS_B1_PREFIXES = ("VIPR", "PTH", "CRHR", "GLP", "GIPR", "GCGR", "SCTR", "GHRHR", "CALCR")


def classify_gene_symbol(gene: str) -> str:
    if not gene:
        return "A"
    g = gene.upper()
    if g in CLASS_F_SYMBOLS or any(g.startswith(p) for p in CLASS_F_PREFIXES):
        return "F"
    if any(g.startswith(p) for p in CLASS_B2_PREFIXES):
        return "B2"
    if any(g.startswith(p) for p in CLASS_C_PREFIXES):
        return "C"
    if g in CLASS_B1_SYMBOLS or any(g.startswith(p) for p in CLASS_B1_PREFIXES):
        return "B1"
    return "A"


def parse_fasta_headers(path: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            acc = entry = gene = ""
            m = FASTA_HEADER_RE.match(line)
            if m:
                acc, entry, gene = m.group(1), m.group(2), m.group(3)
            else:
                # Fallback GN parse
                m2 = GENE_FALLBACK_RE.search(line)
                if m2:
                    gene = m2.group(1)
                # Try to get accession from the second token
                toks = line.split("|")
                if len(toks) >= 3:
                    acc = toks[1]
            items.append((acc, gene))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="Export CSV of UniProt accession, gene, and GPCR class from human reviewed FASTA")
    ap.add_argument("--in", dest="input_fasta", required=True)
    ap.add_argument("--out", dest="output_csv", default=os.path.join(os.getcwd(), "data", "gpcrs_human_reviewed_classes.csv"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    rows: List[Tuple[str, str, str]] = []
    for acc, gene in parse_fasta_headers(args.input_fasta):
        cls = classify_gene_symbol(gene)
        rows.append((acc, gene, cls))

    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["accession", "gene", "class"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
