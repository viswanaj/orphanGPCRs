#!/usr/bin/env python3
"""Query the Allen Human Brain Atlas microarray API for orphan GPCR expression.

For each orphan GPCR found in the local CSV, this script:
  1. Looks up microarray probes in the Human Brain Atlas.
  2. Retrieves per-sample expression z-scores across all donors.
  3. Aggregates by top-level brain structure to find regions of enrichment.
  4. Writes a CSV of orphan GPCRs that have uniquely enriched brain expression.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import requests

API_BASE = "http://api.brain-map.org/api/v2"

# IUPHAR Class A orphans with *no* known pharmacology (2024 list)
IUPHAR_ORPHANS: Set[str] = {
    "GPR20", "GPR22", "GPR26", "GPR33", "GPR45", "GPR50", "GPR62",
    "GPR78", "GPR82", "GPR135", "GPR141", "GPR148", "GPR149", "GPR150",
    "GPR152", "GPR153", "GPR161", "GPR176", "MRGPRF", "MRGPRG", "MRGPRX3",
}

# IUPHAR Class A orphans with surrogate ligands only
IUPHAR_SURROGATE_ORPHANS: Set[str] = {
    "GPR21", "GPR27", "GPR52", "GPR85", "GPR88",
}

EXTENDED_ORPHAN_PATTERN = re.compile(r"^GPR\d+$", re.IGNORECASE)

HUMAN_MA_PRODUCT_ID = 2
ALL_DONOR_IDS = [9861, 10021, 12876, 14380, 15496, 15697]


def _api_get(endpoint: str, params: Optional[dict] = None, retries: int = 3) -> dict:
    url = f"{API_BASE}/{endpoint}"
    backoff = 1.0
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 10)
    return {}


def build_orphan_set(csv_path: str, strict: bool = False) -> Dict[str, str]:
    """Return {gene_symbol: uniprot_accession} for orphan GPCRs in the CSV."""
    orphans: Dict[str, str] = {}
    all_orphan_symbols = IUPHAR_ORPHANS | IUPHAR_SURROGATE_ORPHANS

    with open(csv_path, "r") as f:
        lines = [line for line in f if line.strip()]

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    for row in reader:
        gene = row.get("gene", "").strip()
        acc = row.get("accession", "").strip()
        if not gene:
            continue
        gene_upper = gene.upper()

        if gene_upper in all_orphan_symbols:
            orphans[gene] = acc
        elif not strict and EXTENDED_ORPHAN_PATTERN.match(gene):
            orphans[gene] = acc

    return orphans


def lookup_probes(gene_symbol: str) -> List[dict]:
    """Find all microarray probes for a gene in the Human Brain Atlas."""
    criteria = (
        f"model::Probe,"
        f"rma::criteria,"
        f"products[id$eq{HUMAN_MA_PRODUCT_ID}],"
        f"gene[acronym$eq'{gene_symbol}'],"
        f"rma::include,gene"
    )
    data = _api_get("data/query.json", params={"criteria": criteria})
    return data.get("msg", [])


def get_expression_by_structure(probe_ids: List[int],
                                donor_ids: Optional[List[int]] = None
                                ) -> Dict[str, Dict[str, List[float]]]:
    """Query the human microarray expression service and aggregate by top-level structure.

    Returns {top_level_structure_name: {"z_scores": [...], "expr_levels": [...]}}.
    """
    if donor_ids is None:
        donor_ids = ALL_DONOR_IDS

    probe_str = ",".join(str(p) for p in probe_ids)
    struct_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"z_scores": [], "expr_levels": []}
    )

    for donor_id in donor_ids:
        criteria = (
            f"service::human_microarray_expression"
            f"[probes$eq{probe_str}]"
            f"[donors$eq{donor_id}]"
        )
        try:
            data = _api_get("data/query.json", params={"criteria": criteria})
        except Exception:
            continue

        msg = data.get("msg", {})
        if not isinstance(msg, dict):
            continue

        probes_data = msg.get("probes", [])
        samples = msg.get("samples", [])
        if not probes_data or not samples:
            continue

        for probe_rec in probes_data:
            z_scores = probe_rec.get("z-score", [])
            expr_levels = probe_rec.get("expression_level", [])

            for idx, sample in enumerate(samples):
                if idx >= len(z_scores):
                    break
                top_struct = sample.get("top_level_structure", {})
                struct_name = top_struct.get("name", "unknown")

                try:
                    z = float(z_scores[idx])
                    e = float(expr_levels[idx]) if idx < len(expr_levels) else 0.0
                except (ValueError, TypeError):
                    continue

                struct_data[struct_name]["z_scores"].append(z)
                struct_data[struct_name]["expr_levels"].append(e)

        time.sleep(0.3)

    return dict(struct_data)


def summarise_enrichment(struct_data: Dict[str, Dict[str, List[float]]],
                         top_n: int = 5) -> List[dict]:
    """Return top enriched structures sorted by mean z-score."""
    aggregated = []
    for name, vals in struct_data.items():
        zs = vals["z_scores"]
        es = vals["expr_levels"]
        if not zs:
            continue
        aggregated.append({
            "structure_name": name,
            "mean_z_score": round(sum(zs) / len(zs), 4),
            "mean_expression": round(sum(es) / len(es), 4) if es else 0.0,
            "max_z_score": round(max(zs), 4),
            "n_samples": len(zs),
        })

    aggregated.sort(key=lambda x: x["mean_z_score"], reverse=True)
    return aggregated[:top_n]


def sequence_for_gene(fasta_path: str, gene_symbol: str) -> str:
    """Extract the amino acid sequence for a gene from a FASTA file."""
    capturing = False
    seq_parts: List[str] = []

    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if capturing:
                    break
                if f"GN={gene_symbol} " in line or line.rstrip().endswith(f"GN={gene_symbol}"):
                    capturing = True
                continue
            if capturing:
                seq_parts.append(line.strip())

    return "".join(seq_parts)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Query Allen Human Brain Atlas for orphan GPCR expression enrichment."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join("data", "gpcrs_human_reviewed_classes.csv"),
        help="Input CSV with columns: accession, gene, class",
    )
    parser.add_argument(
        "--fasta",
        default=os.path.join("data", "gpcrs_uniprot_human_reviewed.fasta"),
        help="FASTA file of human reviewed GPCR sequences",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("data", "orphan_gpcr_brain_expression.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top enriched structures to report per gene",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Only use the IUPHAR canonical orphan lists (skip GPR\\d+ heuristic)",
    )
    parser.add_argument(
        "--min_z",
        type=float,
        default=1.0,
        help="Minimum mean z-score to consider a gene as having unique expression",
    )
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print("Building orphan GPCR gene set...")
    orphans = build_orphan_set(args.csv, strict=args.strict)
    print(f"  Found {len(orphans)} orphan GPCRs in CSV")

    results: List[dict] = []
    genes_with_data = 0
    genes_enriched = 0

    for i, (gene, acc) in enumerate(sorted(orphans.items()), 1):
        print(f"  [{i}/{len(orphans)}] Querying Allen API for {gene} ({acc})...")
        try:
            probes = lookup_probes(gene)
        except Exception as exc:
            print(f"    ERROR looking up probes: {exc}")
            continue

        if not probes:
            print(f"    No probes found in Human Brain Atlas")
            continue

        probe_ids = [p["id"] for p in probes if isinstance(p, dict)]
        if not probe_ids:
            print(f"    No valid probe IDs")
            continue

        genes_with_data += 1

        try:
            struct_data = get_expression_by_structure(probe_ids)
        except Exception as exc:
            print(f"    ERROR fetching expression: {exc}")
            continue

        top_structures = summarise_enrichment(struct_data, top_n=args.top_n)

        if not top_structures:
            print(f"    Probes found but no expression data returned")
            continue

        best_z = top_structures[0]["mean_z_score"]
        if best_z < args.min_z:
            print(f"    Max mean z-score {best_z:.2f} below threshold {args.min_z}")
            continue

        genes_enriched += 1
        seq = sequence_for_gene(args.fasta, gene)

        struct_names = "; ".join(s["structure_name"] for s in top_structures)
        z_scores_str = "; ".join(f'{s["mean_z_score"]:.2f}' for s in top_structures)
        energies_str = "; ".join(f'{s["mean_expression"]:.2f}' for s in top_structures)
        probe_id_str = ";".join(str(p) for p in probe_ids)

        results.append({
            "gene_symbol": gene,
            "uniprot_accession": acc,
            "probe_ids": probe_id_str,
            "n_probes": len(probe_ids),
            "top_enriched_structures": struct_names,
            "z_scores": z_scores_str,
            "expression_energies": energies_str,
            "best_z_score": best_z,
            "sequence": seq,
        })

        print(f"    Best mean z={best_z:.2f} in {top_structures[0]['structure_name']} "
              f"({len(probe_ids)} probes)")

        time.sleep(0.5)

    results.sort(key=lambda r: r["best_z_score"], reverse=True)

    fieldnames = [
        "gene_symbol", "uniprot_accession", "probe_ids", "n_probes",
        "top_enriched_structures", "z_scores", "expression_energies",
        "best_z_score", "sequence",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. {genes_with_data} orphan GPCRs had probes; "
          f"{genes_enriched} showed enriched expression (z >= {args.min_z}).")
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
