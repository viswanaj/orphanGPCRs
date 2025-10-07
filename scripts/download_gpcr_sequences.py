#!/usr/bin/env python3

import os
import sys
import time
import argparse
from typing import Optional

import requests


UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
# UniProt keyword for G-protein coupled receptor: KW-0297
GPCR_QUERY = "keyword:KW-0297"


def stream_uniprot_fasta(query: str, output_fasta_path: str, timeout_s: int = 1800, max_retries: int = 5) -> None:
    """Stream FASTA entries from UniProt for the given query and write to file.

    Args:
        query: UniProt query string.
        output_fasta_path: File path to write the FASTA output.
        timeout_s: Request timeout in seconds (covers long streaming time).
        max_retries: Number of retries on transient failures.
    """
    params = {
        "format": "fasta",
        "query": query,
        "compressed": "false",
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_fasta_path), exist_ok=True)

    # Stream and write to file with basic retry logic
    backoff_s = 2
    attempt = 0

    while True:
        attempt += 1
        try:
            with requests.get(UNIPROT_STREAM_URL, params=params, stream=True, timeout=timeout_s) as response:
                response.raise_for_status()
                with open(output_fasta_path, "wb") as fasta_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fasta_file.write(chunk)
            break
        except (requests.RequestException, requests.Timeout) as exc:
            if attempt >= max_retries:
                raise
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, 60)



def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download all GPCR protein sequences from UniProt (keyword KW-0297) as FASTA.")
    parser.add_argument(
        "--out",
        dest="out_fasta",
        default=os.path.join(os.getcwd(), "data", "gpcrs_uniprot.fasta"),
        help="Output FASTA file path (default: ./data/gpcrs_uniprot.fasta)",
    )
    parser.add_argument(
        "--reviewed_only",
        action="store_true",
        help="If set, restrict to reviewed (Swiss-Prot) entries only.",
    )
    parser.add_argument(
        "--taxon",
        type=str,
        default=None,
        help="Optional NCBI taxon identifier to filter (e.g., 9606 for human).",
    )

    args = parser.parse_args(argv)

    query = GPCR_QUERY
    if args.reviewed_only:
        query = f"({query}) AND (reviewed:true)"
    if args.taxon:
        query = f"({query}) AND (organism_id:{args.taxon})"

    print(f"Query: {query}")
    print(f"Downloading FASTA to: {args.out_fasta}")
    stream_uniprot_fasta(query=query, output_fasta_path=args.out_fasta)
    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
