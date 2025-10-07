#!/usr/bin/env python3

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import requests

# EBI Clustal Omega REST API
BASE = "https://www.ebi.ac.uk/Tools/services/rest/clustalo"


def submit_job(fasta_text: str, email: str = "noreply@example.com", title: Optional[str] = None) -> str:
    data = {
        "sequence": fasta_text,
        "email": email,
        "cpu": "4",
        "guidetreeout": "true",
        "mbed": "true",
        "mbediteration": "true",
        "iterations": "1",
        "outfmt": "fa",
        "stype": "protein",
    }
    if title:
        data["title"] = title
    r = requests.post(f"{BASE}/run", data=data, timeout=120)
    r.raise_for_status()
    return r.text.strip()


def poll_status(job_id: str, poll_interval: int = 5, timeout_s: int = 3600) -> None:
    start = time.time()
    while True:
        r = requests.get(f"{BASE}/status/{job_id}", timeout=30)
        r.raise_for_status()
        status = r.text.strip()
        if status in {"FINISHED", "ERROR", "FAILURE"}:
            if status != "FINISHED":
                raise RuntimeError(f"Job {job_id} ended with status {status}")
            return
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Polling timed out for job {job_id}")
        time.sleep(poll_interval)


def fetch_result(job_id: str, result_type: str) -> str:
    r = requests.get(f"{BASE}/result/{job_id}/{result_type}", timeout=120)
    r.raise_for_status()
    return r.text


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Align Class A GPCRs via EBI Clustal Omega and produce alignment and tree.")
    ap.add_argument("--in_fasta", required=True, help="Input FASTA of Class A GPCRs")
    ap.add_argument("--out_dir", default=os.path.join(os.getcwd(), "data", "classA_alignment"), help="Output directory")
    ap.add_argument("--email", default="noreply@example.com", help="Contact email for EBI job submission")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_text = Path(args.in_fasta).read_text()

    print("Submitting job to EBI Clustal Omega...")
    job_id = submit_job(fasta_text, email=args.email, title="classA_gpcr_alignment")
    print(f"Job ID: {job_id}")

    print("Waiting for completion...")
    poll_status(job_id)
    print("Finished.")

    aln = fetch_result(job_id, "aln-fasta")
    # Available result identifiers include: 'tree' (guide tree, Newick) and 'phylotree' (phylogenetic tree)
    newick = fetch_result(job_id, "tree")
    phylotree = fetch_result(job_id, "phylotree")

    (out_dir / "classA_aligned.fasta").write_text(aln)
    (out_dir / "classA_tree.newick").write_text(newick)
    (out_dir / "classA_tree.phylotree").write_text(phylotree)

    print(f"Wrote alignment and trees to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
