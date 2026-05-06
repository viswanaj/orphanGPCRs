#!/usr/bin/env python3
"""
Build a database of human GPCR sequences from SwissProt (UniProt),
cross-referenced with IUPHAR Guide to Pharmacology orphan designations.

IUPHAR source: flat-file CSV downloads from guidetopharmacology.org/DATA/
  - targets_and_families.csv  → GPCR target list + Human SwissProt accession
  - interactions.csv          → endogenous ligand flag per target/species

Orphan logic:
  iuphar_is_orphan = True   if IUPHAR GPCR target has NO human endogenous
                             ligand interactions
  iuphar_is_orphan = False  if it has at least one
  iuphar_is_orphan = NULL   if the UniProt entry is not in IUPHAR

consensus_status prefers IUPHAR; falls back to UniProt annotation text.

Output (relative to repo root):
  gpcr_sequence_db/gpcr_sequences.db   -- SQLite database
  gpcr_sequence_db/gpcr_sequences.csv  -- CSV export (no sequence column)
"""

import csv
import io
import re
import sqlite3
import time
from datetime import date
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR   = REPO_ROOT / "gpcr_sequence_db"
DB_PATH   = OUT_DIR / "gpcr_sequences.db"
CSV_PATH  = OUT_DIR / "gpcr_sequences.csv"

# ── URLs ───────────────────────────────────────────────────────────────────────
UNIPROT_SEARCH        = "https://rest.uniprot.org/uniprotkb/search"
IUPHAR_TARGETS_CSV    = "https://www.guidetopharmacology.org/DATA/targets_and_families.csv"
IUPHAR_INTERACTIONS_CSV = "https://www.guidetopharmacology.org/DATA/interactions.csv"

TODAY = date.today().isoformat()

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


# ─────────────────────────────────────────────────────────────────────────────
# UniProt
# ─────────────────────────────────────────────────────────────────────────────

_UNIPROT_FIELDS = ",".join([
    "accession",
    "id",
    "gene_names",
    "protein_name",
    "sequence",
    "cc_function",
    "ft_binding",
    "xref_guidetopharmacology",
])


def fetch_uniprot_gpcrs() -> list[dict]:
    """Retrieve all human SwissProt GPCR entries via paginated REST API."""
    params = {
        "query":  "reviewed:true AND keyword:KW-0297 AND organism_id:9606",
        "fields": _UNIPROT_FIELDS,
        "format": "json",
        "size":   500,
    }
    entries: list[dict] = []
    url: str | None = UNIPROT_SEARCH
    page = 0
    while url:
        resp = SESSION.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("results", [])
        entries.extend(batch)
        page += 1
        print(f"  UniProt page {page}: +{len(batch)} entries (total {len(entries)})")
        url = _next_link(resp.headers.get("Link", ""))
        params = None   # only sent on the first request
        time.sleep(0.3)
    return entries


def _next_link(link_header: str) -> str | None:
    # Can't split on "," — the URL contains commas in the fields parameter
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else None


def _uniprot_orphan_status(entry: dict) -> tuple[str, str]:
    """
    Infer orphan status from function annotation text.
    Returns (status, ligand_info) where status is 'orphan'|'cognate'|'unknown'.
    """
    func_text = ""
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            for t in comment.get("texts", []):
                func_text += " " + t.get("value", "")
    func_lower = func_text.lower()

    orphan_phrases = (
        "orphan receptor",
        "endogenous ligand is not known",
        "endogenous ligand unknown",
        "whose endogenous ligand",
        "no endogenous ligand",
        "ligand has not been identified",
    )
    is_orphan    = any(p in func_lower for p in orphan_phrases)
    deorphanized = "deorphanized" in func_lower

    # Pull sentences that mention "ligand" for context
    ligand_info = ""
    if "ligand" in func_lower:
        sentences   = re.split(r"(?<=[.!?])\s+", func_text.strip())
        ligand_info = " ".join(s for s in sentences if "ligand" in s.lower())[:800]

    if deorphanized:
        return "cognate", ligand_info
    if is_orphan:
        return "orphan", ligand_info

    # Presence of a binding-site feature is a weak signal for cognate
    has_binding = any(f.get("type") == "Binding site" for f in entry.get("features", []))
    if has_binding:
        return "cognate", ligand_info

    return "unknown", ligand_info


def parse_uniprot_entry(entry: dict) -> dict:
    accession  = entry.get("primaryAccession", "")
    entry_name = entry.get("uniProtkbId", "")

    genes     = entry.get("genes", [])
    gene_name = genes[0].get("geneName", {}).get("value", "") if genes else ""

    rec_name  = entry.get("proteinDescription", {}).get("recommendedName", {})
    full_name = rec_name.get("fullName", {}).get("value", "") if rec_name else ""

    seq_obj  = entry.get("sequence", {})
    sequence = seq_obj.get("value", "")
    seq_len  = seq_obj.get("length", len(sequence))

    # IUPHAR cross-reference stored by UniProt
    iuphar_xref_id: int | None = None
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") in ("GuidetoPHARMACOLOGY", "GuideToPHARMACOLOGY"):
            try:
                iuphar_xref_id = int(xref["id"])
            except (KeyError, ValueError):
                pass
            break

    orphan_status, ligand_info = _uniprot_orphan_status(entry)

    return {
        "uniprot_accession":     accession,
        "uniprot_id":            entry_name,
        "gene_name":             gene_name,
        "protein_name":          full_name,
        "sequence":              sequence,
        "sequence_length":       seq_len,
        "uniprot_orphan_status": orphan_status,
        "uniprot_ligand_info":   ligand_info,
        "_iuphar_xref_id":       iuphar_xref_id,   # temporary join key
    }


# ─────────────────────────────────────────────────────────────────────────────
# IUPHAR  (CSV flat-file downloads — no per-target API calls)
# ─────────────────────────────────────────────────────────────────────────────

def _iuphar_csv(url: str) -> list[dict]:
    """Download an IUPHAR CSV (has a version comment on line 0) and parse it."""
    resp = SESSION.get(url, timeout=120)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    # Line 0: "# GtoPdb Version: ..."   Line 1+: real CSV with header
    reader = csv.DictReader(lines[1:])
    return list(reader)


def load_iuphar_gpcr_targets() -> dict[int, dict]:
    """
    Parse targets_and_families.csv.
    Returns {iuphar_target_id: {iuphar_name, human_uniprot}}.
    """
    rows = _iuphar_csv(IUPHAR_TARGETS_CSV)
    targets: dict[int, dict] = {}
    for r in rows:
        if r.get("Type", "").lower() != "gpcr":
            continue
        try:
            tid = int(r["Target id"])
        except (KeyError, ValueError):
            continue
        # Strip HTML tags from name (e.g. "5-HT<sub>1A</sub> receptor")
        name = re.sub(r"<[^>]+>", "", r.get("Target name", ""))
        targets[tid] = {
            "iuphar_name":   name,
            "human_uniprot": r.get("Human SwissProt", "").strip(),
        }
    print(f"  IUPHAR targets_and_families.csv: {len(targets)} GPCR targets")
    return targets


def load_iuphar_orphan_status(gpcr_target_ids: set[int]) -> dict[int, bool]:
    """
    Parse interactions.csv to determine orphan status for each GPCR target.
    A target is COGNATE if it has ≥1 human endogenous ligand interaction.
    All other GPCR targets are ORPHAN.
    Returns {iuphar_target_id: is_orphan (bool)}.
    """
    rows = _iuphar_csv(IUPHAR_INTERACTIONS_CSV)

    cognate_ids: set[int] = set()
    for r in rows:
        try:
            tid = int(r.get("Target ID", ""))
        except ValueError:
            continue
        if tid not in gpcr_target_ids:
            continue
        if (r.get("Target Species", "").lower() == "human"
                and r.get("Endogenous", "").lower() == "true"):
            cognate_ids.add(tid)

    result = {tid: (tid not in cognate_ids) for tid in gpcr_target_ids}
    orphan_count  = sum(1 for v in result.values() if v)
    cognate_count = sum(1 for v in result.values() if not v)
    print(f"  IUPHAR interactions.csv: {cognate_count} cognate, {orphan_count} orphan")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Merge & store
# ─────────────────────────────────────────────────────────────────────────────

def _consensus(uniprot_status: str, iuphar_is_orphan) -> str:
    """IUPHAR is authoritative when available; fall back to UniProt text."""
    if iuphar_is_orphan is True:
        return "orphan"
    if iuphar_is_orphan is False:
        return "cognate"
    if uniprot_status in ("orphan", "cognate"):
        return uniprot_status
    return "unknown"


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS gpcrs (
    uniprot_accession     TEXT PRIMARY KEY,
    uniprot_id            TEXT,
    gene_name             TEXT,
    protein_name          TEXT,
    organism              TEXT DEFAULT 'Homo sapiens',
    sequence              TEXT,
    sequence_length       INTEGER,
    uniprot_orphan_status TEXT,          -- 'orphan' | 'cognate' | 'unknown'
    uniprot_ligand_info   TEXT,          -- sentences from function annotation
    iuphar_target_id      INTEGER,       -- NULL if not in IUPHAR
    iuphar_name           TEXT,
    iuphar_is_orphan      INTEGER,       -- 1=orphan / 0=cognate / NULL=not in IUPHAR
    consensus_status      TEXT,          -- 'orphan' | 'cognate' | 'unknown'
    date_retrieved        TEXT
);
"""

_INSERT = """
INSERT OR REPLACE INTO gpcrs VALUES (
    :uniprot_accession, :uniprot_id, :gene_name, :protein_name,
    'Homo sapiens', :sequence, :sequence_length,
    :uniprot_orphan_status, :uniprot_ligand_info,
    :iuphar_target_id, :iuphar_name, :iuphar_is_orphan,
    :consensus_status, :date_retrieved
)
"""

_CSV_FIELDS = [
    "uniprot_accession", "uniprot_id", "gene_name", "protein_name",
    "organism", "sequence_length",
    "uniprot_orphan_status", "uniprot_ligand_info",
    "iuphar_target_id", "iuphar_name", "iuphar_is_orphan",
    "consensus_status", "date_retrieved",
]


def write_outputs(rows: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(_CREATE_TABLE)
    conn.executemany(_INSERT, rows)
    conn.commit()
    conn.close()
    print(f"SQLite written : {DB_PATH}  ({len(rows)} rows)")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written    : {CSV_PATH}  (no sequence column)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. UniProt
    print("── Step 1: Fetching human GPCRs from SwissProt ──")
    raw_entries = fetch_uniprot_gpcrs()
    parsed      = [parse_uniprot_entry(e) for e in raw_entries]
    print(f"  Parsed {len(parsed)} UniProt entries")

    # 2. IUPHAR target list (CSV download)
    print("\n── Step 2: Loading IUPHAR GPCR targets ──")
    iuphar_targets = load_iuphar_gpcr_targets()  # {tid: {iuphar_name, human_uniprot}}

    # 3. IUPHAR orphan status via interactions CSV
    print("\n── Step 3: Deriving orphan status from IUPHAR interactions ──")
    iuphar_orphan_map = load_iuphar_orphan_status(set(iuphar_targets.keys()))
    # {tid: is_orphan bool}

    # Build reverse lookup: human_uniprot → iuphar_tid
    uniprot_to_iuphar: dict[str, int] = {
        info["human_uniprot"]: tid
        for tid, info in iuphar_targets.items()
        if info["human_uniprot"]
    }

    # 4. Merge
    print("\n── Step 4: Merging ──")
    rows: list[dict] = []
    for p in parsed:
        acc = p["uniprot_accession"]

        # Prefer xref stored in the UniProt entry; fall back to reverse lookup
        iuphar_id = p.pop("_iuphar_xref_id") or uniprot_to_iuphar.get(acc)

        info             = iuphar_targets.get(iuphar_id, {})
        iuphar_is_orphan = iuphar_orphan_map.get(iuphar_id)   # True/False/None

        row = {
            **p,
            "iuphar_target_id": iuphar_id,
            "iuphar_name":      info.get("iuphar_name"),
            "iuphar_is_orphan": int(iuphar_is_orphan) if iuphar_is_orphan is not None else None,
            "date_retrieved":   TODAY,
        }
        row["consensus_status"] = _consensus(row["uniprot_orphan_status"], iuphar_is_orphan)
        rows.append(row)

    # 5. Write
    print("\n── Step 5: Writing outputs ──")
    write_outputs(rows)

    # 6. Summary
    orphans   = sum(1 for r in rows if r["consensus_status"] == "orphan")
    cognates  = sum(1 for r in rows if r["consensus_status"] == "cognate")
    unknown   = sum(1 for r in rows if r["consensus_status"] == "unknown")
    in_iuphar = sum(1 for r in rows if r["iuphar_target_id"] is not None)

    print(f"""
── Summary ──────────────────────────────
  Total GPCRs stored : {len(rows)}
  In IUPHAR          : {in_iuphar}
  Consensus orphan   : {orphans}
  Consensus cognate  : {cognates}
  Unknown            : {unknown}
─────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
