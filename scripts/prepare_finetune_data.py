#!/usr/bin/env python3
"""
prepare_finetune_data.py — Build the txGemma fine-tuning dataset from ChEMBL.

For every human GPCR in gpcr_sequence_db/gpcr_sequences.db (except the four
held-out validation receptors), pull binding-assay activities from ChEMBL,
dedupe by (target, canonical SMILES), format TDC BindingDB_ki prompts, and
write three splits (random / scaffold / target) to
gpcr_sequence_db/txgemma-finetune/data/.

Held-out validation receptors (NEVER enter training):
  CB1R   P21554
  HT2AR  P28223
  DRD2   P14416
  GLP1R  P43220

Pipeline
--------
  1. Load UniProt → sequence map from local DB; drop held-out UniProts.
  2. Resolve UniProt → ChEMBL target ID (cached).
  3. Per target, page through /activity for human binding assays with pChEMBL
     (cached).
  4. Filter on QC (no censored measurements, pChEMBL in [3, 12]) and dedupe
     by (UniProt, SMILES) keeping median pChEMBL.
  5. Format the TDC BindingDB_ki prompt and the 0–1000 normalized score.
  6. Three splits, 80/10/10:
       random   — uniform over rows
       scaffold — Bemis–Murcko scaffolds disjoint across splits (cold-ligand)
       target   — UniProts disjoint across splits (cold-target)
  7. Write CSVs + dataset_stats.txt + split_metadata.json.

Usage
-----
  python scripts/prepare_finetune_data.py
"""

import csv
import json
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("ERROR: rdkit is required. Install with `pip install rdkit`.")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH   = REPO_ROOT / "gpcr_sequence_db" / "gpcr_sequences.db"
DATA_DIR  = REPO_ROOT / "gpcr_sequence_db" / "txgemma-finetune" / "data"
CACHE_DIR = DATA_DIR / ".cache"
ACT_CACHE_DIR    = CACHE_DIR / "activities"
UNIPROT_MAP_JSON = CACHE_DIR / "uniprot_to_chembl.json"
ALL_PAIRS_CSV    = DATA_DIR / "all_pairs.csv"
STATS_TXT        = DATA_DIR / "dataset_stats.txt"
SPLIT_META_JSON  = DATA_DIR / "split_metadata.json"

# ── Held-out validation receptors ────────────────────────────────────────────
HELD_OUT_UNIPROTS = {
    "P21554",  # CB1R
    "P28223",  # HT2AR
    "P14416",  # DRD2
    "P43220",  # GLP1R
}

# ── ChEMBL ────────────────────────────────────────────────────────────────────
CHEMBL_BASE      = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE        = 1000
REQUEST_TIMEOUT  = 60
RETRY_COUNT      = 3
RATE_LIMIT_SLEEP = 0.15

# ── Dataset config ────────────────────────────────────────────────────────────
MAX_SEQ_LEN              = 512    # matches txgemma_ligand_prediction.py
MIN_COMPOUNDS_PER_TARGET = 5
PCHEMBL_MIN              = 3.0
PCHEMBL_MAX              = 12.0
SCORE_PCHEMBL_FLOOR      = 4.0    # pChEMBL=4  -> score=1000 (weakest)
SCORE_PCHEMBL_CEIL       = 12.0   # pChEMBL=12 -> score=0    (strongest)

SPLIT_FRACS = (0.80, 0.10, 0.10)  # train, val, test
SPLIT_SEED  = 42

# ── TDC BindingDB_ki prompt (verbatim from txgemma_ligand_prediction.py) ─────
_TDC_BINDINGDB_KI_PROMPT = (
    "Instructions: Answer the following question about drug target interactions.\n"
    "Context: Drug-target binding is the physical interaction between a drug "
    "and a specific biological molecule, such as a protein or enzyme. This "
    "interaction is essential for the drug to exert its pharmacological effect. "
    "The strength of the drug-target binding is determined by the binding "
    "affinity, which is a measure of how tightly the drug binds to the target. "
    "Ki is the equilibrium dissociation constant of an inhibitor. It is the "
    "concentration of inhibitor at which half of the target binding sites are "
    "occupied. A lower Ki value indicates a stronger binding affinity.\n"
    "Question: Given the target amino acid sequence and compound SMILES "
    "string, predict their normalized binding affinity Kd from 000 to 1000, "
    "where 000 is minimum Ki and 1000 is maximum Ki.\n"
    "Drug SMILES: {Drug SMILES}\n"
    "Target amino acid sequence: {Target amino acid sequence}\n"
    "Answer:"
)


def format_prompt(smiles: str, sequence: str) -> str:
    seq = sequence[:MAX_SEQ_LEN]
    return (
        _TDC_BINDINGDB_KI_PROMPT
        .replace("{Drug SMILES}", smiles)
        .replace("{Target amino acid sequence}", seq)
    )


def pchembl_to_score(pchembl: float) -> int:
    """pChEMBL → TDC 0–1000 normalized score (low = strong binder)."""
    s = 1000.0 * (1.0 - (pchembl - SCORE_PCHEMBL_FLOOR) / (SCORE_PCHEMBL_CEIL - SCORE_PCHEMBL_FLOOR))
    return int(round(max(0.0, min(1000.0, s))))


def score_to_str(score: int) -> str:
    """Match the TDC training format: '000' .. '999', or '1000'."""
    return "1000" if score >= 1000 else f"{score:03d}"


# ── ChEMBL HTTP ───────────────────────────────────────────────────────────────

def _chembl_get(url: str, params: dict) -> dict:
    last_exc: Exception | None = None
    for attempt in range(RETRY_COUNT):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            time.sleep(RATE_LIMIT_SLEEP)
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(2 ** attempt)
    assert last_exc is not None
    raise last_exc


# ── Phase 1: load local GPCR DB ───────────────────────────────────────────────

def load_gpcrs_from_db() -> dict[str, str]:
    if not DB_PATH.exists():
        sys.exit(f"ERROR: {DB_PATH} not found. Run scripts/build_gpcr_db.py first.")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT uniprot_accession, sequence FROM gpcrs"
    ).fetchall()
    conn.close()
    return {u: s for u, s in rows if u and s}


# ── Phase 2: resolve UniProt → ChEMBL target ID ──────────────────────────────

def resolve_uniprot_to_chembl(uniprots: list[str]) -> dict[str, str]:
    if UNIPROT_MAP_JSON.exists():
        cached = json.loads(UNIPROT_MAP_JSON.read_text())
        missing = [u for u in uniprots if u not in cached]
        if not missing:
            return cached
        print(f"  Resuming: {len(cached)} cached, resolving {len(missing)} more")
        mapping = dict(cached)
    else:
        mapping = {}
        missing = list(uniprots)

    for uniprot in tqdm(missing, desc="Resolving UniProt→ChEMBL", unit="target"):
        try:
            data = _chembl_get(
                f"{CHEMBL_BASE}/target",
                {
                    "target_components__accession": uniprot,
                    "target_type": "SINGLE PROTEIN",
                    "organism": "Homo sapiens",
                    "format": "json",
                    "limit": 5,
                },
            )
        except Exception as e:
            print(f"  WARN: {uniprot} resolve failed: {e}")
            continue

        targets = data.get("targets", [])
        chosen = next(
            (t for t in targets if t.get("target_type") == "SINGLE PROTEIN"),
            targets[0] if targets else None,
        )
        if chosen and chosen.get("target_chembl_id"):
            mapping[uniprot] = chosen["target_chembl_id"]

    UNIPROT_MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    UNIPROT_MAP_JSON.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    return mapping


# ── Phase 3: fetch activities per target (cached) ────────────────────────────

def fetch_activities_for_target(chembl_id: str) -> list[dict]:
    cache_path = ACT_CACHE_DIR / f"{chembl_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    activities: list[dict] = []
    offset = 0
    while True:
        try:
            data = _chembl_get(
                f"{CHEMBL_BASE}/activity",
                {
                    "target_chembl_id":      chembl_id,
                    "pchembl_value__isnull": "false",
                    "assay_type":            "B",
                    "target_organism":       "Homo sapiens",
                    "limit":                 PAGE_SIZE,
                    "offset":                offset,
                    "format":                "json",
                },
            )
        except Exception as e:
            print(f"  WARN: {chembl_id} fetch failed at offset {offset}: {e}")
            return []  # do NOT cache partial — leave for retry on next run
        batch = data.get("activities", [])
        activities.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    ACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(activities))
    return activities


# ── Phase 4: build deduped dataset ───────────────────────────────────────────

def _qc_ok(a: dict) -> bool:
    if not a.get("canonical_smiles") or not a.get("pchembl_value"):
        return False
    try:
        v = float(a["pchembl_value"])
    except (TypeError, ValueError):
        return False
    if not (PCHEMBL_MIN <= v <= PCHEMBL_MAX):
        return False
    # Drop censored measurements (<, >).
    if a.get("standard_relation") not in (None, "=", "~"):
        return False
    return True


def build_dataset_rows(
    uniprot_to_chembl: dict[str, str],
    uniprot_to_seq:   dict[str, str],
) -> list[dict]:
    bucket_vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    bucket_meta: dict[tuple[str, str], dict] = {}

    for uniprot, chembl_id in tqdm(
        uniprot_to_chembl.items(), desc="Fetching activities", unit="target"
    ):
        if uniprot in HELD_OUT_UNIPROTS:
            continue
        seq = uniprot_to_seq.get(uniprot)
        if not seq:
            continue
        for a in fetch_activities_for_target(chembl_id):
            if not _qc_ok(a):
                continue
            key = (uniprot, a["canonical_smiles"])
            bucket_vals[key].append(float(a["pchembl_value"]))
            if key not in bucket_meta:
                bucket_meta[key] = {
                    "molecule_id":    a.get("molecule_chembl_id", ""),
                    "assay_type":     a.get("standard_type", ""),
                    "chembl_target":  chembl_id,
                }

    per_target = Counter(k[0] for k in bucket_vals)
    sparse = {u for u, n in per_target.items() if n < MIN_COMPOUNDS_PER_TARGET}
    if sparse:
        print(f"  Dropping {len(sparse)} targets with <{MIN_COMPOUNDS_PER_TARGET} compounds")

    rows: list[dict] = []
    for (uniprot, smiles), vals in bucket_vals.items():
        if uniprot in sparse:
            continue
        seq = uniprot_to_seq[uniprot]
        pchembl = float(np.median(vals))
        score = pchembl_to_score(pchembl)
        meta = bucket_meta[(uniprot, smiles)]
        rows.append({
            "uniprot":       uniprot,
            "chembl_target": meta["chembl_target"],
            "molecule_id":   meta["molecule_id"],
            "smiles":        smiles,
            "sequence":      seq,
            "pchembl_value": round(pchembl, 3),
            "score":         score,
            "score_str":     score_to_str(score),
            "assay_type":    meta["assay_type"],
            "n_replicates":  len(vals),
            "prompt":        format_prompt(smiles, seq),
        })
    return rows


# ── Phase 5: splits ──────────────────────────────────────────────────────────

def _three_way(items: list, fracs: tuple[float, float, float], seed: int) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * fracs[0])
    n_val   = int(n * fracs[1])
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def random_split(rows: list[dict]) -> dict[str, list[dict]]:
    train, val, test = _three_way(rows, SPLIT_FRACS, SPLIT_SEED)
    return {"train": train, "val": val, "test": test}


def _murcko_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(scaf, canonical=True)
        return s or None
    except Exception:
        return None


def scaffold_split(rows: list[dict]) -> dict[str, list[dict]]:
    """Bemis–Murcko split: most common scaffolds → train, rarest → test."""
    by_scaffold: dict[str, list[dict]] = defaultdict(list)
    no_scaffold: list[dict] = []
    for r in tqdm(rows, desc="Computing scaffolds", unit="cmpd"):
        scaf = _murcko_scaffold(r["smiles"])
        (no_scaffold if scaf is None else by_scaffold[scaf]).append(r)

    groups = sorted(by_scaffold.values(), key=lambda g: (-len(g), g[0]["smiles"]))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(no_scaffold)

    n_total = sum(len(g) for g in groups) + len(no_scaffold)
    n_train_t = int(n_total * SPLIT_FRACS[0])
    n_val_t   = int(n_total * SPLIT_FRACS[1])

    train: list[dict] = []
    val:   list[dict] = []
    test:  list[dict] = []
    for g in groups:
        if   len(train) < n_train_t: train.extend(g)
        elif len(val)   < n_val_t:   val.extend(g)
        else:                        test.extend(g)
    for r in no_scaffold:
        if   len(train) < n_train_t: train.append(r)
        elif len(val)   < n_val_t:   val.append(r)
        else:                        test.append(r)

    return {"train": train, "val": val, "test": test}


def target_split(rows: list[dict]) -> dict[str, list[dict]]:
    """Cold-target split: UniProts disjoint across splits."""
    by_target: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_target[r["uniprot"]].append(r)
    targets = list(by_target.keys())
    train_t, val_t, test_t = _three_way(targets, SPLIT_FRACS, SPLIT_SEED)
    return {
        "train": [r for t in train_t for r in by_target[t]],
        "val":   [r for t in val_t   for r in by_target[t]],
        "test":  [r for t in test_t  for r in by_target[t]],
    }


# ── Phase 6: writing ─────────────────────────────────────────────────────────

CSV_FIELDS = [
    "uniprot", "chembl_target", "molecule_id", "smiles",
    "pchembl_value", "score", "score_str", "assay_type", "n_replicates",
    "prompt",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_split_csvs(splits: dict[str, list[dict]], base_dir: Path) -> None:
    for name, rows in splits.items():
        _write_csv(base_dir / f"{name}.csv", rows)


def _bucket_pchembl(rows: list[dict]) -> dict[str, int]:
    b = Counter()
    for r in rows:
        v = r["pchembl_value"]
        if   v >= 8: b["≥8 (strong)"] += 1
        elif v >= 7: b["7–8"]         += 1
        elif v >= 6: b["6–7"]         += 1
        elif v >= 5: b["5–6"]         += 1
        else:        b["<5 (weak)"]   += 1
    return dict(b)


def write_stats(rows: list[dict], all_splits: dict[str, dict[str, list[dict]]]) -> None:
    L: list[str] = []
    L.append("# txGemma fine-tune dataset stats")
    L.append("")
    L.append(f"Total compound–target pairs: {len(rows)}")
    L.append(f"Unique UniProts:             {len({r['uniprot'] for r in rows})}")
    L.append(f"Unique SMILES:               {len({r['smiles'] for r in rows})}")
    L.append(f"Held-out (excluded):         {sorted(HELD_OUT_UNIPROTS)}")
    L.append("")
    L.append("## pChEMBL distribution")
    for k, v in sorted(_bucket_pchembl(rows).items()):
        L.append(f"  {k:14s}  {v}")
    L.append("")
    scores = np.array([r["score"] for r in rows])
    L.append("## Score (0–1000) distribution")
    L.append(f"  min={int(scores.min())}  median={int(np.median(scores))}  "
             f"max={int(scores.max())}  std={scores.std():.1f}")
    L.append("")
    L.append("## Compounds per target (top 10)")
    for u, n in Counter(r["uniprot"] for r in rows).most_common(10):
        L.append(f"  {u}  {n}")
    L.append("")
    L.append("## Splits")
    for split_name, splits in all_splits.items():
        L.append(f"### {split_name}")
        for k, v in splits.items():
            L.append(f"  {k:5s}  pairs={len(v):>6d}  "
                     f"targets={len({r['uniprot'] for r in v}):>4d}  "
                     f"unique_smiles={len({r['smiles'] for r in v}):>6d}")
        tr_t = {r['uniprot'] for r in splits['train']}
        va_t = {r['uniprot'] for r in splits['val']}
        te_t = {r['uniprot'] for r in splits['test']}
        L.append(f"  target overlap train∩val={len(tr_t & va_t)} "
                 f"train∩test={len(tr_t & te_t)} val∩test={len(va_t & te_t)}")
        # Confirm held-out never appears.
        leaked = {r['uniprot'] for r in splits['train'] + splits['val'] + splits['test']} & HELD_OUT_UNIPROTS
        L.append(f"  held-out leak check: {'NONE' if not leaked else 'LEAKED ' + str(sorted(leaked))}")
        L.append("")
    STATS_TXT.write_text("\n".join(L))


def write_split_metadata(all_splits: dict[str, dict[str, list[dict]]]) -> None:
    meta = {
        "seed":               SPLIT_SEED,
        "fractions":          {"train": SPLIT_FRACS[0], "val": SPLIT_FRACS[1], "test": SPLIT_FRACS[2]},
        "held_out_uniprots":  sorted(HELD_OUT_UNIPROTS),
        "score_mapping": {
            "formula":   "1000 * (1 - (pChEMBL - floor) / (ceil - floor))",
            "floor":     SCORE_PCHEMBL_FLOOR,
            "ceil":      SCORE_PCHEMBL_CEIL,
            "direction": "low score = strong binder (TDC BindingDB_ki convention)",
        },
        "splits": {
            split_name: {
                fold: {
                    "n_pairs":   len(rows),
                    "n_targets": len({r['uniprot'] for r in rows}),
                    "n_smiles":  len({r['smiles']  for r in rows}),
                }
                for fold, rows in splits.items()
            }
            for split_name, splits in all_splits.items()
        },
    }
    SPLIT_META_JSON.write_text(json.dumps(meta, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("── Phase 1: Load GPCRs from local DB ──")
    uniprot_to_seq = load_gpcrs_from_db()
    print(f"  Found {len(uniprot_to_seq)} GPCRs in DB")
    held_present = sorted(HELD_OUT_UNIPROTS & set(uniprot_to_seq))
    for u in held_present:
        del uniprot_to_seq[u]
    print(f"  Excluded held-out: {held_present}")
    print(f"  Eligible for training: {len(uniprot_to_seq)}")

    print("\n── Phase 2: Resolve UniProt → ChEMBL target IDs ──")
    mapping = resolve_uniprot_to_chembl(list(uniprot_to_seq.keys()))
    # Defensive: drop any held-out that may have slipped into a cached mapping.
    for u in HELD_OUT_UNIPROTS:
        mapping.pop(u, None)
    print(f"  Resolved {len(mapping)}/{len(uniprot_to_seq)} targets")

    print("\n── Phase 3+4: Fetch + dedupe activities ──")
    rows = build_dataset_rows(mapping, uniprot_to_seq)
    # Final safety net.
    rows = [r for r in rows if r["uniprot"] not in HELD_OUT_UNIPROTS]
    print(f"  {len(rows)} (target, SMILES) pairs after QC + dedup + safety filter")
    if not rows:
        sys.exit("ERROR: no rows after filtering — aborting.")

    _write_csv(ALL_PAIRS_CSV, rows)
    print(f"  Wrote {ALL_PAIRS_CSV.relative_to(REPO_ROOT)}")

    print("\n── Phase 5: Building splits ──")
    print("  random …");                all_splits = {"random":   random_split(rows)}
    print("  scaffold (cold-ligand) …"); all_splits["scaffold"] = scaffold_split(rows)
    print("  target (cold-target) …");   all_splits["target"]   = target_split(rows)

    for name, splits in all_splits.items():
        write_split_csvs(splits, DATA_DIR / f"{name}_split")

    write_split_metadata(all_splits)
    write_stats(rows, all_splits)

    print("\n── Done ──")
    print(f"  Stats:           {STATS_TXT.relative_to(REPO_ROOT)}")
    print(f"  Split metadata:  {SPLIT_META_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
