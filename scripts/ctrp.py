#!/usr/bin/env python3
"""Download and harmonize PharmacoDB-curated CTRPv2 for external validation.

This script is intentionally read-only with respect to training code. It creates
CSV/JSON artifacts under the chosen output directory:

- ctrpv2_cells.csv
- drugs.csv
- ctrpv2_shared_compounds.csv
- pairs.csv
- ctrpv2_external_pairs_no_gdsc_train.csv
- ctrpv2_external_pairs_no_gdsc_response.csv
- harmonization_report.json

Endpoint note: PharmacoDB exposes AAC (activity area), where larger values
typically indicate greater drug activity/sensitivity. For models trained on
GDSC logIC50, the evaluation script uses resistance_rank = -AAC.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from sklearn.model_selection import train_test_split


GRAPHQL_URL = "https://pharmacodb.ca/graphql"


def graphql(query: str, variables: dict[str, Any] | None = None, *, timeout: int = 120) -> dict[str, Any]:
    payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "BCGDRP-CTRPv2-external/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    if data.get("errors"):
        raise RuntimeError(json.dumps(data["errors"], ensure_ascii=False))
    return data["data"]


def graphql_with_retry(query: str, variables: dict[str, Any] | None = None, retries: int = 4) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return graphql(query, variables)
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            sleep_s = min(60, 5 * (attempt + 1))
            print(f"[retry] GraphQL attempt {attempt + 1}/{retries} failed: {exc}; sleep {sleep_s}s", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"GraphQL failed after {retries} attempts: {last_error}") from last_error


def norm_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def rdkit_inchikey(smiles: Any) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    try:
        from rdkit import Chem
    except Exception:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol) or ""
    except Exception:
        return ""


def load_gdsc_training_cells(root: Path) -> tuple[set[str], set[str], set[str]]:
    response = pd.read_csv(root / "data/GDSC2_IC50.csv", index_col=0)
    cell_ids = sorted(map(str, response.index.values))
    indices = np.arange(len(cell_ids))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    train_cells = {cell_ids[i] for i in train_idx}
    val_test_cells = {cell_ids[i] for i in val_idx} | {cell_ids[i] for i in test_idx}
    all_response_cells = set(cell_ids)
    return train_cells, val_test_cells, all_response_cells


def build_cell_mapping(root: Path, ctrp_cells: pd.DataFrame) -> pd.DataFrame:
    cell_info = pd.read_csv(root / "data/cell/cell_line_info.csv")
    feature_cells = set(pd.read_csv(root / "data/cell/geo_expression_cosmic.csv", index_col=0).index.astype(str))
    train_cells, val_test_cells, all_response_cells = load_gdsc_training_cells(root)

    name_to_ach: dict[str, set[str]] = {}
    for _, row in cell_info.iterrows():
        ach = str(row["DepMap_ID"])
        candidates = [
            row.get("cell_line_name", ""),
            row.get("stripped_cell_line_name", ""),
            row.get("CCLE_Name", ""),
            str(row.get("CCLE_Name", "")).split("_")[0],
        ]
        for candidate in candidates:
            key = norm_name(candidate)
            if key:
                name_to_ach.setdefault(key, set()).add(ach)

    mapped_rows = []
    for _, row in ctrp_cells.iterrows():
        keys = [norm_name(row["cell_name"]), norm_name(str(row["cell_uid"]).split("_")[0])]
        candidates: set[str] = set()
        for key in keys:
            candidates.update(name_to_ach.get(key, set()))
        candidates = {x for x in candidates if x in feature_cells}
        if len(candidates) == 1:
            ach = next(iter(candidates))
            status = "matched"
        elif len(candidates) > 1:
            ach = ""
            status = "ambiguous:" + ";".join(sorted(candidates))
        else:
            ach = ""
            status = "unmatched"
        mapped_rows.append(
            {
                "pharmacodb_cell_id": row["cell_id"],
                "cell_uid": row["cell_uid"],
                "cell_name": row["cell_name"],
                "depmap_id": ach,
                "mapping_status": status,
                "has_feature": bool(ach and ach in feature_cells),
                "in_gdsc_train": bool(ach and ach in train_cells),
                "in_gdsc_val_or_test": bool(ach and ach in val_test_cells),
                "in_gdsc_response": bool(ach and ach in all_response_cells),
            }
        )
    return pd.DataFrame(mapped_rows)


def build_shared_compounds(root: Path, ctrp_compounds: pd.DataFrame) -> pd.DataFrame:
    gdsc = pd.read_csv(root / "data/drug_info.csv")
    gdsc["pubchem_id"] = gdsc["pubchem_id"].astype(str)
    gdsc["gdsc_inchikey"] = gdsc["canonicalsmiles"].apply(rdkit_inchikey)

    by_inchikey = {
        key: row
        for key, row in gdsc.dropna(subset=["gdsc_inchikey"]).set_index("gdsc_inchikey").iterrows()
        if str(key).strip()
    }
    by_pubchem = {str(row["pubchem_id"]): row for _, row in gdsc.iterrows() if str(row["pubchem_id"]).strip()}

    shared = []
    for _, row in ctrp_compounds.iterrows():
        match = None
        match_type = ""
        inchikey = str(row.get("inchikey") or "").strip()
        pubchem = str(row.get("pubchem") or "").strip()
        if inchikey and inchikey in by_inchikey:
            match = by_inchikey[inchikey]
            match_type = "inchikey"
        elif pubchem and pubchem in by_pubchem:
            match = by_pubchem[pubchem]
            match_type = "pubchem"
        if match is None:
            continue
        shared.append(
            {
                "pharmacodb_compound_id": int(row["compound_id"]),
                "compound_uid": row["compound_uid"],
                "ctrp_compound_name": row["compound_name"],
                "ctrp_pubchem": pubchem,
                "ctrp_inchikey": inchikey,
                "ctrp_smiles": row.get("smiles", ""),
                "gdsc_pubchem_id": str(match["pubchem_id"]),
                "gdsc_drug_name": match.get("drug_name", ""),
                "gdsc_inchikey": match.get("gdsc_inchikey", ""),
                "match_type": match_type,
            }
        )
    return pd.DataFrame(shared).drop_duplicates(subset=["pharmacodb_compound_id", "gdsc_pubchem_id"])


def fetch_ctrpv2_metadata(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_file = cache_dir / "pharmacodb_ctrpv2_dataset_and_compounds.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        query = """
        query {
          dataset(datasetId: 2) {
            id
            name
            cell_count
            compound_tested_count
            experiment_count
            cells_tested { id uid name }
            compounds_tested { id uid name }
          }
          compounds(all: true) {
            id
            uid
            name
            annotation { pubchem inchikey smiles }
          }
        }
        """
        data = graphql_with_retry(query)
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    dataset = data["dataset"][0]
    ctrp_ids = {int(x["id"]) for x in dataset["compounds_tested"]}
    annotations = {
        int(row["id"]): (row.get("annotation") or {})
        for row in data["compounds"]
        if int(row["id"]) in ctrp_ids
    }

    cells = pd.DataFrame(
        [
            {"cell_id": int(x["id"]), "cell_uid": x["uid"], "cell_name": x["name"]}
            for x in dataset["cells_tested"]
        ]
    )
    compounds = []
    for x in dataset["compounds_tested"]:
        ann = annotations.get(int(x["id"]), {})
        compounds.append(
            {
                "compound_id": int(x["id"]),
                "compound_uid": x["uid"],
                "compound_name": x["name"],
                "pubchem": ann.get("pubchem") or "",
                "inchikey": ann.get("inchikey") or "",
                "smiles": ann.get("smiles") or "",
            }
        )
    return cells, pd.DataFrame(compounds)


def fetch_experiments_for_compound(compound_id: int, cache_dir: Path) -> list[dict[str, Any]]:
    cache_file = cache_dir / f"pharmacodb_compound_{compound_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["experiments"]
    query = """
    query($compoundId: Int!) {
      experiments(compoundId: $compoundId, all: true) {
        id
        cell_line { id uid name }
        tissue { id name }
        compound { id uid name }
        dataset { id name }
        profile { HS Einf EC50 AAC IC50 DSS1 DSS2 DSS3 }
      }
    }
    """
    data = graphql_with_retry(query, {"compoundId": int(compound_id)})
    cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data["experiments"]


def build_external_pairs(shared: pd.DataFrame, cell_map: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    cell_lookup = cell_map.set_index("pharmacodb_cell_id").to_dict(orient="index")
    shared_lookup = shared.set_index("pharmacodb_compound_id").to_dict(orient="index")

    rows = []
    for n, compound_id in enumerate(shared["pharmacodb_compound_id"].astype(int).tolist(), start=1):
        print(f"[experiments] {n}/{len(shared)} compound_id={compound_id}", flush=True)
        for exp in fetch_experiments_for_compound(compound_id, cache_dir):
            dataset = exp.get("dataset") or {}
            if dataset.get("name") != "CTRPv2":
                continue
            profile = exp.get("profile") or {}
            aac = profile.get("AAC")
            if aac is None:
                continue
            cell = exp.get("cell_line") or {}
            mapped = cell_lookup.get(int(cell["id"]))
            if not mapped or mapped["mapping_status"] != "matched":
                continue
            comp = shared_lookup[int(compound_id)]
            rows.append(
                {
                    "experiment_id": exp["id"],
                    "pharmacodb_cell_id": int(cell["id"]),
                    "cell_uid": cell.get("uid", ""),
                    "cell_name": cell.get("name", ""),
                    "depmap_id": mapped["depmap_id"],
                    "in_gdsc_train": mapped["in_gdsc_train"],
                    "in_gdsc_val_or_test": mapped["in_gdsc_val_or_test"],
                    "in_gdsc_response": mapped["in_gdsc_response"],
                    "pharmacodb_compound_id": int(compound_id),
                    "compound_uid": comp["compound_uid"],
                    "ctrp_compound_name": comp["ctrp_compound_name"],
                    "gdsc_pubchem_id": comp["gdsc_pubchem_id"],
                    "gdsc_drug_name": comp["gdsc_drug_name"],
                    "match_type": comp["match_type"],
                    "aac": float(aac),
                    "resistance_rank": float(-float(aac)),
                    "ic50": profile.get("IC50"),
                    "ec50": profile.get("EC50"),
                    "hs": profile.get("HS"),
                    "einf": profile.get("Einf"),
                    "dss1": profile.get("DSS1"),
                    "dss2": profile.get("DSS2"),
                    "dss3": profile.get("DSS3"),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs = pairs.drop_duplicates(subset=["depmap_id", "gdsc_pubchem_id", "experiment_id"])
    return pairs


def count_eligible_drugs(pairs: pd.DataFrame, min_cells: int) -> int:
    if pairs.empty:
        return 0
    counts = pairs.groupby("gdsc_pubchem_id")["depmap_id"].nunique()
    return int((counts >= min_cells).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", default="outputs/ctrp_download")
    parser.add_argument("--min-cells-per-drug", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    raw_dir = out_dir / "raw_graphql"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cells, compounds = fetch_ctrpv2_metadata(raw_dir)
    cell_map = build_cell_mapping(root, cells)
    shared = build_shared_compounds(root, compounds)
    pairs = build_external_pairs(shared, cell_map, raw_dir)

    cells.to_csv(out_dir / "ctrpv2_cells.csv", index=False)
    compounds.to_csv(out_dir / "drugs.csv", index=False)
    cell_map.to_csv(out_dir / "ctrpv2_cell_mapping.csv", index=False)
    shared.to_csv(out_dir / "ctrpv2_shared_compounds.csv", index=False)
    pairs.to_csv(out_dir / "pairs.csv", index=False)

    no_train = pairs.loc[~pairs["in_gdsc_train"].astype(bool)].copy()
    no_response = pairs.loc[~pairs["in_gdsc_response"].astype(bool)].copy()
    no_train.to_csv(out_dir / "ctrpv2_external_pairs_no_gdsc_train.csv", index=False)
    no_response.to_csv(out_dir / "ctrpv2_external_pairs_no_gdsc_response.csv", index=False)

    report = {
        "source": "PharmacoDB GraphQL datasetId=2 (CTRPv2)",
        "endpoint": GRAPHQL_URL,
        "endpoint_note": "PharmacoDB profile.AAC is activity area; resistance_rank is defined as -AAC.",
        "ctrv2_cells": int(len(cells)),
        "ctrv2_compounds": int(len(compounds)),
        "cell_mapping": cell_map["mapping_status"].value_counts().to_dict(),
        "matched_cells_with_features": int((cell_map["mapping_status"] == "matched").sum()),
        "shared_compounds_total": int(len(shared)),
        "shared_compounds_by_match_type": shared["match_type"].value_counts().to_dict(),
        "pairs_all": int(len(pairs)),
        "pairs_no_gdsc_train": int(len(no_train)),
        "pairs_no_gdsc_response": int(len(no_response)),
        "unique_cells_all": int(pairs["depmap_id"].nunique()) if not pairs.empty else 0,
        "unique_cells_no_gdsc_train": int(no_train["depmap_id"].nunique()) if not no_train.empty else 0,
        "unique_cells_no_gdsc_response": int(no_response["depmap_id"].nunique()) if not no_response.empty else 0,
        "eligible_drugs_all_min_cells": count_eligible_drugs(pairs, args.min_cells_per_drug),
        "eligible_drugs_no_gdsc_train_min_cells": count_eligible_drugs(no_train, args.min_cells_per_drug),
        "eligible_drugs_no_gdsc_response_min_cells": count_eligible_drugs(no_response, args.min_cells_per_drug),
        "min_cells_per_drug": int(args.min_cells_per_drug),
    }
    (out_dir / "harmonization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
