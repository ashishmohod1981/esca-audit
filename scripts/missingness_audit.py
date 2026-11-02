"""Missingness audit of the Stage 2 endpoints (paper Section III, "Missingness Audit", and Table IX).

The question is simple: can the label be guessed from which modalities a patient happens to
have, without looking at any feature value? If it can, a fusion model may learn the pattern of
data collection instead of the tumor, and the endpoint should not be reported.

Three checks are run for each endpoint (histological subtype and stage I and II vs III and IV):

1. Label completeness: the share of patients whose label is actually recorded.
2. Availability only baseline: a logistic regression that sees only the three availability flags
   (CT, PET, genomics), scored by AUROC under five fold cross validation.
3. Source cross tabulation: among patients without a genomic profile, how many have CT and PET,
   how many have a recorded stage, and how often that stage is III or IV, compared with the
   TCGA patients.

Usage (from the repository root)

    # On Kaggle, with the feature stores attached (same paths as the pipeline):
    python scripts/missingness_audit.py --data-root /kaggle/input/.../mm-dataset \
        --curated-manifest /path/to/multimodal_master_dataset_pruned1.csv

    # Anywhere, with only a manifest (availability taken from the manifest columns):
    python scripts/missingness_audit.py --manifest-only /path/to/multimodal_master_dataset_pruned1.csv

In the paper, subtype passed (availability only AUROC 0.561 against 0.870 for the model) and
stage failed (stage readable for 32 of 199 patients in the analytic labels; in the curated
manifest all 19 patients without a genomic profile came from one collection outside TCGA, 18 had
CT, and 15 of them (79%) were stage III or IV, against 57 of 183 TCGA patients (31%)). The
availability only AUROC can move in the third decimal with the fold seed and with the source of
the availability flags.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ("Squamous Cell Carcinoma", "Adenocarcinoma")


def stage_label(s) -> float:
    """1 = stage III or IV, 0 = stage I or II, NaN = not recorded (same rule as the pipeline)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).upper().replace("_", " ").replace("-", " ").strip()
    if t in ("", "NAN", "NONE", "NA", "N/A", "UNKNOWN", "[NOT AVAILABLE]", "[DISCREPANCY]", "[UNKNOWN]", "NOT REPORTED"):
        return np.nan
    t = re.sub(r"\bSTAGE\b", " ", t)
    m = re.search(r"\b(IV|III|II|I)[ABC]?\d?\b", t)
    if m:
        return 1.0 if m.group(1) in ("III", "IV") else 0.0
    m = re.search(r"\b([1-4])(\.0)?[ABC]?\b", t)
    if m:
        return 1.0 if m.group(1) in ("3", "4") else 0.0
    return np.nan


def availability_only_auroc(mask: np.ndarray, y: np.ndarray, seed: int = 0, folds: int = 5) -> Optional[float]:
    """Five fold cross validated AUROC of a logistic regression on the availability flags only."""
    y = np.asarray(y).astype(int)
    if np.bincount(y, minlength=2).min() < folds:
        return None
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    s = cross_val_predict(LogisticRegression(max_iter=1000), mask, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, s))


def manifest_availability(df: pd.DataFrame) -> np.ndarray:
    """CT, PET and genomics flags read from the manifest columns."""
    mods = df.get("Radiology_Endoscopy_Modalities", pd.Series("", index=df.index)).fillna("").astype(str)
    ct = mods.str.contains(r"\bCT\b").astype(float)
    pet = mods.str.contains(r"\bPET\b").astype(float)
    g = df.get("Has_Genomics_WSI", pd.Series(0, index=df.index)).fillna(0).astype(float)
    return np.c_[ct, pet, g]


def source_crosstab(man: pd.DataFrame, id_col: str = "Patient_ID", stage_col: str = "AJCC_Pathologic_Stage") -> Dict:
    """Cross tabulation behind the stage verdict, computed from a curated manifest."""
    df = man[man["Unified_Diagnosis_Merged"].isin(LABELS)].copy()
    mask = manifest_availability(df)
    late = df[stage_col].map(stage_label).values
    tcga = df[id_col].astype(str).str.startswith("TCGA").values
    no_g = mask[:, 2] == 0
    known = ~np.isnan(late)
    out = {
        "patients": int(len(df)),
        "stage_recorded": int(known.sum()),
        "no_genomics_n": int(no_g.sum()),
        "no_genomics_outside_tcga": int((no_g & ~tcga).sum()),
        "no_genomics_with_ct": int((no_g & (mask[:, 0] == 1)).sum()),
        "no_genomics_with_ct_and_pet": int((no_g & (mask[:, 0] == 1) & (mask[:, 1] == 1)).sum()),
        "no_genomics_stage_recorded": int((no_g & known).sum()),
        "no_genomics_late": int(np.nansum(late[no_g])),
        "tcga_n": int(tcga.sum()),
        "tcga_late": int(np.nansum(late[tcga])),
    }
    out["no_genomics_late_pct"] = round(100 * out["no_genomics_late"] / max(out["no_genomics_n"], 1), 1)
    out["tcga_late_pct"] = round(100 * out["tcga_late"] / max(out["tcga_n"], 1), 1)
    return out


def verdict(auroc: Optional[float], complete_pct: float) -> str:
    if complete_pct < 50:
        return "fails: label mostly missing"
    if auroc is None:
        return "cannot evaluate (class too small)"
    if auroc >= 0.70:
        return "fails: availability alone predicts the label"
    return "passes: availability alone is close to chance"


def audit_pipeline(data_root: str, curated: Optional[str]) -> Dict:
    """Audit on the analytic cohort exactly as the pipeline builds it (needs the feature stores)."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import esca_sci_pipeline as P  # imported here so that the manifest mode does not need torch

    cfg = P.Config(data_root=data_root)
    S = P.load_stage2(cfg)                      # analytic cohort, availability mask, subtype labels
    man = pd.read_csv(cfg.path(cfg.manifest), low_memory=False)
    man[cfg.id_col] = man[cfg.id_col].astype(str)
    raw_stage = man.set_index(cfg.id_col).loc[S["ids"], cfg.s2_stage_col].map(stage_label).values
    n = len(S["ids"])
    known = ~np.isnan(raw_stage)
    rep = {
        "mode": "pipeline",
        "patients": n,
        "modality_coverage_CT_PET_G": S["mask"].sum(0).astype(int).tolist(),
        "subtype_readable": n,
        "stage_readable": int(known.sum()),
        "availability_only_AUROC_subtype": availability_only_auroc(S["mask"], S["dx"]),
        "availability_only_AUROC_stage_recorded_only": availability_only_auroc(S["mask"][known], raw_stage[known])
        if known.sum() >= 10 else None,
    }
    rep["verdict_subtype"] = verdict(rep["availability_only_AUROC_subtype"], 100.0)
    rep["verdict_stage"] = verdict(rep["availability_only_AUROC_stage_recorded_only"], 100 * known.mean())
    if curated:
        rep["curated_manifest_crosstab"] = source_crosstab(pd.read_csv(curated, low_memory=False))
    return rep


def audit_manifest(path: str) -> Dict:
    """Audit from a manifest alone (availability from the manifest columns)."""
    man = pd.read_csv(path, low_memory=False)
    df = man[man["Unified_Diagnosis_Merged"].isin(LABELS)].copy()
    mask = manifest_availability(df)
    dx = (df["Unified_Diagnosis_Merged"] == "Adenocarcinoma").astype(int).values
    stage = df["AJCC_Pathologic_Stage"].map(stage_label).values
    known = ~np.isnan(stage)
    rep = {
        "mode": "manifest only",
        "patients": int(len(df)),
        "subtype_readable": int(len(df)),
        "stage_readable": int(known.sum()),
        "availability_only_AUROC_subtype": availability_only_auroc(mask, dx),
        "curated_manifest_crosstab": source_crosstab(man),
    }
    rep["verdict_subtype"] = verdict(rep["availability_only_AUROC_subtype"], 100.0)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", help="folder with the manifest and the CT, PET and genomic feature stores")
    ap.add_argument("--curated-manifest", help="curated manifest used for the source cross tabulation")
    ap.add_argument("--manifest-only", help="run from a manifest alone")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "missingness_audit_report.json"))
    a = ap.parse_args()
    if a.manifest_only:
        rep = audit_manifest(a.manifest_only)
    elif a.data_root:
        rep = audit_pipeline(a.data_root, a.curated_manifest)
    else:
        ap.error("give --data-root (with feature stores) or --manifest-only")
    print(json.dumps(rep, indent=2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2)
    print(f"report written to {a.out}")


if __name__ == "__main__":
    main()
