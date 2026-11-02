"""Recompute the summary statistics quoted in the paper from the saved predictions.

The Kaggle pipeline (src/esca_sci_pipeline.py) writes the predictions; this script only reads
them, so it runs in a few seconds on a laptop without a GPU.

Run from the repository root:
    python scripts/reproduce_paper_statistics.py

It prints each statistic next to the value reported in the paper and writes
results/paper_statistics.json. The script exits with status 1 if any value disagrees with the
paper after rounding to the precision used in the text.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


# ----------------------------------------------------------------------------- helpers
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope(y, p):
    """Slope of a logistic regression of the outcome on logit(p); 1 is ideal."""
    # A very large C makes the fit effectively unpenalized.
    lr = LogisticRegression(C=1e12, max_iter=1000).fit(logit(p)[:, None], y)
    return float(lr.coef_[0, 0])


def brier_skill(y, p):
    prev = y.mean()
    return 1.0 - brier_score_loss(y, p) / (prev * (1 - prev))


def ece(y, p, n_bins=15):
    """Expected calibration error over bins of equal size (same rule as the pipeline)."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))


def conformal_quantiles(y, p, alpha):
    """Class conditional (Mondrian) quantiles of the score 1 - q_y(x)."""
    q = {}
    for c in (0, 1):
        s = 1 - (p[y == 1] if c == 1 else 1 - p[y == 0])
        n = len(s)
        k = min(n, int(math.ceil((n + 1) * (1 - alpha))))
        q[c] = float(np.sort(s)[k - 1])
    return q


def conformal_sets(p, q):
    return np.c_[p <= q[0], (1 - p) <= q[1]]


def error_after_referral(err, unc, frac):
    """Error rate among retained images after referring the most uncertain fraction."""
    n = len(err)
    keep = np.argsort(unc)[: n - int(math.ceil(frac * n))]
    return float(err[keep].mean())


# ----------------------------------------------------------------------------- data
thr = json.load(open(os.path.join(RES, "stage1_results.json")))["operating_points_oof"]
ti = pd.read_csv(os.path.join(RES, "predictions", "internal_test_predictions.csv"))
oof = pd.read_csv(os.path.join(RES, "predictions", "internal_oof_predictions.csv"))
ex = pd.read_csv(os.path.join(RES, "predictions", "external_predictions_recomputed.csv"))

y, p_raw, q = ti.y.values, ti.p_raw.values, ti.p_cal.values
ye, qe = oof.y.values, oof.p_cal.values
zl = ex[ex.cohort.isin(["barretts", "zline"])]
yz, qz = zl.label.values, zl.p_cal.values

out = {}
# Internal discrimination and calibration
out["internal_AUROC"] = (roc_auc_score(y, q), 0.914, 3)
out["internal_AUPRC"] = (average_precision_score(y, q), 0.569, 3)
out["internal_ECE_raw"] = (ece(y, p_raw), 0.080, 3)
out["internal_ECE_calibrated"] = (ece(y, q), 0.044, 3)
out["internal_Brier"] = (brier_score_loss(y, q), 0.060, 3)
out["internal_calibration_slope"] = (calibration_slope(y, q), 1.00, 2)
out["internal_Brier_skill_score"] = (brier_skill(y, q), 0.29, 2)

# Workload on the internal test set
hs = q >= thr["Sensitivity>=95%"]
out["highsens_images_cleared"] = (int((~hs).sum()), 75, 0)
out["highsens_BE_missed"] = (int(((~hs) & (y == 1)).sum()), 0, 0)
qc = conformal_quantiles(ye, qe, 0.10)
sets = conformal_sets(q, qc)
single = sets.sum(1) == 1
out["conformal_single_label_n"] = (int(single.sum()), 149, 0)
out["conformal_single_label_accuracy"] = (float((sets[single, 1] == (y[single] == 1)).mean()), 0.906, 3)
out["conformal_referred_n"] = (int((~single).sum()), 55, 0)
out["conformal_referred_BE"] = (int(((~single) & (y == 1)).sum()), 4, 0)

# Selective deferral by epistemic uncertainty (errors at the F1 threshold)
err = ((q >= thr["F1-optimal"]).astype(int) != y)
out["internal_errors_F1"] = (int(err.sum()), 15, 0)
out["error_refer_10pct"] = (100 * error_after_referral(err, ti.epistemic.values, 0.10), 4.4, 1)
out["error_refer_20pct"] = (100 * error_after_referral(err, ti.epistemic.values, 0.20), 1.8, 1)
out["error_refer_30pct"] = (100 * error_after_referral(err, ti.epistemic.values, 0.30), 0.7, 1)
out["epistemic_error_detection_AUROC"] = (roc_auc_score(err, ti.epistemic.values), 0.898, 3)

# External, BE versus normal Z line, everything frozen
out["external_AUROC"] = (roc_auc_score(yz, qz), 0.872, 3)
out["external_AUPRC"] = (average_precision_score(yz, qz), 0.698, 3)
out["external_ECE"] = (ece(yz, qz), 0.060, 3)
out["external_Brier"] = (brier_score_loss(yz, qz), 0.113, 3)
out["external_calibration_slope"] = (calibration_slope(yz, qz), 0.89, 2)
out["external_Brier_skill_score"] = (brier_skill(yz, qz), 0.35, 2)
erz = ((qz >= thr["F1-optimal"]).astype(int) != yz)
out["external_error_no_referral"] = (100 * erz.mean(), 15.6, 1)
out["external_error_refer_20pct"] = (100 * error_after_referral(erz, zl.epistemic.values, 0.20), 11.2, 1)
out["external_error_refer_30pct"] = (100 * error_after_referral(erz, zl.epistemic.values, 0.30), 7.1, 1)

# ----------------------------------------------------------------------------- report
bad = 0
print(f"{'statistic':38s} {'computed':>10s} {'paper':>8s}  check")
report = {}
for k, (val, paper, nd) in out.items():
    ok = round(float(val), nd) == round(float(paper), nd)
    bad += not ok
    report[k] = {"computed": float(val), "paper": paper, "matches": bool(ok)}
    print(f"{k:38s} {val:10.{max(nd, 0)}f} {paper:8.{max(nd, 0)}f}  {'ok' if ok else 'DIFFERS'}")
json.dump(report, open(os.path.join(RES, "paper_statistics.json"), "w"), indent=2)
print(f"\n{len(out) - bad} of {len(out)} statistics match the paper; report saved to results/paper_statistics.json")
sys.exit(1 if bad else 0)
