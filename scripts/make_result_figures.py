"""Result figures of the paper (Figs. 4, 5 and 7) drawn from the saved predictions.

Run from the repository root:
    python scripts/make_result_figures.py

Inputs : results/predictions/*.csv and results/tables/S1_conformal_test.csv
Outputs: figures/results/fig_internal.(pdf|png), fig_uncertainty.(pdf|png), fig_external.(pdf|png)
"""
import sys, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures", "results")
sys.path.insert(0, os.path.join(ROOT, "src"))
import esca_sci_pipeline as P

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#7d55c7"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "legend.frameon": False, "lines.linewidth": 1.3, "savefig.dpi": 600,
    "pdf.fonttype": 42, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
W2 = 7.16
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def band(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed); g = np.linspace(0, 1, 201)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; T = []
    for _ in range(n):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        f, t, _ = roc_curve(y[i], p[i]); T.append(np.interp(g, f, t))
    return g, np.quantile(T, .025, 0), np.quantile(T, .975, 0)


ti = pd.read_csv(f"{RES}/predictions/internal_test_predictions.csv")
oof = pd.read_csv(f"{RES}/predictions/internal_oof_predictions.csv")
ex = pd.read_csv(f"{RES}/predictions/external_predictions_recomputed.csv")
y, p_raw, q = ti.y.values, ti.p_raw.values, ti.p_cal.values
THR_F1 = 0.3421537117124577  # F1 threshold fixed out of fold (results/stage1_results.json)

# ---------------------------------------------------------------- Fig. 4 internal performance
fig, ax = plt.subplots(1, 4, figsize=(W2, 1.85), gridspec_kw={"wspace": 0.5})
g, lo, hi = band(y, q)
f, t, _ = roc_curve(y, q)
ax[0].fill_between(g, lo, hi, color=C[0], alpha=.15, lw=0, label="95% CI")
ax[0].plot(f, t, color=C[0], label=f"AUROC {roc_auc_score(y, q):.3f}")
ax[0].plot([0, 1], [0, 1], ls="--", color=INK2, lw=.6)
ax[0].set(xlabel="False positive rate", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01), title="(a) ROC")
ax[0].legend(loc="lower right")
pr, rc, _ = precision_recall_curve(y, q)
ax[1].plot(rc, pr, color=C[0], label=f"AUPRC {average_precision_score(y, q):.3f}", drawstyle="steps-post")
ax[1].axhline(y.mean(), color=INK2, lw=.6, ls=":", label=f"prevalence {y.mean():.3f}")
ax[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01), title="(b) Precision recall")
ax[1].legend(loc="upper right")
for c, (lab, p) in zip((C[1], C[0]), (("raw", p_raw), ("Platt", q))):
    o = np.argsort(p); bins = np.array_split(o, 8)
    e, _ = P.ece_mce(y, p)
    ax[2].plot([p[b].mean() for b in bins], [y[b].mean() for b in bins], "o-", ms=2.8, color=c, label=f"{lab}, ECE {e:.3f}")
ax[2].plot([0, 1], [0, 1], ls="--", color=INK2, lw=.6)
ax[2].set(xlabel="Predicted probability", ylabel="Observed frequency", xlim=(0, .7), ylim=(0, .7), title="(c) Reliability")
ax[2].legend(loc="upper left")
th = np.linspace(0.01, 0.6, 120)
prev = y.mean()
ax[3].plot(th, P.net_benefit(y, q, th), color=C[0], label="model")
ax[3].plot(th, prev - (1 - prev) * th / (1 - th), color=INK2, ls="--", lw=.8, label="refer all")
ax[3].axhline(0, color=INK2, ls=":", lw=.8, label="refer none")
ax[3].set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(-0.01, 0.1), title="(d) Decision curve")
ax[3].legend(loc="upper right")
save(fig, "fig_internal")

# ---------------------------------------------------------------- Fig. 5 uncertainty and deferral
corr = ((q >= THR_F1).astype(int) == y); err = ~corr
fig, ax = plt.subplots(1, 4, figsize=(W2, 1.85), gridspec_kw={"wspace": 0.6, "width_ratios": [0.8, 1, 1, 1]})
rng = np.random.default_rng(0)
u = ti.epistemic.values
ax[0].boxplot([u[corr], u[err]], widths=.5, showfliers=False, medianprops=dict(color=INK, lw=1),
              boxprops=dict(color=INK2, lw=.7), whiskerprops=dict(color=INK2, lw=.7), capprops=dict(color=INK2, lw=.7))
ax[0].scatter(1 + rng.uniform(-.16, .16, corr.sum()), u[corr], s=3, color=C[0], alpha=.45, lw=0)
ax[0].scatter(2 + rng.uniform(-.12, .12, err.sum()), u[err], s=7, color=C[1], lw=.3, edgecolor="white", zorder=3)
ax[0].set_xticks([1, 2], [f"correct\n(n={corr.sum()})", f"error\n(n={err.sum()})"])
ax[0].set(ylabel="Epistemic uncertainty (nats)", title="(a) Error vs correct")
for c, k, lab in ((C[0], "epistemic", "epistemic"), (C[1], "aleatoric", "aleatoric"), (C[2], "total", "total")):
    cov, risk, aurc, _, _ = P.risk_coverage(err, ti[k].values)
    ax[1].plot(cov, risk * 100, color=c, label=f"{lab} {aurc:.4f}")
ax[1].set(xlabel="Coverage (fraction retained)", ylabel="Error among retained (%)", xlim=(0.3, 1), ylim=(0, 11),
          title="(b) Risk coverage")
ax[1].legend(loc="upper left", title="AURC", title_fontsize=6)
cf = pd.read_csv(f"{RES}/tables/S1_conformal_test.csv")
nom = 1 - cf.alpha.values
ax[2].plot([.65, 1], [.65, 1], ls="--", color=INK2, lw=.6, label="nominal")
ax[2].plot(nom, cf.coverage_class0, "o-", ms=3, color=C[0], label="normal Z line")
ax[2].plot(nom, cf.coverage_class1, "s-", ms=3, color=C[1], label="BE")
ax[2].set(xlabel="Nominal coverage (1 − α)", ylabel="Empirical coverage", xlim=(.66, .97), ylim=(.65, 1.01),
          title="(c) Coverage by class")
ax[2].legend(loc="upper left")
xs = np.arange(len(cf)); bot = np.zeros(len(cf))
for col, lab, c in (("singleton", "single label", C[0]), ("ambiguous", "both labels (defer)", C[1]), ("empty", "empty (defer)", C[2])):
    ax[3].bar(xs, cf[col], .62, bottom=bot, color=c, label=lab, edgecolor="white", lw=.6); bot += cf[col].values
ax[3].set_xticks(xs, [f"{v:.2f}".lstrip("0") for v in nom]); ax[3].set_ylim(0, 1)
ax[3].set(xlabel="Nominal coverage (1 − α)", ylabel="Fraction of images", title="(d) Set composition")
ax[3].legend(loc="lower right", frameon=True, framealpha=.92, edgecolor=GRID, fontsize=5.6)
ax[3].grid(axis="x", visible=False)
save(fig, "fig_uncertainty")

# ---------------------------------------------------------------- Fig. 7 external validation and shift
be = ex[ex.cohort == "barretts"]
names = {"zline": "normal Z line", "esophagitis": "esophagitis", "normal_esophagus": "normal esophagus"}
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.05), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1.45, 1]})
for c, k in zip(C, names):
    ng = ex[ex.cohort == k]; yy = np.r_[np.ones(len(be)), np.zeros(len(ng))]; pp = np.r_[be.p_cal, ng.p_cal]
    f, t, _ = roc_curve(yy, pp)
    if k == "zline":
        g, lo, hi = band(yy, pp, 600); ax[0].fill_between(g, lo, hi, color=c, alpha=.15, lw=0)
    ax[0].plot(f, t, color=c, label=f"{names[k]}  {roc_auc_score(yy, pp):.3f}")
ax[0].plot([0, 1], [0, 1], ls="--", color=INK2, lw=.6)
ax[0].set(xlabel="False positive rate", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01), title="(a) External ROC")
ax[0].legend(loc="lower right", fontsize=5.6, title="BE vs (AUROC)", title_fontsize=5.6, frameon=True, framealpha=0.95, edgecolor=GRID, facecolor="white")
groups = [("Internal\ntest", ti.epistemic.values, INK2), ("Z line", ex[ex.cohort == "zline"].epistemic.values, C[0]),
          ("Esophagitis", ex[ex.cohort == "esophagitis"].epistemic.values, C[1]),
          ("Normal\nesophagus", ex[ex.cohort == "normal_esophagus"].epistemic.values, C[2]),
          ("BE", be.epistemic.values, C[3])]
ax[1].boxplot([g_[1] for g_ in groups], widths=.5, showfliers=False, medianprops=dict(color=INK, lw=1),
              boxprops=dict(color=INK2, lw=.7), whiskerprops=dict(color=INK2, lw=.7), capprops=dict(color=INK2, lw=.7))
for i, (lab, v, c) in enumerate(groups):
    ax[1].scatter(i + 1 + rng.uniform(-.17, .17, len(v)), v, s=2.2, color=c, alpha=.4, lw=0)
    ax[1].text(i + 1, -0.012, f"{np.median(v):.3f}", ha="center", fontsize=5.8, color=INK)
ax[1].set_xticks(range(1, 6), [g_[0].replace("\n", " ") for g_ in groups], fontsize=5.8, rotation=28, ha="right", rotation_mode="anchor")
ax[1].annotate("", xy=(5.4, 0.133), xytext=(1.6, 0.133), arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
ax[1].text(3.5, 0.136, "GastroVision", ha="center", fontsize=5.8, color=INK2)
ax[1].set_ylim(-0.02, 0.15)
ax[1].set(ylabel="Epistemic uncertainty (nats)", title="(b) Epistemic uncertainty by cohort")
yy = np.r_[np.ones(len(be)), np.zeros((ex.cohort == "zline").sum())]
pp = np.r_[be.p_cal, ex[ex.cohort == "zline"].p_cal]; uu = np.r_[be.epistemic, ex[ex.cohort == "zline"].epistemic]
e2 = ((pp >= THR_F1).astype(int) != yy)
cov, risk, aurc, _, r0 = P.risk_coverage(e2, uu)
ax[2].plot(cov, risk * 100, color=C[0], label=f"epistemic (AURC {aurc:.3f})")
ax[2].axhline(r0 * 100, color=INK2, ls=":", lw=.8, label=f"no deferral ({r0 * 100:.1f}%)")
ax[2].set(xlabel="Coverage (fraction retained)", ylabel="Error among retained (%)", xlim=(0.3, 1), ylim=(0, 20),
          title="(c) Deferral, BE vs Z line")
ax[2].legend(loc="upper left")
save(fig, "fig_external")
print("figures written")
