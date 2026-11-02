"""
esca_sci_pipeline.py
====================

Analysis pipeline for the paper

    A. P. Mohod and K. P. Yadav, "Calibrated Deep Learning with Conformal Deferral for
    Barrett's Esophagus Recognition and Multimodal Characterization of Esophageal Cancer."

Every number in the paper is read from the files this module writes.

Stage 1   Barrett's esophagus versus normal Z line on white light endoscopy (HyperKvasir)
          * duplicate audit: each augmented copy is linked to its source image (identifier
            offset rule and mutual nearest neighbour rule) and every split is grouped
          * 14 multiscale views per image encoded by a frozen DINOv2 ViT-B/14
          * five view aggregators (mean, max, gated attention, TransMIL variant, scale aware
            gated attention) compared out of fold; the winner is chosen by out of fold AUPRC
            and never by the test set (mean pooling in the reported run)
          * cross fitted ensemble of 5 seeds x 5 grouped folds with EMA weights
          * calibrator chosen by cross validated NLL (temperature, Platt or beta)
          * uncertainty decomposition, selective prediction (AURC, E-AURC), class conditional
            (Mondrian) conformal prediction, decision curve analysis, DeLong tests

External  GastroVision scored by the same ensemble with the calibrator, thresholds and
          conformal quantiles frozen; cross dataset near duplicate audit; Mahalanobis shift
          score; optional local recalibration reported separately

Stage 2   CT, PET and genomic embeddings fused by masked multihead self attention, evaluated by
          repeated nested cross validation (each patient is scored only by models that never saw
          that patient) with an ablation that uses genomics only. The stage head is trained as an
          auxiliary task; the missingness audit (scripts/missingness_audit.py) explains why stage
          is not reported as an endpoint.

The code is written for a Kaggle GPU session: bags are kept on the GPU, no DataLoader workers
are used, and mixed precision is on. Figures are saved as 300 dpi PNG and vector PDF, and tables
as CSV and LaTeX.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 1. Configuration
# =============================================================================
@dataclass
class Config:
    # ---- data -------------------------------------------------------------------------
    data_root: str = "/kaggle/input/datasets/bhavanamohod/final-mm-features/mm-dataset"
    manifest: str = "multimodal_master_dataset_pruned.csv"
    endo_instance: str = "endo_instance_feat.parquet"
    rad_feat: str = "rad_sequence_feat.parquet"
    pet_feat: str = "pet_sequence_feat.parquet"
    geno_feat: str = "genomics_snn_feat.parquet"
    out_dir: str = "/kaggle/working/esca_sci"
    id_col: str = "Patient_ID"
    label_col: str = "Unified_Diagnosis_Merged"
    pos_label: str = "Barrett's Esophagus"
    neg_label: str = "Normal"
    label_from_id: Dict[str, str] = field(default_factory=lambda: {"positive": r"^EN-BE", "negative": r"^EN-NZ"})

    # ---- grouping (leakage control) -----------------------------------------------------
    group_auto_offset: bool = True        # detect ID-offset duplicates (e.g. EN-BE-00001 <-> EN-BE-00095)
    group_sim_threshold: float = 0.90     # mutual-nearest-neighbour cosine (centred bag embeddings)
    eval_one_per_group: bool = True       # score one record per source image in test / calibration

    # ---- protocol -----------------------------------------------------------------------
    seeds: Tuple[int, ...] = (42, 123, 223, 323, 423)
    n_folds: int = 5
    test_frac: float = 0.20
    split_seed: int = 2026

    # ---- model / training ---------------------------------------------------------------
    aggregators: Tuple[str, ...] = ("mean", "max", "gated", "transmil", "gated_sa")
    final_aggregator: str = "auto"        # 'auto' = best out-of-fold AUPRC (never the test set)
    d_hidden: int = 256
    d_att: int = 128
    epochs: int = 40
    batch: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    label_smoothing: float = 0.05
    ema_decay: float = 0.99
    inst_drop: float = 0.10               # embedding-space bag augmentation
    feat_noise: float = 0.05
    mc_samples: int = 50
    amp: bool = True

    # ---- evaluation ---------------------------------------------------------------------
    bootstrap_n: int = 2000
    ece_bins: int = 15
    alphas: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30)
    target_sensitivity: float = 0.95
    abstain_coverage: float = 0.90

    # ---- external (GastroVision) --------------------------------------------------------
    gv_root: str = "/kaggle/input/datasets/orvile/gastrovision-gastrointestinal-disease-detection/Gastrovision"
    gv_feature_file: str = "/kaggle/working/esca_sci/external/gastrovision_instances.parquet"
    gv_folders: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "barretts": ("barrett",),
        "zline": ("z-line", "z line", "zline"),
        "normal_esophagus": ("normal esophagus", "normal oesophagus"),
        "esophagitis": ("esophagitis", "oesophagitis"),
    })
    phash_hamming: int = 0                # 0 = exact perceptual-hash duplicates only
    leakage_sim: float = 0.98             # external bag flagged if cosine to any internal bag >= this
    local_recal_frac: float = 0.30
    local_recal_reps: int = 200

    # ---- Stage 2 ------------------------------------------------------------------------
    s2_require_genomics: bool = False     # False keeps every SCC / adenocarcinoma patient (reported setting)
    s2_id_prefix: Optional[str] = None    # e.g. "TCGA" to restrict; None keeps TCGA + VAREPOP
    s2_drop_unknown_stage: bool = False   # False counts an unrecorded stage as I-II (needed only for the auxiliary head)
    s2_stage_col: str = "AJCC_Pathologic_Stage"
    s2_ensemble_seeds: int = 3            # models averaged inside every outer fold
    s2_outer_folds: int = 5
    s2_repeats: int = 3
    s2_epochs: int = 60
    s2_lr: float = 3e-4
    s2_weight_decay: float = 5e-2
    s2_modality_drop: float = 0.15

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def path(self, name: str) -> str:
        return name if os.path.isabs(name) else os.path.join(self.data_root, name)

    def quick(self) -> "Config":
        """Small settings for a smoke test (minutes, not hours). Never use for reported numbers."""
        self.seeds, self.n_folds, self.epochs, self.mc_samples = (42, 123), 3, 6, 10
        self.bootstrap_n, self.local_recal_reps, self.s2_repeats, self.s2_epochs = 200, 20, 1, 8
        self.s2_ensemble_seeds = 1
        return self

    def resolve_paths(self) -> "Config":
        """Find the dataset folders under /kaggle/input if the configured paths do not exist."""
        if not os.path.exists(self.path(self.manifest)) and os.path.isdir("/kaggle/input"):
            for d, _, files in os.walk("/kaggle/input"):
                if self.manifest in files:
                    self.data_root = d
                    break
        if not os.path.isdir(self.gv_root) and os.path.isdir("/kaggle/input"):
            for d, dirs, _ in os.walk("/kaggle/input"):
                if any("barrett" in x.lower() for x in dirs):
                    self.gv_root = d
                    break
        log(f"data_root = {self.data_root}\ngv_root   = {self.gv_root}")
        return self


CFG = Config()


# =============================================================================
# 2. Utilities, logging, plotting style
# =============================================================================
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: Config) -> Dict[str, str]:
    d = {k: os.path.join(cfg.out_dir, k) for k in ("figures", "tables", "predictions", "external", "stage2")}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#7d55c7")   # categorical slots 1-4 (fixed order)


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.6, "savefig.dpi": 300,
        "figure.dpi": 110, "pdf.fonttype": 42,
    })


def save_fig(fig, dirs: Dict[str, str], name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(dirs["figures"], f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, dirs: Dict[str, str], name: str, sub: str = "tables") -> None:
    df.to_csv(os.path.join(dirs[sub], f"{name}.csv"), index=False)
    try:
        with open(os.path.join(dirs[sub], f"{name}.tex"), "w") as fh:
            fh.write(df.to_latex(index=False, float_format="%.4f", escape=True))
    except Exception:
        pass


# =============================================================================
# 3. Data loading
# =============================================================================
def _to_bag(v, d: int = 768) -> np.ndarray:
    try:
        a = np.stack([np.asarray(x, dtype=np.float32) for x in v])
    except Exception:
        flat: List[float] = []

        def _walk(x):
            if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
                for e in x:
                    _walk(e)
            else:
                flat.append(float(x))
        _walk(v)
        a = np.asarray(flat, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, d)
    return a.astype(np.float32)


def load_instance_table(path: str, id_col: str = "Patient_ID") -> Tuple[np.ndarray, List[np.ndarray], pd.DataFrame]:
    df = pd.read_parquet(path)
    feat_col = next(c for c in df.columns if "inst" in c.lower())
    ids = df[id_col].astype(str).values
    bags = [_to_bag(v) for v in df[feat_col].values]
    return ids, bags, df.drop(columns=[feat_col])


def pad_bags(bags: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    k = max(b.shape[0] for b in bags)
    d = bags[0].shape[1]
    X = np.zeros((len(bags), k, d), dtype=np.float32)
    M = np.zeros((len(bags), k), dtype=bool)
    for i, b in enumerate(bags):
        X[i, : len(b)] = b
        M[i, : len(b)] = True
    return X, M


def load_stage1(cfg: Config) -> Dict[str, object]:
    ids, bags, _ = load_instance_table(cfg.path(cfg.endo_instance), cfg.id_col)
    man = pd.read_csv(cfg.path(cfg.manifest), low_memory=False)
    lab = {}
    if cfg.label_col in man.columns:
        sub = man[man[cfg.label_col].isin([cfg.pos_label, cfg.neg_label])]
        lab = dict(zip(sub[cfg.id_col].astype(str), (sub[cfg.label_col] == cfg.pos_label).astype(int)))
    y = []
    for pid in ids:
        if pid in lab:
            y.append(lab[pid])
        elif re.search(cfg.label_from_id["positive"], pid):
            y.append(1)
        elif re.search(cfg.label_from_id["negative"], pid):
            y.append(0)
        else:
            y.append(-1)
    y = np.asarray(y)
    keep = (y >= 0) & np.array([np.any(b) for b in bags])
    n_zero = int(np.sum((y >= 0) & ~np.array([np.any(b) for b in bags])))
    ids, y = ids[keep], y[keep]
    bags = [b for b, k in zip(bags, keep) if k]
    X, M = pad_bags(bags)
    log(f"Stage 1: {len(ids)} records ({y.sum()} BE, {len(y) - y.sum()} normal); {n_zero} all-zero bags excluded; "
        f"bag size {X.shape[1]} x {X.shape[2]}")
    return {"ids": ids, "y": y, "X": X, "M": M, "n_zero": n_zero}


# =============================================================================
# 4. Source-image grouping (leakage control)
# =============================================================================
def bag_embedding(X: np.ndarray, M: np.ndarray, center: Optional[np.ndarray] = None) -> np.ndarray:
    """Mean-pooled bag embedding, optionally centred, L2-normalised (float32, chunked to save RAM)."""
    out = np.zeros((len(X), X.shape[2]), dtype=np.float32)
    for s in range(0, len(X), 1024):
        x, m = X[s:s + 1024].astype(np.float32), M[s:s + 1024, :, None]
        out[s:s + 1024] = (x * m).sum(1) / np.clip(m.sum(1), 1, None)
    if center is not None:
        out = out - center
    return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-8, None)


def embedding_center(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    e = bag_embedding(X, M)
    return e.mean(0)


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _nearest_neighbours(E: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    T = torch.from_numpy(E).to(device)
    nn_idx = np.zeros(len(E), dtype=int)
    nn_sim = np.zeros(len(E), dtype=np.float32)
    for s in range(0, len(E), 2048):
        S = T[s:s + 2048] @ T.T
        S[torch.arange(S.shape[0]), torch.arange(s, s + S.shape[0])] = -2
        v, i = S.max(1)
        nn_idx[s:s + len(v)] = i.cpu().numpy()
        nn_sim[s:s + len(v)] = v.float().cpu().numpy()
    return nn_idx, nn_sim


def discover_groups(ids: np.ndarray, X: np.ndarray, M: np.ndarray, y: np.ndarray, cfg: Config) -> Tuple[np.ndarray, Dict]:
    """Group records that come from the same source image.

    Rule 1 (ID offset): augmented copies were written with a constant numeric offset
    (EN-BE n <-> n+94, EN-NZ n <-> n+932). For every ID prefix the half-range offset is
    tested and accepted only if paired embeddings are far more similar than random pairs.
    Rule 2 (content): mutual nearest neighbours with centred cosine >= threshold and the
    same label are also merged. Union-find joins both rules.
    """
    E = bag_embedding(X, M, embedding_center(X, M))
    uf = _UnionFind(len(ids))
    report = {"offset_pairs": 0, "nn_pairs": 0, "offset_rules": []}
    rng = np.random.default_rng(0)

    if cfg.group_auto_offset:
        prefixes: Dict[str, Dict[int, int]] = {}
        for i, sid in enumerate(ids):
            m = re.match(r"^(.*?)(\d+)$", sid)
            if m:
                prefixes.setdefault(m.group(1), {})[int(m.group(2))] = i
        for pre, num2idx in prefixes.items():
            if len(num2idx) < 8:
                continue
            nums = np.array(sorted(num2idx))
            span = int(nums.max() - nums.min() + 1)
            cands = sorted({span // 2, (span + 1) // 2, len(nums) // 2})
            all_idx = np.array(list(num2idx.values()))
            idx2num = {v: n for n, v in num2idx.items()}
            ra, rb = rng.choice(all_idx, 4000), rng.choice(all_idx, 4000)
            diff = np.array([abs(idx2num[i] - idx2num[j]) for i, j in zip(ra, rb)])
            ok_r = (ra != rb) & ~np.isin(diff, cands)          # exclude self-pairs and candidate offsets
            sim_rand = np.sum(E[ra[ok_r]] * E[rb[ok_r]], 1)
            q99 = float(np.quantile(sim_rand, 0.99))
            best = None
            for k in cands:
                if k <= 0:
                    continue
                pa = [(num2idx[n], num2idx[n + k]) for n in nums if n + k in num2idx and n < nums.min() + k]
                if len(pa) < 4:
                    continue
                a, b = np.array(pa).T
                sim_pair = np.sum(E[a] * E[b], 1)
                med = float(np.median(sim_pair))
                if best is None or med > best[1]:
                    best = (k, med, a, b, sim_pair)
            if best is None:
                continue
            k, med, a, b, sim_pair = best
            if med > q99:
                cut = 0.5 * (med + q99)
                ok = sim_pair >= cut
                for i, j in zip(a[ok], b[ok]):
                    uf.union(int(i), int(j))
                report["offset_pairs"] += int(ok.sum())
                report["offset_rules"].append({"prefix": pre, "offset": int(k), "pairs_tested": int(len(a)),
                                               "pairs_accepted": int(ok.sum()), "median_pair_sim": med,
                                               "q99_random_sim": q99})
            log(f"  offset test {pre!r}: offset {k}, median paired cosine {med:.3f} vs random q99 {q99:.3f} -> "
                f"{'accepted' if med > q99 else 'rejected'}")

    nn_idx, nn_sim = _nearest_neighbours(E.astype(np.float32), cfg.device)
    for i, j in enumerate(nn_idx):
        if nn_idx[j] == i and nn_sim[i] >= cfg.group_sim_threshold and y[i] == y[j]:
            uf.union(i, int(j))
            report["nn_pairs"] += 1
    report["nn_pairs"] //= 2
    roots = np.array([uf.find(i) for i in range(len(ids))])
    _, groups = np.unique(roots, return_inverse=True)
    sizes = np.bincount(groups)
    report.update({"n_records": int(len(ids)), "n_groups": int(groups.max() + 1),
                   "group_size_counts": {int(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))},
                   "nn_sim_median": float(np.median(nn_sim))})
    log(f"Grouping: {report['n_records']} records -> {report['n_groups']} source groups "
        f"(offset pairs {report['offset_pairs']}, NN pairs {report['nn_pairs']})")
    return groups, {**report, "nn_sim": nn_sim}


def representatives(groups: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """One record per group (lowest index) among idx."""
    seen, out = set(), []
    for i in sorted(idx):
        g = groups[i]
        if g not in seen:
            seen.add(g)
            out.append(i)
    return np.asarray(out)


def grouped_test_split(y: np.ndarray, groups: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n_splits = max(2, int(round(1 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dev, test = next(sgkf.split(np.zeros(len(y)), y, groups))
    assert not set(groups[dev]) & set(groups[test]), "group leakage between dev and test"
    return dev, test


# =============================================================================
# 5. Models
# =============================================================================
DROP_RATES = tuple(np.linspace(0.1, 0.5, 5).round(2))


class BagNet(nn.Module):
    """MIL head over a bag of instance embeddings.

    agg: 'mean' | 'max' | 'gated' (Ilse et al.) | 'transmil' (self-attention + gated pooling)
         | 'gated_sa' (gated attention with learned scale-and-position tokens for the 14 views)
    """

    def __init__(self, agg: str = "gated", d_in: int = 768, d: int = 256, d_att: int = 128,
                 n_inst: int = 14, inst_drop: float = 0.1, feat_noise: float = 0.05):
        super().__init__()
        self.agg, self.inst_drop, self.feat_noise = agg, inst_drop, feat_noise
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU())
        if agg == "gated_sa":
            self.pos = nn.Parameter(torch.randn(1, n_inst, d) * 0.02)
        if agg == "transmil":
            self.ctx_norm = nn.LayerNorm(d)
            self.mha = nn.MultiheadAttention(d, 8, dropout=0.1, batch_first=True)
        if agg in ("gated", "transmil", "gated_sa"):
            self.V, self.U, self.w = nn.Linear(d, d_att), nn.Linear(d, d_att), nn.Linear(d_att, 1)
        self.classifier = nn.Linear(d, 2)

    def embed(self, x: torch.Tensor, m: torch.Tensor):
        if self.training and self.feat_noise > 0:
            x = x + self.feat_noise * x.detach().std() * torch.randn_like(x)
        h = self.proj(x)
        if self.agg == "gated_sa":
            h = h + self.pos[:, : h.shape[1]]
        if self.training and self.inst_drop > 0:
            keep = (torch.rand(m.shape, device=m.device) > self.inst_drop) & m
            keep[keep.sum(1) == 0] = m[keep.sum(1) == 0]
            m = keep
        mf = m.float().unsqueeze(-1)
        if self.agg == "mean":
            z = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
            return z, m.float() / m.float().sum(1, keepdim=True)
        if self.agg == "max":
            z = h.masked_fill(~m.unsqueeze(-1), -1e4).max(1)[0]
            return z, m.float() / m.float().sum(1, keepdim=True)
        if self.agg == "transmil":
            hn = self.ctx_norm(h)
            ctx, _ = self.mha(hn, hn, hn, key_padding_mask=~m)
            h = h + ctx
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = torch.softmax(a.float().masked_fill(~m, float("-inf")), 1)
        z = torch.bmm(a.unsqueeze(1).to(h.dtype), h).squeeze(1)
        return z, a

    def forward(self, x, m):
        z, a = self.embed(x, m)
        if self.training:      # multi-sample dropout: average logits over five dropout rates
            logits = torch.stack([self.classifier(F.dropout(z, p, True)) for p in DROP_RATES]).mean(0)
        else:
            logits = self.classifier(z)
        return logits, a

    @torch.no_grad()
    def mc(self, x, m, T: int) -> torch.Tensor:
        z, _ = self.embed(x, m)
        out = [torch.softmax(self.classifier(F.dropout(z, DROP_RATES[t % len(DROP_RATES)], True)).float(), 1)[:, 1]
               for t in range(T)]
        return torch.stack(out)            # (T, n)


class _EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)


def _class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    c = np.bincount(y, minlength=2).astype(np.float64)
    w = 1.0 / np.sqrt(c + 1e-5)
    return torch.tensor(w / w.sum(), dtype=torch.float32, device=device)


def train_member(X: torch.Tensor, M: torch.Tensor, y: np.ndarray, agg: str, seed: int, cfg: Config) -> BagNet:
    seed_everything(seed)
    dev = cfg.device
    model = BagNet(agg, X.shape[2], cfg.d_hidden, cfg.d_att, X.shape[1], cfg.inst_drop, cfg.feat_noise).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(y) / cfg.batch)
    total, warm = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    crit = nn.CrossEntropyLoss(weight=_class_weights(y, dev), label_smoothing=cfg.label_smoothing)
    Y = torch.from_numpy(y).long().to(dev)
    ema = _EMA(model, cfg.ema_decay)
    use_amp = cfg.amp and dev == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(y), device=dev)
        for s in range(0, len(y), cfg.batch):
            b = perm[s:s + cfg.batch]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, _ = model(X[b].float(), M[b])
                loss = crit(logits.float(), Y[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
    ema.apply(model)
    model.eval()
    return model


@torch.no_grad()
def predict(models: List[BagNet], X: torch.Tensor, M: torch.Tensor, bs: int = 2048) -> np.ndarray:
    out = np.zeros(len(X))
    for m in models:
        ps = []
        for s in range(0, len(X), bs):
            logits, _ = m(X[s:s + bs].float(), M[s:s + bs])
            ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        out += np.concatenate(ps)
    return out / len(models)


@torch.no_grad()
def predict_mc(models: List[BagNet], X: torch.Tensor, M: torch.Tensor, T: int, bs: int = 2048) -> np.ndarray:
    res = []
    for m in models:
        res.append(np.concatenate([m.mc(X[s:s + bs].float(), M[s:s + bs], T).cpu().numpy()
                                   for s in range(0, len(X), bs)], axis=1))
    return np.stack(res)                    # (M, T, n)


@torch.no_grad()
def attention_weights(models: List[BagNet], X: torch.Tensor, M: torch.Tensor) -> np.ndarray:
    acc = None
    for m in models:
        _, a = m(X.float(), M)
        acc = a.float().cpu().numpy() if acc is None else acc + a.float().cpu().numpy()
    return acc / len(models)


# =============================================================================
# 6. Calibration
# =============================================================================
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _nll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Calibrator:
    METHODS = ("temperature", "platt", "beta")

    def __init__(self, method: str = "auto"):
        self.method, self.cv_nll, self.params = method, {}, None

    @staticmethod
    def _fit(method, y, p):
        if method == "temperature":
            z = _logit(p)
            r = minimize_scalar(lambda t: _nll(y, 1 / (1 + np.exp(-z / t))), bounds=(0.05, 20), method="bounded")
            return ("temperature", r.x)
        feats = _logit(p)[:, None] if method == "platt" else np.c_[np.log(np.clip(p, 1e-6, 1)), -np.log(np.clip(1 - p, 1e-6, 1))]
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(feats, y)
        return (method, lr)

    @staticmethod
    def _apply(params, p):
        method, obj = params
        if method == "temperature":
            return 1 / (1 + np.exp(-_logit(p) / obj))
        feats = _logit(p)[:, None] if method == "platt" else np.c_[np.log(np.clip(p, 1e-6, 1)), -np.log(np.clip(1 - p, 1e-6, 1))]
        return obj.predict_proba(feats)[:, 1]

    def fit(self, y: np.ndarray, p: np.ndarray, seed: int = 0) -> "Calibrator":
        methods = self.METHODS if self.method == "auto" else (self.method,)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for mth in methods:
            q = np.zeros_like(p)
            for tr, va in skf.split(p, y):
                q[va] = self._apply(self._fit(mth, y[tr], p[tr]), p[va])
            self.cv_nll[mth] = _nll(y, q)
        self.method = min(self.cv_nll, key=self.cv_nll.get)
        self.params = self._fit(self.method, y, p)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self._apply(self.params, p)


# =============================================================================
# 7. Metrics and statistics
# =============================================================================
def ece_mce(y, p, n_bins=15):
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    gaps = [abs(y[b].mean() - p[b].mean()) for b in bins if len(b)]
    w = [len(b) / len(p) for b in bins if len(b)]
    return float(np.sum(np.array(gaps) * np.array(w))), float(np.max(gaps))


def at_threshold(y, p, thr) -> Dict[str, float]:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
    mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0
    return dict(threshold=float(thr), TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn), sensitivity=sens,
                specificity=spec, PPV=ppv, NPV=npv, F1=f1, MCC=mcc, accuracy=(tp + tn) / len(y))


def operating_points(y, p, target_sens=0.95) -> Dict[str, float]:
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0, 1)
    youden = thr[np.argmax(tpr - fpr)]
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    f1opt = th[np.nanargmax(f1)]
    prev = np.quantile(p, 1 - y.mean())
    ok = np.where(tpr >= target_sens)[0]
    sens_t = thr[ok[0]] if len(ok) else thr[-1]
    return {"Youden": float(youden), "F1-optimal": float(f1opt), "Prevalence-matched": float(prev),
            f"Sensitivity>={int(target_sens * 100)}%": float(sens_t)}


def summary_metrics(y, p) -> Dict[str, float]:
    e, m = ece_mce(y, p)
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p), "Brier": brier_score_loss(y, p),
            "ECE": e, "MCE": m, "prevalence": float(y.mean()), "n": int(len(y))}


def bootstrap_ci(y, p, fn, n=2000, seed=0, alpha=0.05) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        try:
            vals.append(fn(y[idx], p[idx]))
        except ValueError:
            continue
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(fn(y, p)), float(lo), float(hi)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_test(y, p1, p2) -> Tuple[float, float, float]:
    """Paired DeLong test (Sun & Xu fast algorithm). Returns (auc1, auc2, two-sided p)."""
    order = np.argsort(-y)
    y, preds = y[order], np.vstack([p1, p2])[:, order]
    m = int(y.sum())
    n = len(y) - m
    aucs, v01, v10 = [], [], []
    for r in preds:
        tx, ty, tz = _midrank(r[:m]), _midrank(r[m:]), _midrank(r)
        aucs.append(tz[:m].sum() / m / n - (m + 1) / 2 / n)
        v01.append((tz[:m] - tx) / n)
        v10.append(1 - (tz[m:] - ty) / m)
    sx, sy = np.cov(np.array(v01)), np.cov(np.array(v10))
    cov = sx / m + sy / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / math.sqrt(max(var, 1e-12))
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


def hanley_mcneil(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    se = math.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg))
    return max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se)


_trapz = getattr(np, "trapezoid", None) or np.trapz


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def decompose_uncertainty(P: np.ndarray) -> Dict[str, np.ndarray]:
    """P: (M, T, n) probabilities of class 1."""
    pbar = P.mean((0, 1))
    pm = P.mean(1)
    total = entropy(pbar)
    alea = entropy(P).mean((0, 1))
    epi = np.maximum(total - alea, 0)
    epi_member = np.maximum(total - entropy(pm).mean(0), 0)
    return {"total": total, "aleatoric": alea, "epistemic": epi, "epistemic_between": epi_member,
            "epistemic_dropout": np.maximum(epi - epi_member, 0),
            "ensemble_sd": pm.std(0), "mc_sd": P.reshape(-1, P.shape[2]).std(0), "p_mean": pbar}


def risk_coverage(err, unc, grid=None):
    order = np.argsort(unc, kind="stable")
    e = err[order].astype(float)
    cum = np.cumsum(e) / np.arange(1, len(e) + 1)
    cov = np.arange(1, len(e) + 1) / len(e)
    aurc = float(_trapz(cum, cov))
    r = e.mean()
    opt = np.cumsum(np.sort(e)) / np.arange(1, len(e) + 1)
    e_aurc = aurc - float(_trapz(opt, cov))
    return cov, cum, aurc, e_aurc, r


def conformal_fit(y, p, alpha):
    q = {}
    for c in (0, 1):
        s = 1 - (p[y == 1] if c == 1 else 1 - p[y == 0])
        n = len(s)
        k = min(n, int(math.ceil((n + 1) * (1 - alpha))))
        q[c] = float(np.sort(s)[k - 1]) if n else 1.0
    return q


def conformal_sets(p, q):
    in0 = (1 - (1 - p)) <= q[0]
    in1 = (1 - p) <= q[1]
    return np.c_[in0, in1]


def conformal_report(y, sets):
    size = sets.sum(1)
    cov = sets[np.arange(len(y)), y]
    return {"marginal_coverage": cov.mean(), "coverage_class0": cov[y == 0].mean(), "coverage_class1": cov[y == 1].mean(),
            "mean_set_size": size.mean(), "singleton": (size == 1).mean(), "ambiguous": (size == 2).mean(),
            "empty": (size == 0).mean()}


def net_benefit(y, p, thresholds):
    n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def mahalanobis_scorer(E_ref: np.ndarray, n_comp: int = 64):
    pca = PCA(n_components=min(n_comp, E_ref.shape[1], len(E_ref) - 1), random_state=0).fit(E_ref)
    Z = pca.transform(E_ref)
    lw = LedoitWolf().fit(Z)
    prec, mu = lw.precision_, lw.location_

    def score(E):
        d = pca.transform(E) - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))
    return score


# =============================================================================
# 8. Plots
# =============================================================================
def _band(y, p, n=300, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 101)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    tprs = []
    for _ in range(n):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        f, t, _ = roc_curve(y[idx], p[idx])
        tprs.append(np.interp(grid, f, t))
    return grid, np.quantile(tprs, 0.025, 0), np.quantile(tprs, 0.975, 0)


def plot_roc_pr(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], dirs, name, title=""):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for c, (lab, (y, p)) in zip(SERIES, curves.items()):
        f, t, _ = roc_curve(y, p)
        g, lo, hi = _band(y, p)
        ax[0].fill_between(g, lo, hi, color=c, alpha=0.15, lw=0)
        ax[0].plot(f, t, color=c, label=f"{lab}  AUROC {roc_auc_score(y, p):.3f}")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, color=c, label=f"{lab}  AUPRC {average_precision_score(y, p):.3f}")
        ax[1].axhline(y.mean(), color=c, lw=0.8, ls=":")
    ax[0].plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    ax[0].set(xlabel="1 - specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01), title="(a) ROC")
    ax[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01), title="(b) Precision-recall")
    for a in ax:
        a.legend(loc="lower right" if a is ax[0] else "lower left", fontsize=7.5)
    if title:
        fig.suptitle(title, fontsize=9)
    save_fig(fig, dirs, name)


def plot_reliability(y, p_raw, p_cal, dirs, name, n_bins=10):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
    for c, (lab, p) in zip(SERIES, (("uncalibrated", p_raw), ("calibrated", p_cal))):
        order = np.argsort(p)
        bins = np.array_split(order, n_bins)
        mp = [p[b].mean() for b in bins]
        fy = [y[b].mean() for b in bins]
        e, _ = ece_mce(y, p)
        ax[0].plot(mp, fy, "o-", color=c, ms=4, label=f"{lab} (ECE {e:.3f})")
    ax[0].plot([0, 1], [0, 1], ls="--", color=INK2, lw=0.8)
    ax[0].set(xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1),
              title="(a) Reliability (equal-mass bins)")
    ax[0].legend(loc="upper left", fontsize=7.5)
    bins = np.linspace(0, 1, 26)
    ax[1].hist(p_cal[y == 0], bins, color=SERIES[0], alpha=0.75, label="normal")
    ax[1].hist(p_cal[y == 1], bins, color=SERIES[1], alpha=0.75, label="BE")
    ax[1].set(yscale="log", xlabel="Calibrated probability", ylabel="Count (log)", title="(b) Score distribution")
    ax[1].legend(fontsize=7.5)
    save_fig(fig, dirs, name)


def plot_confusions(y, p, ops: Dict[str, float], dirs, name):
    fig, axes = plt.subplots(1, len(ops), figsize=(1.85 * len(ops), 2.2))
    for a, (lab, t) in zip(np.atleast_1d(axes), ops.items()):
        r = at_threshold(y, p, t)
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        a.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else INK, fontsize=9)
        a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        a.set_title(f"{lab}\nsens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}", fontsize=7.5)
        a.grid(False)
    save_fig(fig, dirs, name)


def plot_uncertainty(unc: Dict[str, np.ndarray], correct: np.ndarray, dirs, name):
    keys = ("epistemic", "aleatoric", "total")
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for a, k in zip(axes, keys):
        u = unc[k]
        err = ~correct
        auc = roc_auc_score(err, u) if err.any() and correct.any() else np.nan
        a.boxplot([u[correct]] + ([u[err]] if err.sum() >= 5 else []), widths=0.45, showfliers=False,
                  medianprops=dict(color=INK), boxprops=dict(color=INK2), whiskerprops=dict(color=INK2),
                  capprops=dict(color=INK2))
        a.scatter(1 + rng.uniform(-0.15, 0.15, correct.sum()), u[correct], s=6, color=SERIES[0], alpha=0.45, lw=0)
        if err.any():
            a.scatter(2 + rng.uniform(-0.12, 0.12, err.sum()), u[err], s=14, color=SERIES[1], zorder=3,
                      edgecolor="white", lw=0.5)
        a.set_xticks([1, 2], [f"correct\n(n={correct.sum()})", f"error\n(n={err.sum()})"])
        a.set_xlim(0.5, 2.5)
        a.set_title(f"{k}\nerror-detection AUROC {auc:.3f}", fontsize=8)
    axes[0].set_ylabel("Uncertainty (nats)")
    save_fig(fig, dirs, name)


def plot_risk_coverage(rc: Dict[str, Tuple], dirs, name):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for c, (lab, (cov, risk, aurc, eaurc, _)) in zip(SERIES, rc.items()):
        ax.plot(cov, risk, color=c, label=f"{lab}  AURC {aurc:.4f}")
    ax.set(xlabel="Coverage", ylabel="Selective risk (error rate)", xlim=(0, 1), title="Risk-coverage")
    ax.legend(fontsize=7.5)
    save_fig(fig, dirs, name)


def plot_conformal(df: pd.DataFrame, dirs, name):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    nom = 1 - df["alpha"].values
    ax[0].plot(nom, df["coverage_class0"], "o-", color=SERIES[0], ms=5, label="normal")
    ax[0].plot(nom, df["coverage_class1"], "s-", color=SERIES[1], ms=5, label="BE")
    lo = min(0.6, float(np.nanmin(df[["coverage_class0", "coverage_class1"]].values)) - 0.02)
    ax[0].plot([lo, 1], [lo, 1], ls="--", color=INK2, lw=0.8, label="nominal")
    ax[0].set(xlabel="Nominal coverage (1 - alpha)", ylabel="Empirical coverage", ylim=(lo, 1.01),
              title="(a) Class-conditional coverage")
    ax[0].legend(fontsize=7.5, loc="lower right")
    xs = np.arange(len(df))
    bottom = np.zeros(len(df))
    for col, lab, c in (("singleton", "single label", SERIES[0]), ("ambiguous", "both labels (defer)", SERIES[1]),
                        ("empty", "empty (defer)", SERIES[2])):
        ax[1].bar(xs, df[col], 0.6, bottom=bottom, color=c, label=lab, edgecolor="white", lw=1)
        bottom += df[col].values
    ax[1].set_xticks(xs, [f"{v:.2f}" for v in nom])
    ax[1].set(xlabel="Nominal coverage (1 - alpha)", ylabel="Fraction of cases", ylim=(0, 1.0),
              title="(b) Prediction-set composition")
    ax[1].legend(fontsize=7, loc="lower left", framealpha=0.9, frameon=True, edgecolor=GRID)
    save_fig(fig, dirs, name)


def plot_dca(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], dirs, name, tmax=0.6):
    th = np.linspace(0.01, tmax, 120)
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    y0 = next(iter(curves.values()))[0]
    prev = y0.mean()
    ax.plot(th, prev - (1 - prev) * th / (1 - th), color=INK2, lw=0.9, ls="--", label="refer all")
    ax.axhline(0, color=INK2, lw=0.9, ls=":", label="refer none")
    for c, (lab, (y, p)) in zip(SERIES, curves.items()):
        ax.plot(th, net_benefit(y, p, th), color=c, label=lab)
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(-0.02, max(prev, 0.05) * 1.15),
           title="Decision curve")
    ax.legend(fontsize=7.5)
    save_fig(fig, dirs, name)


def plot_forest(df: pd.DataFrame, dirs, name, col="AUPRC"):
    fig, ax = plt.subplots(figsize=(4.0, 0.42 * len(df) + 0.8))
    yy = np.arange(len(df))[::-1]
    ax.errorbar(df[col], yy, xerr=[df[col] - df[f"{col}_lo"], df[f"{col}_hi"] - df[col]], fmt="o", color=SERIES[0],
                ecolor=INK2, capsize=3, ms=5)
    ax.set_yticks(yy, df["aggregator"])
    ax.set(xlabel=f"{col} (95% bootstrap CI)", title="Aggregator comparison (out-of-fold)")
    save_fig(fig, dirs, name)


def plot_group_diagnostic(nn_sim, thr, dirs, name):
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.hist(nn_sim, bins=60, color=SERIES[0], alpha=0.85)
    ax.axvline(thr, color=SERIES[1], lw=1.2, label=f"grouping threshold {thr:.2f}")
    ax.set(xlabel="Cosine similarity to nearest neighbour", ylabel="Records", title="Near-duplicate structure")
    ax.legend(fontsize=7.5)
    save_fig(fig, dirs, name)


def plot_attention_by_scale(att: np.ndarray, y: np.ndarray, dirs, name, agg: str = ""):
    """Per-view attention relative to uniform (1.0 = average share), drawn on the 14-view layout."""
    if att is None or att.shape[1] != 14:
        return
    rel = att * 14
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.2), gridspec_kw={"width_ratios": [1, 1, 1]})
    from matplotlib.colors import TwoSlopeNorm
    dev_ = max(0.05, float(np.max(np.abs(np.r_[rel[y == 0].mean(0), rel[y == 1].mean(0)] - 1))) * 1.1)
    norm = TwoSlopeNorm(vmin=max(0.0, 1 - dev_), vcenter=1, vmax=1 + dev_)
    cmap = plt.get_cmap("RdBu_r")
    for r, (lab, sel) in enumerate((("normal", y == 0), ("BE", y == 1))):
        mu = rel[sel].mean(0)
        grids = [mu[0:1].reshape(1, 1), mu[1:5].reshape(2, 2), mu[5:14].reshape(3, 3)]
        for c, (g, t) in enumerate(zip(grids, ("global", "2x2 quadrants", "3x3 tiles"))):
            a = axes[r, c]
            im = a.imshow(g, cmap=cmap, norm=norm)
            for (i, j), v in np.ndenumerate(g):
                a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=INK)
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            if r == 0:
                a.set_title(t, fontsize=8.5)
            if c == 0:
                a.set_ylabel(f"{lab} (n={int(sel.sum())})", fontsize=8.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("Attention relative to uniform", fontsize=8)
    fig.suptitle(f"Where the model looks ({agg} aggregator, internal test)", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(dirs["figures"], f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_shift_uncertainty(unc: Dict[str, np.ndarray], dirs, name):
    labs = list(unc)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(0.95 * len(labs) + 1.6, 2.9))
    cols = (INK2,) + SERIES
    ax.boxplot([unc[k] for k in labs], widths=0.5, showfliers=False, medianprops=dict(color=INK),
               boxprops=dict(color=INK2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for i, (k, c) in enumerate(zip(labs, cols)):
        ax.scatter(i + 1 + rng.uniform(-0.17, 0.17, len(unc[k])), unc[k], s=5, color=c, alpha=0.45, lw=0)
    ax.set_xticks(range(1, len(labs) + 1), [l.replace("GastroVision ", "GV\n").replace("_", " ") for l in labs],
                  fontsize=7.5)
    ax.set(ylabel="Epistemic uncertainty (nats)", title="Uncertainty under dataset shift")
    save_fig(fig, dirs, name)


def plot_shift(scores: Dict[str, np.ndarray], dirs, name):
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    allv = np.concatenate(list(scores.values()))
    grid = np.linspace(np.quantile(allv, 0.002), np.quantile(allv, 0.998), 300)
    cols = (INK2,) + SERIES
    for c, (lab, s) in zip(cols, scores.items()):
        if len(s) > 2 and np.std(s) > 0:
            k = stats.gaussian_kde(s)(grid)
            ax.fill_between(grid, k, color=c, alpha=0.12, lw=0)
            ax.plot(grid, k, color=c, label=f"{lab} (n={len(s)})")
    ax.set(xlabel="Mahalanobis distance to development data", ylabel="Density", title="Feature-space shift")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    ax.legend(fontsize=6.8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save_fig(fig, dirs, name)


def cross_fitted_ensemble(agg: str, Xg: torch.Tensor, Mg: torch.Tensor, y: np.ndarray, groups: np.ndarray,
                          dev: np.ndarray, test: np.ndarray, cfg: Config) -> Dict:
    """n_seeds x n_folds members; every dev record gets an out-of-fold score from each seed."""
    oof = np.zeros((len(cfg.seeds), len(dev)))
    members: List[BagNet] = []
    for si, seed in enumerate(cfg.seeds):
        sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=seed)
        for tr, va in sgkf.split(np.zeros(len(dev)), y[dev], groups[dev]):
            itr, iva = dev[tr], dev[va]
            m = train_member(Xg[itr], Mg[itr], y[itr], agg, seed * 100 + len(members), cfg)
            oof[si, va] = predict([m], Xg[iva], Mg[iva])
            members.append(m)
    return {"agg": agg, "members": members, "oof_prob": oof.mean(0), "oof_prob_by_seed": oof,
            "test_prob": predict(members, Xg[test], Mg[test])}


def score_ensemble(R: Dict, Xg, Mg, test, cfg: Config,
                   extra: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None) -> Dict:
    """MC-dropout draws, attention maps and external scores for an already trained ensemble."""
    members = R["members"]
    R["test_mc"] = predict_mc(members, Xg[test], Mg[test], cfg.mc_samples)
    R["test_att"] = attention_weights(members, Xg[test], Mg[test])
    if extra:
        R["extra_prob"] = {k: predict(members, Xe, Me) for k, (Xe, Me) in extra.items()}
        R["extra_mc"] = {k: predict_mc(members, Xe, Me, cfg.mc_samples) for k, (Xe, Me) in extra.items()}
    return R


def run_stage1(cfg: Config = CFG, external_bags: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None) -> Dict:
    set_style()
    dirs = ensure_dirs(cfg)
    t0 = time.time()
    D = load_stage1(cfg)
    ids, y, X, M = D["ids"], D["y"], D["X"], D["M"]
    groups, grep = discover_groups(ids, X, M, y, cfg)
    plot_group_diagnostic(grep.pop("nn_sim"), cfg.group_sim_threshold, dirs, "S1_group_diagnostic")
    dev, test = grouped_test_split(y, groups, cfg.test_frac, cfg.split_seed)
    dev_eval = representatives(groups, dev) if cfg.eval_one_per_group else dev
    test_eval = representatives(groups, test) if cfg.eval_one_per_group else test
    log(f"Split: dev {len(dev)} records / {len(np.unique(groups[dev]))} groups; "
        f"test {len(test)} records / {len(test_eval)} scored ({y[test_eval].sum()} BE)")

    Xg = torch.from_numpy(X).to(cfg.device, torch.float16 if cfg.device == "cuda" else torch.float32)
    Mg = torch.from_numpy(M).to(cfg.device)
    extra = None
    if external_bags:
        extra = {k: (torch.from_numpy(Xe).to(Xg.device, Xg.dtype), torch.from_numpy(Me).to(Mg.device))
                 for k, (Xe, Me) in external_bags.items()}

    pos_dev = {i: k for k, i in enumerate(dev)}
    dev_eval_pos = np.array([pos_dev[i] for i in dev_eval])
    test_pos = {i: k for k, i in enumerate(test)}
    test_eval_pos = np.array([test_pos[i] for i in test_eval])
    y_oof, y_test = y[dev_eval], y[test_eval]

    # ---- aggregator comparison (cross-fitted, out-of-fold, no test-set use) ----------------
    runs, rows = {}, []
    for agg in cfg.aggregators:
        tA = time.time()
        r = cross_fitted_ensemble(agg, Xg, Mg, y, groups, dev, test, cfg)
        runs[agg] = {k: v for k, v in r.items()}
        p = r["oof_prob"][dev_eval_pos]
        a, alo, ahi = bootstrap_ci(y_oof, p, roc_auc_score, cfg.bootstrap_n)
        b, blo, bhi = bootstrap_ci(y_oof, p, average_precision_score, cfg.bootstrap_n)
        rows.append(dict(aggregator=agg, AUROC=a, AUROC_lo=alo, AUROC_hi=ahi, AUPRC=b, AUPRC_lo=blo, AUPRC_hi=bhi,
                         seconds=time.time() - tA))
        log(f"  {agg:<9s} OOF AUROC {a:.4f} [{alo:.4f}, {ahi:.4f}]  AUPRC {b:.4f}  ({time.time() - tA:.0f}s)")
    agg_df = pd.DataFrame(rows)
    final = cfg.final_aggregator if cfg.final_aggregator != "auto" else agg_df.sort_values("AUPRC").iloc[-1]["aggregator"]
    ref = runs[final]["oof_prob"][dev_eval_pos]
    agg_df["DeLong_p_vs_final"] = [delong_test(y_oof, ref, runs[a]["oof_prob"][dev_eval_pos])[2] if a != final else np.nan
                                   for a in agg_df["aggregator"]]
    agg_df["test_AUROC_after_selection"] = [roc_auc_score(y_test, runs[a]["test_prob"][test_eval_pos])
                                            for a in agg_df["aggregator"]]
    agg_df["test_AUPRC_after_selection"] = [average_precision_score(y_test, runs[a]["test_prob"][test_eval_pos])
                                            for a in agg_df["aggregator"]]
    save_table(agg_df, dirs, "S1_aggregator_comparison_oof")
    plot_forest(agg_df, dirs, "S1_aggregator_forest")
    log(f"Final aggregator (selected on out-of-fold AUPRC): {final}")

    # ---- final model: reuse the selected ensemble (no retraining), add MC + external -------
    R = score_ensemble(runs[final], Xg, Mg, test, cfg, extra)
    att_aggs = [a for a in agg_df.sort_values("AUPRC", ascending=False)["aggregator"]
                if a in ("gated", "gated_sa", "transmil")]
    att_agg = final if final in ("gated", "gated_sa", "transmil") else (att_aggs[0] if att_aggs else None)
    att_test = attention_weights(runs[att_agg]["members"], Xg[test], Mg[test]) if att_agg else None
    for a in list(runs):
        if a != final:
            runs[a].pop("members", None)
    gc.collect()
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    torch.save({"aggregator": final, "config": asdict(cfg), "state_dicts": [m.state_dict() for m in R["members"]],
                "d_in": int(X.shape[2]), "n_inst": int(X.shape[1])},
               os.path.join(dirs["predictions"], "stage1_ensemble.pt"))
    p_oof = R["oof_prob"][dev_eval_pos]
    p_test = R["test_prob"][test_eval_pos]
    cal = Calibrator("auto").fit(y_oof, p_oof)
    q_oof, q_test = cal.transform(p_oof), cal.transform(p_test)
    log(f"Calibrator: {cal.method}  CV NLL {json.dumps({k: round(v, 5) for k, v in cal.cv_nll.items()})}")

    ops = operating_points(y_oof, q_oof, cfg.target_sensitivity)
    op_rows = [{"rule": k, **at_threshold(y_test, q_test, t)} for k, t in ops.items()]
    for r_ in op_rows:
        r_["sens_CI"] = wilson(r_["TP"], r_["TP"] + r_["FN"])
        r_["spec_CI"] = wilson(r_["TN"], r_["TN"] + r_["FP"])
    op_df = pd.DataFrame(op_rows)
    save_table(op_df, dirs, "S1_operating_points_test")

    summ = {}
    for lab, (yy, pp) in {"test_raw": (y_test, p_test), "test_calibrated": (y_test, q_test)}.items():
        s = summary_metrics(yy, pp)
        s["AUROC_CI"] = bootstrap_ci(yy, pp, roc_auc_score, cfg.bootstrap_n)[1:]
        s["AUPRC_CI"] = bootstrap_ci(yy, pp, average_precision_score, cfg.bootstrap_n)[1:]
        s["Brier_CI"] = bootstrap_ci(yy, pp, brier_score_loss, cfg.bootstrap_n)[1:]
        summ[lab] = s

    # uncertainty (on the scored test records)
    P = R["test_mc"][:, :, test_eval_pos]
    U = decompose_uncertainty(P)
    thr_f1 = ops["F1-optimal"]
    correct = (q_test >= thr_f1).astype(int) == y_test
    err = ~correct
    unc_rows = [{"measure": k, "error_detection_AUROC": (roc_auc_score(err, v) if err.any() and correct.any() else np.nan),
                 "n_errors": int(err.sum())} for k, v in U.items() if k != "p_mean"]
    save_table(pd.DataFrame(unc_rows), dirs, "S1_uncertainty_error_detection")
    rc = {k: risk_coverage(err, U[k]) for k in ("epistemic", "aleatoric", "total")}
    rc_df = pd.DataFrame([{"measure": k, "AURC": v[2], "E-AURC": v[3], "full_coverage_risk": v[4]} for k, v in rc.items()])
    save_table(rc_df, dirs, "S1_risk_coverage")

    # conformal (fitted on out-of-fold calibrated scores, one per group)
    conf_rows = []
    for a in cfg.alphas:
        qh = conformal_fit(y_oof, q_oof, a)
        conf_rows.append({"alpha": a, **conformal_report(y_test, conformal_sets(q_test, qh))})
    conf_df = pd.DataFrame(conf_rows)
    save_table(conf_df, dirs, "S1_conformal_test")

    # abstention at fixed coverage on epistemic uncertainty
    cut = np.quantile(U["epistemic"], cfg.abstain_coverage)
    keep = U["epistemic"] <= cut
    abst = {"cut": float(cut), "coverage": float(keep.mean()), "referred": int((~keep).sum()),
            "acc_retained": float(correct[keep].mean()), "acc_referred": float(correct[~keep].mean()) if (~keep).any() else np.nan}

    # figures
    plot_roc_pr({"internal test": (y_test, q_test)}, dirs, "S1_roc_pr_test")
    plot_reliability(y_test, p_test, q_test, dirs, "S1_reliability_test")
    plot_confusions(y_test, q_test, ops, dirs, "S1_confusion_test")
    plot_uncertainty(U, correct, dirs, "S1_uncertainty_test")
    plot_risk_coverage(rc, dirs, "S1_risk_coverage_test")
    plot_conformal(conf_df, dirs, "S1_conformal_test")
    plot_dca({"internal test": (y_test, q_test)}, dirs, "S1_decision_curve_test")
    if att_test is not None:
        plot_attention_by_scale(att_test[test_eval_pos], y_test, dirs, "S1_attention_by_view", att_agg)

    np.savez_compressed(os.path.join(dirs["predictions"], "stage1_predictions.npz"),
                        test_ids=ids[test_eval], test_label=y_test, test_prob=p_test, test_prob_cal=q_test,
                        oof_ids=ids[dev_eval], oof_label=y_oof, oof_prob=p_oof, oof_prob_cal=q_oof,
                        test_mc=P.astype(np.float16), groups=groups, dev=dev, test=test)
    results = {"config": {k: v for k, v in asdict(cfg).items()}, "grouping": grep, "final_aggregator": final,
               "calibrator": cal.method, "calibrator_cv_nll": cal.cv_nll, "summary": summ, "operating_points_oof": ops,
               "abstention": abst, "n_dev_records": int(len(dev)), "n_test_records": int(len(test)),
               "n_test_scored": int(len(test_eval)), "runtime_min": (time.time() - t0) / 60}
    with open(os.path.join(cfg.out_dir, "stage1_results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"Stage 1 done in {results['runtime_min']:.1f} min | test AUROC {summ['test_calibrated']['AUROC']:.4f} "
        f"AUPRC {summ['test_calibrated']['AUPRC']:.4f} ECE {summ['test_calibrated']['ECE']:.4f}")
    return {"results": results, "run": R, "calibrator": cal, "ops": ops, "y_oof": y_oof, "q_oof": q_oof,
            "X": X, "M": M, "dev": dev, "test_eval": test_eval, "y_test": y_test, "q_test": q_test, "dirs": dirs,
            "conformal_q": conformal_fit(y_oof, q_oof, 0.10), "ids": ids, "y": y, "test_idx": test}


# =============================================================================
# 10. External data: extraction, deduplication, leakage audit, evaluation
# =============================================================================
def _find_folders(root: str, keywords: Dict[str, Tuple[str, ...]]) -> Dict[str, str]:
    found = {}
    for dirpath, dirnames, _ in os.walk(root):
        for dn in dirnames:
            low = dn.lower().replace("_", " ")
            for key, kws in keywords.items():
                if key not in found and any(k in low for k in kws):
                    found[key] = os.path.join(dirpath, dn)
    return found


def extract_external_features(cfg: Config = CFG, batch_images: int = 8) -> str:
    """Multiscale DINOv2 extraction identical to the development pipeline (no random projection)."""
    import cv2
    from torchvision import transforms
    folders = _find_folders(cfg.gv_root, cfg.gv_folders)
    log(f"GastroVision folders: { {k: os.path.basename(v) for k, v in folders.items()} }")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    recs = []
    for cohort, path in folders.items():
        for f in sorted(os.listdir(path)):
            if os.path.splitext(f)[1].lower() in exts:
                recs.append({"Patient_ID": f"{cohort}/{f}", "Image_Path": os.path.join(path, f), "cohort": cohort,
                             "label": int(cohort == "barretts")})
    man = pd.DataFrame(recs)
    dev = cfg.device
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(dev).eval()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def prep(path):
        img = cv2.imread(path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        h, w = img.shape[:2]
        s = min(h, w)
        img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t[None], size=(224, 224), mode="bilinear", align_corners=False, antialias=True)[0]
        t = norm(t)
        crops = [t]
        for g in (2, 3):
            hs = 224 // g
            for r in range(g):
                for c in range(g):
                    crops.append(F.interpolate(t[None, :, r * hs:(r + 1) * hs, c * hs:(c + 1) * hs], size=(224, 224),
                                               mode="bilinear", align_corners=False)[0])
        return torch.stack(crops)

    feats, keep = [], []
    buf, buf_i = [], []
    use_amp = cfg.amp and dev == "cuda"

    def flush():
        if not buf:
            return
        x = torch.cat(buf).to(dev)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            f = backbone(x).float().cpu().numpy().reshape(len(buf), 14, -1)
        feats.extend(list(f))
        keep.extend(buf_i)
        buf.clear()
        buf_i.clear()

    for i, p in enumerate(man["Image_Path"]):
        t = prep(p)
        if t is not None:
            buf.append(t)
            buf_i.append(i)
        if len(buf) == batch_images:
            flush()
        if (i + 1) % 200 == 0:
            log(f"  extracted {i + 1}/{len(man)}")
    flush()
    out = man.iloc[keep].copy()
    out["Endo_Instances"] = [f.tolist() for f in feats]
    os.makedirs(os.path.dirname(cfg.gv_feature_file), exist_ok=True)
    out.to_parquet(cfg.gv_feature_file, index=False)
    log(f"Saved {len(out)} external bags to {cfg.gv_feature_file}")
    backbone = None
    gc.collect()
    torch.cuda.empty_cache() if dev == "cuda" else None
    return cfg.gv_feature_file


def dedup_phash(df: pd.DataFrame, max_hamming: int = 0) -> Tuple[pd.DataFrame, int]:
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        log("imagehash not installed; skipping perceptual-hash deduplication")
        return df, 0
    hashes, keep = [], []
    for i, p in enumerate(df["Image_Path"]):
        try:
            h = imagehash.phash(Image.open(p))
        except Exception:
            continue
        if not any(h - s <= max_hamming for s in hashes):
            hashes.append(h)
            keep.append(i)
    return df.iloc[keep].reset_index(drop=True), len(df) - len(keep)


def load_external(cfg: Config = CFG) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    ids, bags, meta = load_instance_table(cfg.gv_feature_file, "Patient_ID")
    meta = meta.reset_index(drop=True)
    meta["_row"] = np.arange(len(meta))
    rep = {"n_raw": int(len(meta))}
    if "Image_Path" in meta.columns:
        parts, removed = [], 0
        for c, g in meta.groupby("cohort"):
            g2, r = dedup_phash(g, cfg.phash_hamming)
            parts.append(g2)
            removed += r
        meta = pd.concat(parts, ignore_index=True)
        rep["phash_removed_within_cohort"] = int(removed)
    X, M = pad_bags([bags[i] for i in meta["_row"]])
    rep["n_after_dedup"] = int(len(meta))
    rep["cohort_counts"] = meta["cohort"].value_counts().to_dict()
    log(f"External: {rep}")
    return meta, X, M, rep


def external_bag_dict(meta: pd.DataFrame, X: np.ndarray, M: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return {c: (X[(meta["cohort"] == c).values], M[(meta["cohort"] == c).values]) for c in meta["cohort"].unique()}


def evaluate_external(S1: Dict, meta: pd.DataFrame, X: np.ndarray, M: np.ndarray, cfg: Config = CFG) -> Dict:
    set_style()
    dirs = S1["dirs"]
    R, cal, ops = S1["run"], S1["calibrator"], S1["ops"]
    probs = R["extra_prob"]
    pos = "barretts"
    if pos not in probs:
        raise ValueError("No Barrett's cohort in external data")

    # leakage audit: nearest internal bag for every external bag
    c_int = embedding_center(S1["X"], S1["M"])
    E_int = bag_embedding(S1["X"], S1["M"], c_int)
    E_ext = bag_embedding(X, M, c_int)
    sims = []
    Ti = torch.from_numpy(E_int.astype(np.float32)).to(cfg.device)
    for s in range(0, len(E_ext), 2048):
        sims.append((torch.from_numpy(E_ext[s:s + 2048].astype(np.float32)).to(cfg.device) @ Ti.T).max(1)[0].cpu().numpy())
    sims = np.concatenate(sims)
    flagged = sims >= cfg.leakage_sim         # centred cosine: near-identical content only
    meta = meta.copy()
    meta["max_sim_to_internal"] = sims
    meta["leakage_flag"] = flagged
    meta.to_csv(os.path.join(dirs["external"], "external_leakage_audit.csv"), index=False)
    log(f"Leakage audit: {flagged.sum()} external images with cosine >= {cfg.leakage_sim} to an internal image (excluded)")

    # feature-space shift
    scorer = mahalanobis_scorer(E_int[S1["dev"]])
    shift_scores = {"internal test": scorer(E_int[S1["test_eval"]])}

    rows, curves, raw, ext_rc = [], {}, {}, {}
    keep_mask = {c: ~flagged[(meta["cohort"] == c).values] for c in probs}
    ppos = cal.transform(probs[pos][keep_mask[pos]])
    for c in probs:
        if c == pos:
            continue
        pneg = cal.transform(probs[c][keep_mask[c]])
        yy = np.r_[np.ones(len(ppos)), np.zeros(len(pneg))].astype(int)
        pp = np.r_[ppos, pneg]
        s = summary_metrics(yy, pp)
        s.update({"cohort": f"BE vs {c}", "AUROC_CI_boot": bootstrap_ci(yy, pp, roc_auc_score, cfg.bootstrap_n)[1:],
                  "AUROC_CI_HanleyMcNeil": hanley_mcneil(s["AUROC"], int(yy.sum()), int((1 - yy).sum())),
                  "AUPRC_CI_boot": bootstrap_ci(yy, pp, average_precision_score, cfg.bootstrap_n)[1:]})
        for k, t in ops.items():           # thresholds fixed from internal out-of-fold data
            r_ = at_threshold(yy, pp, t)
            s[f"{k}_sens"], s[f"{k}_spec"] = r_["sensitivity"], r_["specificity"]
        qh = S1["conformal_q"]
        s.update({f"conformal_{k}": v for k, v in conformal_report(yy, conformal_sets(pp, qh)).items()})
        rows.append(s)
        curves[f"BE vs {c}"] = (yy, pp)
        raw[c] = (yy, pp)
        Ec = E_ext[(meta["cohort"] == c).values][keep_mask[c]]
        shift_scores[f"GastroVision {c}"] = scorer(Ec)
        if "extra_mc" in R:
            Pm = np.concatenate([R["extra_mc"][pos][:, :, keep_mask[pos]], R["extra_mc"][c][:, :, keep_mask[c]]], 2)
            Ue = decompose_uncertainty(Pm)
            err = ((pp >= ops["F1-optimal"]).astype(int) != yy)
            cov_, risk_, aurc_, eaurc_, r0 = risk_coverage(err, Ue["epistemic"])
            s.update({"mean_epistemic": float(Ue["epistemic"].mean()), "AURC_epistemic": aurc_, "E-AURC_epistemic": eaurc_,
                      "risk_full": r0, "risk_at_80pct_coverage": float(risk_[int(0.8 * len(risk_)) - 1])})
            ext_rc[f"BE vs {c}"] = (cov_, risk_, aurc_, eaurc_, r0)
    ext_df = pd.DataFrame(rows)
    save_table(ext_df, dirs, "EXT_external_validation")
    plot_roc_pr(dict(list(curves.items())[:3]), dirs, "EXT_roc_pr")
    plot_dca(dict(list(curves.items())[:3]), dirs, "EXT_decision_curve")
    plot_shift(dict(list(shift_scores.items())[:5]), dirs, "EXT_feature_shift")
    if ext_rc:
        plot_risk_coverage(dict(list(ext_rc.items())[:3]), dirs, "EXT_risk_coverage")
    # internal-test epistemic uncertainty for comparison with external cohorts
    U_int = decompose_uncertainty(R["test_mc"][:, :, [list(S1["test_idx"]).index(i) for i in S1["test_eval"]]])
    unc_cmp = {"internal test": U_int["epistemic"]}
    for c in probs:
        unc_cmp[f"GastroVision {c}"] = decompose_uncertainty(R["extra_mc"][c][:, :, keep_mask[c]])["epistemic"]
    plot_shift_uncertainty(unc_cmp, dirs, "EXT_epistemic_under_shift")

    # local recalibration on a small external calibration split (reported separately)
    recal_rows = []
    rng = np.random.default_rng(0)
    for c, (yy, pp) in raw.items():
        res = []
        for _ in range(cfg.local_recal_reps):
            idx_cal, idx_ev = [], []
            for cls in (0, 1):
                w = np.where(yy == cls)[0]
                rng.shuffle(w)
                k = max(2, int(round(cfg.local_recal_frac * len(w))))
                idx_cal += list(w[:k])
                idx_ev += list(w[k:])
            idx_cal, idx_ev = np.array(idx_cal), np.array(idx_ev)
            lc = Calibrator("platt").fit(yy[idx_cal], pp[idx_cal])
            qc, qe = lc.transform(pp[idx_cal]), lc.transform(pp[idx_ev])
            t = operating_points(yy[idx_cal], qc, cfg.target_sensitivity)["F1-optimal"]
            r_ = at_threshold(yy[idx_ev], qe, t)
            e, _ = ece_mce(yy[idx_ev], qe, 10)
            res.append([r_["sensitivity"], r_["specificity"], r_["F1"], e])
        res = np.array(res)
        recal_rows.append({"cohort": f"BE vs {c}", "sens_mean": res[:, 0].mean(), "sens_sd": res[:, 0].std(),
                           "spec_mean": res[:, 1].mean(), "spec_sd": res[:, 1].std(), "F1_mean": res[:, 2].mean(),
                           "ECE_mean": res[:, 3].mean(), "calib_frac": cfg.local_recal_frac, "reps": cfg.local_recal_reps})
    save_table(pd.DataFrame(recal_rows), dirs, "EXT_local_recalibration")
    out = {"external": ext_df.to_dict(orient="records"), "local_recalibration": recal_rows,
           "leakage_flagged": int(flagged.sum()),
           "shift_auroc": {k: float(roc_auc_score(np.r_[np.zeros(len(shift_scores["internal test"])), np.ones(len(v))],
                                                   np.r_[shift_scores["internal test"], v]))
                           for k, v in shift_scores.items() if k != "internal test"}}
    with open(os.path.join(cfg.out_dir, "external_results.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("External evaluation done: " + "; ".join(f"{r['cohort']} AUROC {r['AUROC']:.3f}" for r in rows))
    return out


# =============================================================================
# 11. Stage 2: CT / PET / genomics fusion with nested cross-validation
# =============================================================================
class FusionNet(nn.Module):
    def __init__(self, dims=(512, 512, 1500), d=256, heads=4):
        super().__init__()
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(k, d), nn.LayerNorm(d), nn.GELU()) for k in dims])
        self.att = nn.MultiheadAttention(d, heads, dropout=0.2, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(d, 128), nn.LayerNorm(128), nn.GELU())
        self.dx, self.st = nn.Linear(128, 2), nn.Linear(128, 2)

    def forward(self, xs, mask):
        seq = torch.stack([e(x) for e, x in zip(self.enc, xs)], 1)
        kpm = mask == 0
        kpm[kpm.all(1)] = False
        out, w = self.att(seq, seq, seq, key_padding_mask=kpm)
        rep = self.fuse((out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1))
        if self.training:
            d = torch.stack([self.dx(F.dropout(rep, p, True)) for p in DROP_RATES]).mean(0)
            s = torch.stack([self.st(F.dropout(rep, p, True)) for p in DROP_RATES]).mean(0)
        else:
            d, s = self.dx(rep), self.st(rep)
        return d, s, w


def _load_vec_store(path, col_hint, dim):
    """{Patient_ID: vector}; prefers the original column names (Rad_Features / PET_Features / Genomic_Features)."""
    if not os.path.exists(path):
        log(f"  [warn] {os.path.basename(path)} not found -> modality absent for everyone")
        return {}
    df = pd.read_parquet(path)
    cands = [c for c in df.columns if c != "Patient_ID"]
    col = next((c for c in cands if c.lower().startswith(col_hint.lower())), cands[0])
    bank, wrong, zero = {}, 0, 0
    for pid, v in zip(df["Patient_ID"].astype(str), df[col]):
        try:
            a = np.asarray(v, dtype=np.float32).ravel()
        except Exception:
            wrong += 1
            continue
        if a.size != dim:
            wrong += 1
        elif not np.any(a):
            zero += 1
        else:
            bank[pid] = a
    log(f"  [load] {os.path.basename(path):<28s} column {col!r}: {len(bank)} patients"
        + (f", {wrong} wrong-dim rows" if wrong else "") + (f", {zero} all-zero rows treated as missing" if zero else ""))
    return bank


def _stage_label(s) -> float:
    """1 = stage III-IV, 0 = stage I-II, NaN = not recorded / unreadable."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).upper().replace("_", " ").replace("-", " ").strip()
    if t in ("", "NAN", "NONE", "NA", "N/A", "UNKNOWN", "[NOT AVAILABLE]", "[DISCREPANCY]", "[UNKNOWN]", "NOT REPORTED"):
        return np.nan
    if "LATE" in t or "ADVANCED" in t:
        return 1.0
    if "EARLY" in t:
        return 0.0
    t = re.sub(r"\bSTAGE\b", " ", t)
    m = re.search(r"\b(IV|III|II|I)[ABC]?\d?\b", t)
    if m:
        return 1.0 if m.group(1) in ("III", "IV") else 0.0
    m = re.search(r"\b([1-4])(\.0)?[ABC]?\b", t)
    if m:
        return 1.0 if m.group(1) in ("3", "4") else 0.0
    return np.nan


def load_stage2(cfg: Config = CFG) -> Dict:
    man = pd.read_csv(cfg.path(cfg.manifest), low_memory=False)
    df = man[man[cfg.label_col].isin(["Squamous Cell Carcinoma", "Adenocarcinoma"])].copy()
    df[cfg.id_col] = df[cfg.id_col].astype(str)
    if cfg.s2_id_prefix:
        df = df[df[cfg.id_col].str.startswith(cfg.s2_id_prefix)]
    banks = [_load_vec_store(cfg.path(cfg.rad_feat), "Rad", 512), _load_vec_store(cfg.path(cfg.pet_feat), "PET", 512),
             _load_vec_store(cfg.path(cfg.geno_feat), "Gen", 1500)]
    if cfg.s2_require_genomics:
        df = df[df[cfg.id_col].isin(banks[2])]
    df["dx"] = (df[cfg.label_col] == "Adenocarcinoma").astype(int)
    stage_col = cfg.s2_stage_col if cfg.s2_stage_col in df.columns else \
        next((c for c in df.columns if "stage" in c.lower()), None)
    df["stage"] = df[stage_col].map(_stage_label) if stage_col else np.nan
    n_unknown = int(df["stage"].isna().sum())
    if stage_col:
        vc = df[stage_col].fillna("<blank>").astype(str).value_counts().head(15)
        log(f"  stage column {stage_col!r} (top values): " + "; ".join(f"{k}: {v}" for k, v in vc.items()))
    log(f"  stage parsed: early {int((df['stage'] == 0).sum())}, late {int((df['stage'] == 1).sum())}, "
        f"not recorded {n_unknown} -> {'excluded' if cfg.s2_drop_unknown_stage else 'set to early (auxiliary head only)'}")
    if cfg.s2_drop_unknown_stage:
        df = df[df["stage"].notna()]
    else:
        df["stage"] = df["stage"].fillna(0.0)
    df["stage"] = df["stage"].astype(int)
    ids = df[cfg.id_col].values
    dims = (512, 512, 1500)
    Xs = [np.stack([b.get(i, np.zeros(d, np.float32)) for i in ids]) for b, d in zip(banks, dims)]
    mask = np.stack([[float(i in b) for b in banks] for i in ids]).astype(np.float32)
    keep = mask.sum(1) > 0                                  # at least one modality
    if (~keep).any():
        log(f"  {int((~keep).sum())} patients have no CT, PET or genomics vector -> excluded")
        df, ids, mask = df[keep], ids[keep], mask[keep]
        Xs = [x[keep] for x in Xs]
    log(f"Stage 2: {len(ids)} patients (SCC {int((df['dx'] == 0).sum())}, adeno {int((df['dx'] == 1).sum())}; "
        f"early {int((df['stage'] == 0).sum())}, late {int((df['stage'] == 1).sum())}); modality coverage CT/PET/G = "
        f"{mask.sum(0).astype(int).tolist()}")
    for name, v in (("subtype", df["dx"].values), ("stage", df["stage"].values)):
        if len(v) < 20 or np.bincount(v, minlength=2).min() < 5:
            raise ValueError(f"Stage 2 cohort too small for {name}: counts {np.bincount(v, minlength=2).tolist()}. "
                             f"Check the log lines above (stage values, ID filter, modality files).")
    return {"ids": ids, "dx": df["dx"].values, "stage": df["stage"].values, "Xs": Xs, "mask": mask,
            "n_unknown_stage": n_unknown}


def _train_fusion(Xs, mask, dx, st, tr, va, seed, cfg, genomics_only=False):
    seed_everything(seed)
    dev = cfg.device
    T = [torch.from_numpy(x).to(dev) for x in Xs]
    Mk = torch.from_numpy(mask).to(dev)
    if genomics_only:
        Mk = Mk.clone()
        Mk[:, :2] = 0
    model = FusionNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.s2_lr, weight_decay=cfg.s2_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)
    cd = nn.CrossEntropyLoss(weight=_class_weights(dx[tr], dev), label_smoothing=0.10)
    cs = nn.CrossEntropyLoss(weight=_class_weights(st[tr], dev), label_smoothing=0.15)
    Yd, Ys = torch.from_numpy(dx).long().to(dev), torch.from_numpy(st).long().to(dev)
    best, best_state = -1, None
    tr_t, va_t = torch.from_numpy(tr).to(dev), torch.from_numpy(va).to(dev)
    for _ in range(cfg.s2_epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        for s in range(0, len(perm), 32):
            b = perm[s:s + 32]
            mb = Mk[b].clone()
            drop = (torch.rand_like(mb[:, :2]) < cfg.s2_modality_drop) & (mb[:, :2] > 0)
            mb[:, :2] = torch.where(drop, torch.zeros_like(mb[:, :2]), mb[:, :2])
            mb[mb.sum(1) == 0] = Mk[b][mb.sum(1) == 0]
            d, s_, _ = model([t[b] for t in T], mb)
            loss = cd(d, Yd[b]) + cs(s_, Ys[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            d, s_, _ = model([t[va_t] for t in T], Mk[va_t])
            pd_, ps_ = d.argmax(1).cpu().numpy(), s_.argmax(1).cpu().numpy()
        score = 0.5 * (f1_score(dx[va], pd_, average="macro") + f1_score(st[va], ps_, average="macro"))
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, T, Mk


def _strat_labels(candidates: List[np.ndarray], n_splits: int) -> np.ndarray:
    """First label vector whose smallest class has at least n_splits members (else all zeros)."""
    for lab in candidates:
        if np.bincount(pd.factorize(lab)[0]).min() >= n_splits:
            return lab
    return np.zeros(len(candidates[0]), dtype=int)


def run_stage2(cfg: Config = CFG) -> Dict:
    set_style()
    dirs = ensure_dirs(cfg)
    D = load_stage2(cfg)
    dx, st, Xs, mask = D["dx"], D["stage"], D["Xs"], D["mask"]
    comp = dx * 2 + st
    variants = {"fusion": False, "genomics only": True}
    oof = {v: {"dx": np.zeros((cfg.s2_repeats, len(dx))), "st": np.zeros((cfg.s2_repeats, len(dx)))} for v in variants}
    att = []
    for rep in range(cfg.s2_repeats):
        skf = StratifiedKFold(cfg.s2_outer_folds, shuffle=True, random_state=1000 + rep)
        outer_lab = _strat_labels([comp, dx, st], cfg.s2_outer_folds)
        for k, (tr, te) in enumerate(skf.split(dx, outer_lab)):
            te_t = torch.from_numpy(te).to(cfg.device)
            for v, gonly in variants.items():
                pd_acc, ps_acc, w_acc = 0, 0, 0
                for e in range(cfg.s2_ensemble_seeds):
                    inner = StratifiedShuffleSplit(1, test_size=0.15, random_state=100 * rep + 10 * k + e)
                    itr, iva = next(inner.split(tr, _strat_labels([comp[tr], dx[tr], st[tr]], 2)))
                    model, T, Mk = _train_fusion(Xs, mask, dx, st, tr[itr], tr[iva], 1000 * rep + 10 * k + e, cfg, gonly)
                    with torch.no_grad():
                        d, s_, w = model([t[te_t] for t in T], Mk[te_t])
                    pd_acc = pd_acc + torch.softmax(d, 1)[:, 1].cpu().numpy()
                    ps_acc = ps_acc + torch.softmax(s_, 1)[:, 1].cpu().numpy()
                    w_acc = w_acc + w.mean(1).cpu().numpy()
                oof[v]["dx"][rep, te] = pd_acc / cfg.s2_ensemble_seeds
                oof[v]["st"][rep, te] = ps_acc / cfg.s2_ensemble_seeds
                if v == "fusion":
                    att.append((w_acc / cfg.s2_ensemble_seeds, mask[te]))
        log(f"  Stage 2 repeat {rep + 1}/{cfg.s2_repeats} done")
    rows, curves_dx, curves_st = [], {}, {}
    for v in variants:
        for task, yy in (("subtype", dx), ("stage", st)):
            pp = oof[v]["dx" if task == "subtype" else "st"].mean(0)
            a = bootstrap_ci(yy, pp, roc_auc_score, cfg.bootstrap_n)
            r_ = at_threshold(yy, pp, 0.5)
            rows.append({"model": v, "task": task, "AUROC": a[0], "AUROC_lo": a[1], "AUROC_hi": a[2],
                         "AUPRC": average_precision_score(yy, pp), "Brier": brier_score_loss(yy, pp),
                         "sensitivity": r_["sensitivity"], "specificity": r_["specificity"], "MCC": r_["MCC"],
                         "n": len(yy)})
            (curves_dx if task == "subtype" else curves_st)[v] = (yy, pp)
        if v == "fusion":
            d_auc = delong_test(dx, oof["fusion"]["dx"].mean(0), oof["genomics only"]["dx"].mean(0))
            s_auc = delong_test(st, oof["fusion"]["st"].mean(0), oof["genomics only"]["st"].mean(0))
    df = pd.DataFrame(rows)
    df["DeLong_p_fusion_vs_genomics"] = [np.nan if r["model"] == "fusion" else (d_auc[2] if r["task"] == "subtype" else s_auc[2])
                                         for _, r in df.iterrows()]
    save_table(df, dirs, "S2_nested_cv_results", "stage2")
    plot_roc_pr(curves_dx, dirs, "S2_subtype_roc_pr", "Histological subtype (nested CV, out-of-fold)")
    plot_roc_pr(curves_st, dirs, "S2_stage_roc_pr", "Stage I-II vs III-IV (nested CV, out-of-fold)")
    A = np.concatenate([a for a, _ in att]).reshape(-1, 3)
    Mav = np.concatenate([m for _, m in att]).reshape(-1, 3) > 0
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    names = ["CT", "PET", "genomics"]
    mu, ci, ns = [], [], []
    for j in range(3):
        v = A[Mav[:, j], j]
        mu.append(v.mean() if len(v) else 0)
        ci.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ns.append(int(Mav[:, j].sum() / max(cfg.s2_repeats, 1)))
    ax.bar(names, mu, yerr=ci, color=SERIES[0], width=0.55, capsize=3, ecolor=INK2)
    for j, (m_, n_) in enumerate(zip(mu, ns)):
        ax.text(j, m_ + ci[j] + 0.015, f"n={n_}", ha="center", fontsize=7.5, color=INK2)
    ax.set(ylabel="Attention when modality present", ylim=(0, max(mu) * 1.35 + 0.05),
           title="Modality attention (fusion, out-of-fold)")
    save_fig(fig, dirs, "S2_modality_attention")
    out = {"results": df.to_dict(orient="records"), "n": int(len(dx)), "n_unknown_stage": D["n_unknown_stage"],
           "modality_coverage": mask.sum(0).astype(int).tolist()}
    with open(os.path.join(cfg.out_dir, "stage2_results.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("Stage 2 done:\n" + df[["model", "task", "AUROC", "AUROC_lo", "AUROC_hi"]].to_string(index=False))
    return out


# =============================================================================
# 12. One-call entry point
# =============================================================================
def run_all(cfg: Config = CFG, extract_external: bool = True) -> Dict:
    cfg.resolve_paths()
    ensure_dirs(cfg)
    ext_bags, meta, Xe, Me = None, None, None, None
    if extract_external and not os.path.exists(cfg.gv_feature_file) and os.path.exists(cfg.gv_root):
        extract_external_features(cfg)
    if os.path.exists(cfg.gv_feature_file):
        meta, Xe, Me, _ = load_external(cfg)
        ext_bags = external_bag_dict(meta, Xe, Me)
    S1 = run_stage1(cfg, ext_bags)
    EXT = evaluate_external(S1, meta, Xe, Me, cfg) if ext_bags else None
    S2 = run_stage2(cfg)
    return {"stage1": S1["results"], "external": EXT, "stage2": S2}


if __name__ == "__main__":
    run_all(CFG)
