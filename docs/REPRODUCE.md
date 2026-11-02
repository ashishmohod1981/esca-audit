# Reproducing the results

There are three levels of reproduction. Each one uses the output of the level above it.

## 1. Full pipeline on Kaggle (GPU, about 20 minutes)

Import `notebooks/ESCA_SCI_Kaggle.ipynb` into Kaggle, attach the two datasets listed in
`docs/DATA.md`, switch on a GPU and internet access, and run all cells with `QUICK = False`.

| Cell | What it does |
|---|---|
| 1 | installs `imagehash` |
| 2 | writes `esca_sci_pipeline.py` (identical to `src/esca_sci_pipeline.py`) |
| 3 | configuration; dataset paths are found under `/kaggle/input` if they differ |
| 4 | encodes the GastroVision images with the same preprocessing and 14 views (cached) |
| 5 | Stage 1: duplicate audit, aggregator comparison, ensemble, calibration, uncertainty, conformal sets |
| 6 | external validation with the thresholds, calibrator and conformal quantiles frozen |
| 7 | Stage 2: fusion network under repeated nested cross validation, ablation with genomics only |
| 8 | displays the figures and zips the output folder |

Expected key outputs:

* `stage1_results.json`: final aggregator `mean`, calibrator `platt`, 204 test images scored,
  runtime about 11.5 minutes.
* `tables/S1_operating_points_test.csv`: AUROC 0.914 on the internal test set.
* `tables/EXT_external_validation.csv`: AUROC 0.872 for BE versus normal Z line.
* `stage2/S2_nested_cv_results.csv`: subtype AUROC 0.870 (fusion) and 0.867 (genomics only).

Small differences in the last digit can occur on different GPU types because of nondeterministic
CUDA kernels. Setting `QUICK = True` runs a smoke test in a few minutes; its numbers must not be
reported.

The Stage 2 stage head is trained as an auxiliary task. The pipeline still writes its metrics
(`task = stage`) because the missingness audit needs them, but stage is not a reported endpoint.

## 2. Missingness audit

On Kaggle, after the datasets are attached:

```bash
python scripts/missingness_audit.py \
    --data-root /kaggle/input/datasets/bhavanamohod/final-mm-features/mm-dataset \
    --curated-manifest /path/to/multimodal_master_dataset_pruned1.csv
```

Anywhere, with only the curated manifest:

```bash
python scripts/missingness_audit.py --manifest-only /path/to/multimodal_master_dataset_pruned1.csv
```

The manifest mode reproduces the cross tabulation of Table IX exactly (19 patients without a
genomic profile, all outside TCGA; 18 with CT, 6 also with PET; 15 of 19 at stage III or IV,
against 57 of 183 TCGA patients). Its availability only AUROC for subtype is 0.560, because the
availability flags come from the manifest and not from the feature stores; the pipeline mode,
which uses the 199 patient analytic cohort, gives the value of 0.561 reported in the paper.

## 3. Statistics and figures from the saved predictions (CPU, seconds)

```bash
python scripts/reproduce_paper_statistics.py   # 27 statistics checked against the paper
python scripts/make_result_figures.py          # Figs. 4, 5 and 7 into figures/results/
```

`results/predictions/` contains:

| File | Content |
|---|---|
| `internal_test_predictions.csv` | 204 internal test images: label, raw and calibrated probability, uncertainty |
| `internal_oof_predictions.csv` | 813 development images: out of fold raw and calibrated probability |
| `external_predictions_recomputed.csv` | 671 GastroVision images scored by the saved ensemble with the frozen calibrator; the metrics agree with `external_results.json` |
| `stage1_predictions.npz` | the same predictions plus Monte Carlo samples and the source image groups |
| `stage1_ensemble.pt` | weights of the 25 ensemble members, the aggregator name and the run configuration (the Platt calibrator is refitted from the out of fold predictions) |

## Diagrams

Figs. 1 to 3 are drawn in TikZ:

```bash
python scripts/make_diagram_assets.py --frame path/to/deidentified_frame.png --ct path/to/CTChest.nrrd
cd figures/tikz
for f in overview stage1 stage2; do pdflatex $f.tex; done
```

The image panels are already in `figures/assets/`, so the second step works on its own.
