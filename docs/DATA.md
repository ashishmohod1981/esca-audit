# Data sources

No new data were collected for this study. All data are public and deidentified.

## Stage 1, internal: HyperKvasir

* Source: Borgli et al., "HyperKvasir, a comprehensive multi-class image and video dataset for
  gastrointestinal endoscopy," Scientific Data 7, 283 (2020).
* Classes used: `barretts` (41 images) and `barretts-short-segment` (53 images) as BE, and
  `z-line` as normal.
* In the project feature store each source image has two records, the original and one
  augmented copy. Nine normal images whose features failed to extract were excluded, which
  leaves 1,017 source images (94 BE, 923 normal) and 2,034 records.

## Stage 1, external: GastroVision

* Source: Jha et al., "GastroVision: A multi-class endoscopy image dataset for computer aided
  gastrointestinal disease detection," ICML Workshop ML4MHD (2023).
* Kaggle mirror: `orvile/gastrovision-gastrointestinal-disease-detection`.
* Folders used: Barrett's esophagus (95), normal Z line (330), normal esophagus (140) and
  esophagitis (107). One exact duplicate was removed, leaving 671 images.
* The pipeline compares every external image with all internal images and would exclude any
  pair with a centered cosine similarity of 0.98 or more. None reached that level (maximum 0.975).
  The audit is saved in `results/tables/external_leakage_audit.csv`.

## Stage 2: CT, PET and genomics

* Genomic profiles from TCGA esophageal carcinoma (Genomic Data Commons) and imaging from The
  Cancer Imaging Archive.
* Precomputed embeddings (genomics 1,500 dimensions, CT and PET 512 dimensions each) are in the
  Kaggle dataset `bhavanamohod/final-mm-features`, folder `mm-dataset`:
  `multimodal_master_dataset_pruned.csv`, `rad_sequence_feat.parquet`,
  `pet_sequence_feat.parquet`, `genomics_snn_feat.parquet` and `endo_instance_feat.parquet`.
* An embedding made of zeros only is treated as missing. The analytic cohort has 199 patients
  (107 SCC, 92 adenocarcinoma); genomics is available for 180, CT for 34 and PET for 5.
* The source cross tabulation of the missingness audit uses the curated manifest
  `multimodal_master_dataset_pruned1.csv`. Place it next to the other files, or pass its path to
  `scripts/missingness_audit.py` with `--curated-manifest`.

## Images in the diagrams

* The white light frame in Figs. 1 and 2 is one deidentified sample image. All burned in text
  was removed and only the circular field of view is kept.
* The CT slice in Fig. 3 is slice 84 of the `CTChest` sample volume distributed with 3D Slicer.
* The PET uptake and the genomic heatmap in Fig. 3 are illustrations and do not show patient data.
