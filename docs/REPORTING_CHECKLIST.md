# Reporting checklist (TRIPOD+AI topics)

The paper is reported following the TRIPOD+AI statement (Collins et al., BMJ 2024;385:e078378).
The table below shows where each main topic is covered. Journals usually ask for the official
checklist form; the section names below can be copied into it.

| Topic | Where it is reported |
|---|---|
| Title and abstract: model type, task, data, main results | Title; Abstract |
| Background and rationale, intended use | Introduction (paragraphs 1 to 4) |
| Objectives | Introduction, list of contributions |
| Data sources, setting and dates | Methods A (Study Design), B (Stage 1 Data), J (Stage 2 Data); Table II |
| Eligibility and exclusions | Methods B (nine images with failed features), J (no informative embedding) |
| Data preparation | Methods C (Preprocessing and Multiscale Instances) |
| Outcome definition and reference standard | Methods B and J; Limitations (endoscopic reference standard) |
| Predictors and their handling | Methods C, D, K; Table III |
| Sample size | Table II; Limitations (19 BE test images) |
| Missing data | Methods J and M (Missingness Audit); Results, Stage 2 Missingness Audit; Table IX |
| Data leakage and splitting | Methods B (Grouping by Source Image, Partitioning); Results A |
| Model development and selection | Methods D, E, K, L; Table III; Table IV |
| Calibration | Methods F; Results C; Table V; Fig. 4c |
| Performance measures and uncertainty of estimates | Methods F, G, H, N (Statistical Analysis) |
| Class imbalance | Methods E (class weights), F (AUPRC and MCC) |
| Model output and decision thresholds | Methods F (four thresholds), H (conformal sets); Results D, F |
| External validation | Methods I; Results H and I; Table VIII; Fig. 7 |
| Model updating | Results H (local recalibration on 30% of external images) |
| Interpretation and clinical use | Discussion A to H |
| Limitations | Section VI (Limitations) |
| Ethics | Ethics Statement |
| Funding and conflicts of interest | Funding and Conflicts of Interest |
| Data availability | Data Availability; docs/DATA.md |
| Code availability | Code Availability; this repository |
| Use of AI tools in writing or coding | Use of AI Assisted Tools |

Patient and public involvement: none, because only public deidentified datasets were used.
Fairness across subgroups could not be assessed, because the public datasets do not provide
age, sex or ethnicity for the images.
