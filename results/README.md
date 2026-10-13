# Results

This directory holds the output of `src/train.py` (per-epoch training history)
and `src/evalnoise.py` (corruption-robustness evaluation). The files at the top
level of `results/` are the **final, reported run** — the one used in
`papers/conference_101719.tex` — produced on 2026-10-11 with all four
architectures (`trialcnn`, `resnet18`, `mobilenet_v2`, `efficientnet_b0`)
trained from scratch on the `tform_robust` (grayscale, median-filtered,
CLAHE-equalized) preprocessing pipeline in `src/preproc/transforms.py`.

## File naming

- `history_<model>_<dataset>_mixed.csv` — per-epoch training/test loss,
  accuracy, F1, precision, recall for `<model>` on `<dataset>`. Despite the
  `mixed` suffix (inherited from the `useprob=True` flag in `train.py`), these
  runs used `cprob=0.0`, i.e. **clean-data training only** — no corruption was
  injected during training. The last row of each file is the numbers reported
  in Table II (clean baseline) of the paper.
- `<model>_<DATASET>_noise_conf_mat_<timestamp>.csv` — confusion matrix for
  `<model>` on `<DATASET>`, accumulated across all 14 corruption types and 5
  severities from `src/evalnoise.py`.
- `noise_results_all_models_20261011_131132.csv` — the master table: one row
  per (model, dataset, corruption, severity) combination (840 rows total),
  with accuracy/F1/precision/recall. This is the source data behind Table III
  and Figures 1–3 in the paper.

## `archive/`

Superseded runs kept for provenance, not used in the paper:

- `noise_results*.csv`, `*_noise_conf_mat_2026-09-23*.csv` (Sep 2026) — the
  project's original single-CNN baseline, trained with the plain
  resize-and-normalize RGB transform (the version still commented out at the
  top of `src/preproc/makedat.py`), before the robust grayscale preprocessing
  shield and the three additional architectures were introduced.
- `history_gtsrb_mixed.csv`, `history_btsd_mixed.csv`, `history_ctsd_mixed.csv`,
  `history_*_clean.csv`, `history_btsd_fullnoise.csv` — early training-history
  files from that same pre-refactor baseline (no `model` column yet), plus a
  one-off BTSD full-noise-training experiment that was not repeated for the
  other datasets/architectures.
- `*_noise_conf_mat_20261008_*.csv`, `*_noise_conf_mat_20261010_*.csv`,
  `noise_results_all_models_20261008_*.csv`,
  `noise_results_all_models_20261010_*.csv` — intermediate per-model
  evaluation runs captured while the four-architecture pipeline was being
  stood up, superseded by the 2026-10-11 run above.

If you need to regenerate the top-level files, re-run `src/train.py` followed
by `src/evalnoise.py`; both read/write paths are controlled by `RESULTS` in
`src/config.py`.
