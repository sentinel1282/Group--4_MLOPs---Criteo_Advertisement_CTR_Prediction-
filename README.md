# Criteo Advertisement CTR Prediction — MLOps on AWS
**AAI-540 Machine Learning Operations · Group 4** — Prashant Khare, Manu Malla, Laxminag Mamillapalli

Predicts the probability that a user clicks a display ad, using the Criteo Display Advertising Challenge
dataset (45.8 M labelled impressions, 13 numeric + 26 hashed categorical features). This repository holds the
data engineering and feature engineering stages of the system described in our ML System Design Document.

## Module 2–3 scope (this milestone)

| Requirement | Where | Output |
|---|---|---|
| Collect raw data and store it in an S3 data lake | `notebooks/01` | `s3://<bucket>/criteo-ctr/raw/train/train.txt` (immutable) + curated Parquet sample |
| Athena tables for cataloging and querying | `notebooks/01`, `04` | `criteo_ctr_db.raw_train`, `curated_sample`, `model_dataset` |
| Exploratory data analysis in SageMaker | `notebooks/02` | figures in `reports/figures/`, `artifacts/eda_summary.json` |
| Feature engineering → Feature Store | `notebooks/03` | 3 feature groups, offline store in S3 |
| Train ~40% / validation ~10% / test ~10% / production ~40% | `notebooks/04` | `splits/…`, `artifacts/split_manifest.json` |

## Run order
Open in a **SageMaker notebook instance** (`conda_python3` kernel, `ml.m5.xlarge` or larger, **volume ≥ 50 GB**)
and run the notebooks in order. Each is idempotent and safe to re-run.

1. **`01_data_lake_and_athena.ipynb`** — downloads the dataset (Kaggle API, or an archive you place in
   `s3://<bucket>/criteo-ctr/landing/`), lands it in S3, builds the chronological sample, creates Athena tables,
   and validates row counts and CTR through SQL.
2. **`02_exploratory_data_analysis.ipynb`** — label balance, CTR over time, missingness (and whether it predicts
   clicks), numeric skew and correlation, univariate signal per feature, categorical cardinality, and the
   unseen-level rate after the training window. Findings are computed, not hand-written.
3. **`03_feature_store.ipynb`** — engineers stateless features, creates the three feature groups, ingests, and
   verifies the offline store through Athena. Set `CRITEO_FS_INGEST_LIMIT=5000` for a quick dry run first.
4. **`04_data_splits.ipynb`** — joins the offline tables, splits chronologically, fits the categorical encoder on
   train only, runs eight validation checks, writes XGBoost-ready CSVs, and reconciles counts in Athena.

## Key design decisions
**Chronological systematic sample.** `train.txt` is ordered in time across 7 days but has no timestamp. We keep
every 45th row (≈ 1.02 M rows) so the sample spans all 7 days in order; `record_id` is the row's position in the
original file. `event_time` is **synthetic** — spread uniformly over 7 days by position — because Feature Store
requires one. Treat time-of-day analyses as approximate.

**Chronological 40 / 10 / 10 / 40 split.** A deployed CTR model always predicts future traffic, so the splits are
contiguous in time and *production* is the most recent 40%. Random splitting would let same-session impressions
leak across splits and hide drift.

**Stateless features in the Feature Store; stateful ones fitted on train only.**

| Feature group | Features |
|---|---|
| `criteo-ctr-numeric-fg` | `I1–I13` as `sign(x)·log1p(|x|)` (I2 has negatives), missing → 0, plus `I*_missing` flags; `day_index` |
| `criteo-ctr-categorical-fg` | `C1–C26` hex ids hashed into 2²⁰ buckets (0 = missing, 1 reserved for rare/unseen) |
| `criteo-ctr-label-fg` | `label` — isolated so no inference path can read it |

Rare-level collapsing and frequency encoding learn from data, so they are fitted in notebook 04 on the training
split only and saved to `artifacts/categorical_encoder.json` for reuse at inference.

## S3 layout
```
s3://<bucket>/criteo-ctr/
├── raw/train/train.txt              immutable source (Athena: raw_train)
├── raw/test_unlabeled/test.txt      Kaggle test set - no labels, archived only
├── curated/sample/day_index=*/      chronological sample, Parquet (Athena: curated_sample)
├── feature-store/                   offline store for the 3 feature groups
├── splits/xgboost/{train,validation,test}/   label-first CSV, no header
├── splits/xgboost/production/production_features.csv   no label (Batch Transform input)
├── splits/ground_truth/production_labels.parquet        held back for model-quality monitoring
├── splits/parquet/split=*/          full frames (Athena: model_dataset)
└── artifacts/                       manifests, encoder, feature column order, EDA summary
```

## Testing
All transformation and split logic lives in `src/criteo_ctr/` as pure functions, so the notebooks call tested code.

```bash
pip install pandas numpy pyarrow pytest
pytest -q tests                       # 21 tests: sampling, transforms, splits, leakage, CSV format
python tools/local_dry_run.py         # end-to-end on synthetic Criteo-format data, no AWS
```
The leakage test asserts the encoder's category counts equal counts from the training split alone, and that a
level first seen after the training window maps to the rare bucket with frequency 0.
`tests/synthetic.py` generates **schema-only fake data** for tests; it is never used for analysis.

## Cost notes
Full-table Athena queries on the raw TSV scan ≈ 11 GB (≈ $0.05 each); the Parquet sample is far cheaper.
Feature Store ingestion writes ≈ 3 M records to an offline-only store. Stop the notebook instance when idle.
