# Criteo Advertisement CTR Prediction — MLOps on AWS

**AAI-540 Machine Learning Operations · Group 4** — Prashant Khare, Manu Malla, Laxminag Mamillapalli

Predicts the probability that a user clicks a display ad, using the
[Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)
dataset (45.8M labelled impressions, 13 numeric + 26 hashed categorical features). This
repository is the complete codebase for the ML system described in our ML System Design
Document: data lake, feature store, training and evaluation, dual-mode deployment,
monitoring, and a gated CI/CD pipeline.

![as-built architecture](reports/figures/architecture_as_built.png)

Two diagrams, deliberately: `docs/architecture.md` holds this **as-built** view beside the
**reference architecture** from the design document, with a table of what is implemented and
what is scope. A design document whose diagram promises services the repository does not
contain is a wish, not a design.

---

## Quick start

```bash
git clone <this repo> && cd Group--4_MLOPs---Criteo_Advertisement_CTR_Prediction-
pip install -r requirements.txt

pytest tests -q                        # 104 tests, no AWS needed, ~6s
python tools/local_dry_run.py          # whole system end to end on synthetic data, ~3s
```

Both run with no AWS account and no Criteo data. The dry run exercises every stage —
features, splits, benchmarks, XGBoost, the evaluation payload the pipeline gate reads, and
the data- and model-quality monitors — against generated Criteo-shaped data, so a broken
column order or a leaking encoder fails in seconds rather than twenty minutes into a paid
pipeline execution.

To run the real system, open the notebooks in a **SageMaker notebook instance**
(`conda_python3`, `ml.m5.xlarge` or larger, **volume ≥ 50 GB**) and run them in order.

---

## Run order

| # | Notebook | What it produces |
|---|---|---|
| 01 | `data_lake_and_athena` | S3 raw zone (immutable `train.txt`), chronological Parquet sample, Athena `raw_train` + `curated_sample`, row-count and CTR validation in SQL |
| 02 | `exploratory_data_analysis` | label balance, CTR across the 7 days, missingness and its effect on clicks, numeric skew, categorical cardinality, unseen-level rate; `eda_summary.json` + figures |
| 03 | `feature_store` | three SageMaker Feature Store groups (numeric, categorical, label) with an offline store, verified through Athena |
| 04 | `data_splits` | chronological 40/10/10/40 splits, encoder fitted on train only, 8 validation checks, XGBoost CSVs, held-back production labels, Athena `model_dataset` |
| 05 | `benchmark_model` | global-CTR heuristic + two-feature logistic baseline; **sets the CI/CD quality-gate threshold** |
| 06 | `model_training` | XGBoost training job with Debugger rules, Bayesian tuning on `validation:logloss`, one-time test scoring, benchmark comparison |
| 07 | `deployment` | Model Registry version + model card, manual approval, Batch Transform over the production split, real-time endpoint with data capture, **batch/real-time parity check** |
| 08 | `monitoring` | data-quality and model-quality baselines and hourly schedules, ground truth released on a lag, Clarify attribution |
| 09 | `infrastructure_monitoring` | SNS topic, 5 CloudWatch alarms, the system dashboard, a deliberately triggered alarm |
| 10 | `cicd_pipeline` | the SageMaker Pipeline DAG, run twice: once passing the gate, once failing it on purpose |

Each notebook is idempotent and safe to re-run.

---

## Repository layout

```
notebooks/        01-10, in run order; every figure is generated, never pasted
src/criteo_ctr/   the library the notebooks, the pipeline and the tests all import
  config.py       buckets, column lists, split fractions, thresholds — defined once
  io_utils.py     S3 or local-directory store, Athena-safe column naming
  athena.py       minimal boto3-only Athena client
  features.py     stateless transforms + the stateful CategoricalEncoder
  datasets.py     chronological splits, validation checks, XGBoost CSV format
  benchmark.py    global-CTR heuristic and two-feature logistic baseline
  evaluation.py   log loss, normalised entropy, AUC, PR-AUC, calibration, lift
  monitoring.py   baselines, drift, delayed-label join, alarms, dashboard
  deployment.py   registration, serving, batch/real-time parity, model card
  pipeline.py     the SageMaker Pipeline definition and its contract
scripts/          pipeline step entry points: preprocess.py, evaluate.py,
                  record_preprocessor.py (Model Monitor hook)
tests/            104 unit + integration tests, including the leakage test
tools/            local_dry_run.py, make_architecture_diagram.py
docs/             data documentation, model card, runbook, validation evidence
reports/figures/  generated figures
artifacts/        manifests, encoder, evaluation payloads, pipeline definition
```

**One implementation per transform.** The notebooks, the Processing job in the pipeline, and
the tests all import the same functions from `src/criteo_ctr`. A transform implemented twice
is training-serving skew waiting for a deadline.

---

## Data

Raw data is **not** in this repository. It lives in S3 and is documented in
[`docs/data.md`](docs/data.md) — source, licence, S3 layout, Athena tables, the data
dictionary for all 39 features, and the derived columns.

```
s3://<bucket>/criteo-ctr/
├── raw/train/train.txt                        immutable source (Athena: raw_train)
├── curated/sample/day_index=*/                chronological sample, Parquet
├── feature-store/                             offline store, 3 feature groups
├── splits/xgboost/{train,validation,test}/    label-first CSV, no header
├── splits/xgboost/production/                 features only — Batch Transform input
├── splits/ground_truth/                       held-back labels for model-quality monitoring
├── splits/parquet/split=*/                    full frames (Athena: model_dataset)
├── models/                                    artefacts + evaluation.json
├── predictions/                               batch output, aligned predictions
├── monitoring/                                baselines, capture, reports, dashboard
└── artifacts/                                 encoder, manifests, model card, records
```

---

## Decisions worth knowing before reading the code

**Chronological splits, never random.** A deployed CTR model always predicts *future*
traffic. A random split lets same-session impressions and same-day campaigns appear on both
sides, inflating measured performance and hiding the drift the monitoring exists to catch.
`production` is deliberately the most recent 40%, with its labels held back.

**The encoder is fitted on the training split alone.** Rare-level collapsing and frequency
encoding learn from data, so they run *after* the split. `tests/test_datasets.py` asserts
the encoder's category counts equal counts computed from training rows only, and that a
level first seen after the training window maps to the rare bucket. Without that test,
nothing in the system would notice the one mistake that would invalidate every reported
number.

**Log loss is primary; normalised entropy makes it readable.** Raw log loss is not
comparable across datasets with different click rates. Normalised entropy — log loss divided
by the entropy of the base rate — is below 1.0 exactly when the model beats predicting the
average, which is the question that actually decides whether to deploy.

**AUC is reported beside it, always.** The global-CTR benchmark scores a respectable log
loss while having no ranking power whatsoever (AUC 0.5). Gating on log loss alone against
that baseline would pass a model with no discriminative value.

**The CI/CD gate uses log loss, not the lab's MSE.** The Module 6 lab predicts abalone age,
a regression target. Clicks are binary. Using MSE would leave the gate, the training
objective and the production monitors disagreeing about what "better" means.

**Drift on categorical features is judged against the training rate.** High-cardinality
Criteo columns already send a large share of *training* rows to the rare bucket by
construction, so an absolute ceiling fires on healthy traffic and the monitor gets muted
within a week. `monitoring.baseline_constraints` records the per-column training rate and
alarms on the increase.

**Model approval is manual.** The model sets bid prices. The pipeline registers candidates
as `PendingApproval` and a person promotes them after reading the metrics and the model card.

**`event_time` is synthetic.** Criteo documents `train.txt` as chronologically ordered but
ships no timestamp, and Feature Store requires an event-time feature. We spread rows
uniformly across the 7-day window by position. Treat time-of-day analysis as approximate —
this is stated wherever `event_time` is used rather than buried.

---

## Testing

```bash
pytest tests -q                        # 104 tests
pytest tests -q -k leak                # the leakage guarantees specifically
python tools/local_dry_run.py          # end-to-end on synthetic data
```

| Suite | Covers |
|---|---|
| `test_features.py` | signed log transform, missing flags, deterministic hashing, record keys, sampling |
| `test_datasets.py` | split geometry, **leakage**, rare-bucket mapping, encoder round-trip, CSV format |
| `test_evaluation.py` | every metric against a closed-form value, not against itself |
| `test_benchmark.py` | both baselines, including that the constant predictor has AUC 0.5 |
| `test_monitoring.py` | baselines, PSI, unseen levels, delayed-label join, degradation detection, alarms, dashboard |
| `test_deployment.py` | payload column order, batch output alignment, parity, model card contents |
| `test_pipeline_contract.py` | step and parameter names; that the gate has **both** branches |
| `test_scripts.py` | `preprocess.py` and `evaluate.py` run end to end, including the `model.tar.gz` round trip |

`tests/synthetic.py` generates **schema-only** fake data: the column names, dtypes, hex
category ids, missing patterns and chronological ordering match `train.txt`, with a
deliberate signal injected and a set of categorical levels that appear only in the final
quarter so the drift monitor has something real to detect. It is never used for analysis.

CI (`.github/workflows/ci.yml`) runs the suite, the dry run, the pipeline-contract check and
a notebook-parse check on every push and pull request.

---

## Costs and cleanup

Full-table Athena queries over the raw TSV scan ≈ 11 GB (≈ $0.05 each); the Parquet sample
is far cheaper. The real-time endpoint bills by the hour whether or not it is serving —
**notebook 07's last cell deletes it**, and leaving it up is the easiest way to exhaust a
Learner Lab budget. Monitoring schedules run hourly; delete them with
`sm.delete_monitoring_schedule(...)` when the demonstration is recorded.

---

## Deliverables

| Deliverable | Where |
|---|---|
| ML System Design Document | submitted separately; links back to this repository |
| ML System Operation Validation (video) | evidence checklist in [`docs/operation_validation.md`](docs/operation_validation.md) |
| Codebase | this repository |

Operational procedures are in [`docs/runbook.md`](docs/runbook.md); the model's intended use
and its limitations are in [`docs/model_card.md`](docs/model_card.md).
