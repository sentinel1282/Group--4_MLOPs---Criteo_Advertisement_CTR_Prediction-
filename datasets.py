"""Assemble train / validation / test / production datasets from feature-store data.

Input is one row per record_id carrying the stateless features from all three
feature groups (numeric, categorical-hashed, label). Output is four contiguous,
chronological splits with the stateful categorical encoding fitted on train only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from . import config as C
from . import features as F

HASH_COLS = [f"{c}_hash" for c in C.CAT_COLS]
NUMERIC_FS_COLS = list(C.NUM_COLS) + [f"{c}_missing" for c in C.NUM_COLS]
META_COLS = [C.RECORD_ID, C.EVENT_TIME, C.DAY_INDEX]


@dataclass
class ModelDatasets:
    splits: Dict[str, pd.DataFrame]
    encoder: F.CategoricalEncoder
    manifest: Dict = field(default_factory=dict)


def build_model_datasets(joined: pd.DataFrame, min_count: int = C.RARE_MIN_COUNT) -> ModelDatasets:
    required = META_COLS + [C.LABEL] + NUMERIC_FS_COLS + HASH_COLS
    missing = [c for c in required if c not in joined.columns]
    if missing:
        raise KeyError(f"joined feature data is missing columns: {missing[:5]}...")

    df = joined.sort_values(C.RECORD_ID, kind="mergesort").reset_index(drop=True)
    if df[C.RECORD_ID].duplicated().any():
        raise ValueError("duplicate record_id after feature-store join; dedupe offline store first")

    df["split"] = F.assign_chronological_split(len(df))
    is_train = df["split"] == "train"

    encoder = F.CategoricalEncoder(min_count=min_count).fit(df.loc[is_train, HASH_COLS])
    encoded = encoder.transform(df[HASH_COLS])

    feature_cols = F.model_feature_columns()
    full = pd.concat([df[META_COLS + [C.LABEL, "split"]], df[NUMERIC_FS_COLS], encoded], axis=1)
    full[C.LABEL] = full[C.LABEL].astype("int64")

    splits = {name: full.loc[full["split"] == name, META_COLS + [C.LABEL] + feature_cols]
              .reset_index(drop=True) for name in C.SPLIT_ORDER}
    result = ModelDatasets(splits=splits, encoder=encoder)
    result.manifest = _manifest(result, total=len(df))
    return result


def _manifest(result: ModelDatasets, total: int) -> Dict:
    info = {"total_records": int(total), "split_order": C.SPLIT_ORDER,
            "target_fractions": C.SPLIT_FRACTIONS, "rare_min_count": result.encoder.min_count,
            "encoder_fitted_on": "train", "encoder_n_train": int(result.encoder.n_train_),
            "n_features": len(F.model_feature_columns()), "splits": {}}
    for name, d in result.splits.items():
        info["splits"][name] = {
            "rows": int(len(d)),
            "fraction": round(len(d) / total, 6) if total else 0.0,
            "ctr": round(float(d[C.LABEL].mean()), 6) if len(d) else None,
            "record_id_min": int(d[C.RECORD_ID].min()) if len(d) else None,
            "record_id_max": int(d[C.RECORD_ID].max()) if len(d) else None,
            "day_index_min": int(d[C.DAY_INDEX].min()) if len(d) else None,
            "day_index_max": int(d[C.DAY_INDEX].max()) if len(d) else None,
        }
    return info


def validate_model_datasets(result: ModelDatasets, tolerance: float = 0.001) -> Dict[str, bool]:
    """Hard checks that must all pass before any split is written to S3."""
    s, m = result.splits, result.manifest
    total = m["total_records"]
    ids = {k: set(v[C.RECORD_ID]) for k, v in s.items()}
    names = C.SPLIT_ORDER
    checks = {
        "row_counts_sum_to_total": sum(len(v) for v in s.values()) == total,
        "proportions_within_tolerance": all(
            abs(len(s[k]) / total - C.SPLIT_FRACTIONS[k]) <= max(tolerance, 1.5 / total) for k in names),
        "no_record_overlap": all(ids[a].isdisjoint(ids[b])
                                 for i, a in enumerate(names) for b in names[i + 1:]),
        "strictly_chronological": all(
            s[names[i]][C.RECORD_ID].max() < s[names[i + 1]][C.RECORD_ID].min()
            for i in range(len(names) - 1) if len(s[names[i]]) and len(s[names[i + 1]])),
        "no_missing_values": all(not v.isna().any().any() for v in s.values()),
        "encoder_fitted_on_train_only": result.encoder.n_train_ == len(s["train"]),
        "both_classes_in_train": s["train"][C.LABEL].nunique() == 2,
        "feature_columns_consistent": len({tuple(v.columns) for v in s.values()}) == 1,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"dataset validation failed: {failed}")
    return checks


def to_xgboost_csv(frame: pd.DataFrame, include_label: bool = True) -> str:
    """SageMaker built-in XGBoost CSV format: label in column 0, no header, no index."""
    cols = ([C.LABEL] if include_label else []) + F.model_feature_columns()
    return frame[cols].to_csv(header=False, index=False, float_format="%.6g")
