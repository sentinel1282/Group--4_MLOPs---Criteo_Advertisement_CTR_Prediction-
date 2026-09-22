"""Feature engineering and data-splitting logic for the Criteo CTR project.

Design rule: everything in this module is a pure function of its inputs, so the
same code runs in a SageMaker notebook, a Processing job, a unit test, and (later)
the inference path. There are two kinds of transformation, and the distinction
is what prevents leakage:

* **Stateless** transforms (signed log, missing flags, hashing) depend only on
  the row itself. They are safe to compute before splitting and are what we
  store in SageMaker Feature Store.
* **Stateful** transforms (rare-level collapsing, frequency encoding) learn
  statistics from data. They are fitted on the *training split only* and then
  applied to validation, test and production.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, Iterator, List

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------- loading
def raw_dtypes() -> Dict[str, str]:
    """dtypes for reading train.txt. Nullable Int64 keeps missing ints as <NA>."""
    d = {C.LABEL: "int8"}
    d.update({c: "Int64" for c in C.NUM_COLS})
    d.update({c: "string" for c in C.CAT_COLS})
    return d


def read_raw_chunks(path_or_buf, chunksize: int = C.READ_CHUNKSIZE) -> Iterator[pd.DataFrame]:
    """Stream the tab-separated, header-less Criteo file in order."""
    return pd.read_csv(
        path_or_buf, sep="\t", header=None, names=C.RAW_COLUMNS,
        dtype=raw_dtypes(), chunksize=chunksize, na_values=[""], keep_default_na=False,
    )


def systematic_sample(chunks: Iterable[pd.DataFrame], every: int) -> pd.DataFrame:
    """Keep every ``every``-th row of the stream, preserving order.

    Adds ``record_id`` = the row's 0-based position in the *original* file, so
    chronology survives sampling and each record can be traced to its source.
    """
    kept: List[pd.DataFrame] = []
    offset = 0
    for chunk in chunks:
        positions = np.arange(offset, offset + len(chunk), dtype=np.int64)
        mask = positions % every == 0
        if mask.any():
            part = chunk.loc[mask].copy()
            part.insert(0, C.RECORD_ID, positions[mask])
            kept.append(part)
        offset += len(chunk)
    out = pd.concat(kept, ignore_index=True)
    out.attrs["source_rows"] = offset
    return out


def fast_systematic_sample(path, every: int, total_rows: int) -> pd.DataFrame:
    """Same output as systematic_sample, but lets the C parser skip unwanted lines.

    Parsing only 1 row in ``every`` is several times faster on the 11 GB file
    than parsing every row and discarding most of them. ``total_rows`` comes from
    ``wc -l`` and is recorded so callers can compute day_index.
    """
    df = pd.read_csv(path, sep="\t", header=None, names=C.RAW_COLUMNS, dtype=raw_dtypes(),
                     na_values=[""], keep_default_na=False,
                     skiprows=lambda i: i % every != 0)
    df.insert(0, C.RECORD_ID, np.arange(len(df), dtype=np.int64) * every)
    df.attrs["source_rows"] = int(total_rows)
    return df


def add_record_keys(df: pd.DataFrame, total_rows: int,
                    start_epoch: int = C.SYNTHETIC_START_EPOCH,
                    days: int = C.COLLECTION_DAYS) -> pd.DataFrame:
    """Add ``day_index`` and a synthetic, monotonic ``event_time``.

    Criteo states the training rows are in chronological order but provides no
    timestamp. Feature Store requires an event-time feature, so we spread the
    rows uniformly across the 7-day window by position. This is an
    approximation (traffic is not uniform across the day) and must be described
    as such wherever event_time is used.
    """
    out = df.copy()
    frac = out[C.RECORD_ID].to_numpy(dtype=np.float64) / float(total_rows)
    out[C.DAY_INDEX] = np.minimum((frac * days).astype(np.int64), days - 1)
    out[C.EVENT_TIME] = start_epoch + frac * days * 86_400.0
    return out


# ------------------------------------------------------ stateless features
def signed_log1p(x: pd.Series) -> pd.Series:
    """sign(x) * log1p(|x|). Plain log1p fails on I2, which has negative values."""
    v = x.astype("float64")
    return np.sign(v) * np.log1p(np.abs(v))


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Signed-log transform + explicit missing flags for I1..I13.

    Missing values are imputed with 0 *after* the transform and paired with an
    ``*_missing`` flag, because in ad-serving logs missingness is informative
    rather than random; the flag lets the model learn that directly.
    """
    out = pd.DataFrame(index=df.index)
    for c in C.NUM_COLS:
        missing = df[c].isna()
        out[c] = signed_log1p(df[c]).fillna(0.0).astype("float64")
        out[f"{c}_missing"] = missing.astype("int64")
    return out


def hash_hex(series: pd.Series, buckets: int = C.HASH_BUCKETS) -> pd.Series:
    """Map Criteo's 8-hex-char category ids into a fixed-width integer space.

    Deterministic across processes (unlike Python's built-in hash()).
    Missing -> MISSING_ID (0). Ids 0 and 1 are reserved, so real levels map to
    [2, buckets).
    """
    out = pd.Series(C.MISSING_ID, index=series.index, dtype="int64")
    present = series.notna()
    if present.any():
        as_int = series[present].astype(str).map(lambda h: int(h, 16))
        out.loc[present] = (as_int % (buckets - 2) + 2).astype("int64")
    return out


def categorical_features(df: pd.DataFrame, buckets: int = C.HASH_BUCKETS) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in C.CAT_COLS:
        out[f"{c}_hash"] = hash_hex(df[c], buckets)
    return out


def build_feature_group_frames(sample: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split the engineered sample into the three Feature Store groups.

    The label lives in its own group so that nothing on the inference path can
    ever read it by accident.
    """
    keys = sample[[C.RECORD_ID, C.EVENT_TIME]].reset_index(drop=True)
    num = numeric_features(sample).reset_index(drop=True)
    cat = categorical_features(sample).reset_index(drop=True)
    numeric_fg = pd.concat([keys, sample[[C.DAY_INDEX]].reset_index(drop=True), num], axis=1)
    categorical_fg = pd.concat([keys, cat], axis=1)
    label_fg = pd.concat([keys, sample[[C.LABEL]].reset_index(drop=True).astype("int64")], axis=1)
    return {C.FG_NUMERIC: numeric_fg, C.FG_CATEGORICAL: categorical_fg, C.FG_LABEL: label_fg}


# ------------------------------------------------------------------ splits
def assign_chronological_split(n: int, fractions: Dict[str, float] = None,
                               order: List[str] = None) -> np.ndarray:
    """Contiguous, time-ordered split labels for n rows already sorted by time.

    Default: first 40% train, next 10% validation, next 10% test, last 40%
    production. Any rounding remainder goes to the final split.
    """
    fractions = fractions or C.SPLIT_FRACTIONS
    order = order or C.SPLIT_ORDER
    if abs(sum(fractions[s] for s in order) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    labels = np.empty(n, dtype=object)
    start = 0
    for i, name in enumerate(order):
        end = n if i == len(order) - 1 else start + int(round(fractions[name] * n))
        labels[start:end] = name
        start = end
    return labels


# ------------------------------------------------------ stateful features
class CategoricalEncoder:
    """Rare-level collapsing + frequency encoding, fitted on the training split only.

    * ``C{i}_enc``  : the hashed id if it appeared >= min_count times in training,
                      else RARE_ID (1). MISSING_ID (0) is preserved.
    * ``C{i}_freq`` : share of training rows carrying that hashed id (0.0 if unseen).
    """

    def __init__(self, min_count: int = C.RARE_MIN_COUNT):
        self.min_count = min_count
        self.counts_: Dict[str, Dict[int, int]] = {}
        self.n_train_: int = 0

    def fit(self, cat_hashed_train: pd.DataFrame) -> "CategoricalEncoder":
        self.n_train_ = len(cat_hashed_train)
        for c in C.CAT_COLS:
            vc = cat_hashed_train[f"{c}_hash"].value_counts()
            vc = vc[vc.index != C.MISSING_ID]
            self.counts_[c] = {int(k): int(v) for k, v in vc.items()}
        return self

    def transform(self, cat_hashed: pd.DataFrame) -> pd.DataFrame:
        if not self.counts_:
            raise RuntimeError("encoder is not fitted")
        out = pd.DataFrame(index=cat_hashed.index)
        for c in C.CAT_COLS:
            h = cat_hashed[f"{c}_hash"].astype("int64")
            counts = h.map(self.counts_[c]).fillna(0).astype("int64")
            keep = (counts >= self.min_count) | (h == C.MISSING_ID)
            out[f"{c}_enc"] = h.where(keep, C.RARE_ID).astype("int64")
            out[f"{c}_freq"] = (counts / max(self.n_train_, 1)).astype("float64")
        return out

    def to_json(self) -> str:
        return json.dumps({"min_count": self.min_count, "n_train": self.n_train_,
                           "counts": {c: {str(k): v for k, v in d.items()}
                                      for c, d in self.counts_.items()}})

    @classmethod
    def from_json(cls, s: str) -> "CategoricalEncoder":
        obj = json.loads(s)
        enc = cls(min_count=obj["min_count"])
        enc.n_train_ = obj["n_train"]
        enc.counts_ = {c: {int(k): v for k, v in d.items()} for c, d in obj["counts"].items()}
        return enc


def model_feature_columns() -> List[str]:
    """Final model input columns, in the fixed order used for every CSV we write."""
    cols = list(C.NUM_COLS) + [f"{c}_missing" for c in C.NUM_COLS]
    for c in C.CAT_COLS:
        cols += [f"{c}_enc", f"{c}_freq"]
    return cols
