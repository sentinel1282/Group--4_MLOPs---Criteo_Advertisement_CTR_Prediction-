"""Read/write helpers that target S3 in SageMaker, or a local folder for offline testing.

Set ``CRITEO_LOCAL_ROOT=/some/dir`` to redirect every S3 key to a path under that
directory. This exists so the EDA and split logic can be exercised without AWS;
it is not used when the notebooks run in SageMaker.
"""
from __future__ import annotations

import glob
import io
import json
import os
import shutil
import tempfile
from typing import Optional

import pandas as pd

from . import config as C

LOCAL_ROOT = os.environ.get("CRITEO_LOCAL_ROOT")

# Athena and Glue lower-case every column name; map back to the canonical names.
_CANONICAL = {c.lower(): c for c in
              [C.LABEL, C.RECORD_ID, C.EVENT_TIME, C.DAY_INDEX] + C.NUM_COLS + C.CAT_COLS
              + [f"{c}_missing" for c in C.NUM_COLS]
              + [f"{c}{s}" for c in C.CAT_COLS for s in ("_hash", "_enc", "_freq")]}


def canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _CANONICAL.get(c.lower(), c) for c in df.columns})


def lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.lower() for c in df.columns})


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


class Store:
    """Uniform put/get over S3 or a local directory."""

    def __init__(self, bucket: str, boto_session=None, local_root: Optional[str] = LOCAL_ROOT):
        self.bucket = bucket
        self.local_root = local_root
        self.s3 = None if local_root else boto_session.client("s3")

    # ------------------------------------------------------------ primitives
    def _local(self, key: str) -> str:
        path = os.path.join(self.local_root, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def upload_file(self, local_path: str, key: str) -> str:
        if self.local_root:
            shutil.copyfile(local_path, self._local(key))
        else:
            self.s3.upload_file(local_path, self.bucket, key)
        return s3_uri(self.bucket, key)

    def put_bytes(self, data: bytes, key: str) -> str:
        if self.local_root:
            with open(self._local(key), "wb") as f:
                f.write(data)
        else:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        return s3_uri(self.bucket, key)

    def put_json(self, obj, key: str) -> str:
        return self.put_bytes(json.dumps(obj, indent=2, default=str).encode(), key)

    def put_text(self, text: str, key: str) -> str:
        return self.put_bytes(text.encode(), key)

    def put_parquet(self, df: pd.DataFrame, key: str) -> str:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return self.put_bytes(buf.getvalue(), key)

    def list_keys(self, prefix: str):
        if self.local_root:
            root = os.path.join(self.local_root, prefix)
            return sorted(os.path.relpath(p, self.local_root)
                          for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                          if os.path.isfile(p))
        keys, token = [], None
        while True:
            kw = dict(Bucket=self.bucket, Prefix=prefix)
            if token:
                kw["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kw)
            keys += [o["Key"] for o in resp.get("Contents", [])]
            if not resp.get("IsTruncated"):
                return keys
            token = resp["NextContinuationToken"]

    def delete_prefix(self, prefix: str) -> int:
        keys = self.list_keys(prefix)
        if self.local_root:
            for k in keys:
                os.remove(os.path.join(self.local_root, k))
            return len(keys)
        for i in range(0, len(keys), 1000):
            self.s3.delete_objects(Bucket=self.bucket, Delete={
                "Objects": [{"Key": k} for k in keys[i:i + 1000]]})
        return len(keys)

    # ------------------------------------------------------------ datasets
    def read_parquet_prefix(self, prefix: str) -> pd.DataFrame:
        """Read every .parquet object under prefix; hive partitions (k=v/) become columns."""
        # Athena UNLOAD writes extension-less object names, so filter by exclusion.
        skip = (".json", ".csv", ".txt", ".metadata", ".png", "_SUCCESS", "/")
        keys = [k for k in self.list_keys(prefix) if not k.endswith(skip)]
        if not keys:
            raise FileNotFoundError(f"no parquet objects under {prefix}")
        frames = []
        for k in keys:
            if self.local_root:
                df = pd.read_parquet(os.path.join(self.local_root, k))
            else:
                body = self.s3.get_object(Bucket=self.bucket, Key=k)["Body"].read()
                df = pd.read_parquet(io.BytesIO(body))
            for part in os.path.relpath(k, prefix).split("/")[:-1]:
                if "=" in part:
                    col, val = part.split("=", 1)
                    df[col] = int(val) if val.lstrip("-").isdigit() else val
            frames.append(df)
        return canonical_columns(pd.concat(frames, ignore_index=True))


def default_data_dir() -> str:
    """Large local scratch space. On a SageMaker notebook instance ~/SageMaker is the EBS volume."""
    env = os.environ.get("CRITEO_DATA_DIR")
    if env:
        return env
    sm = os.path.expanduser("~/SageMaker")
    base = sm if os.path.isdir(sm) else (tempfile.gettempdir() if LOCAL_ROOT else os.path.expanduser("~"))
    return os.path.join(base, "criteo-data")
