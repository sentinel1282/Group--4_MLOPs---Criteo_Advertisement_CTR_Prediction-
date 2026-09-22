"""Central configuration for the Criteo CTR MLOps project (AAI-540, Group 4).

Every notebook imports from here so that bucket names, column lists, and split
fractions are defined exactly once. Override any value with an environment
variable of the same name (e.g. ``export CRITEO_SAMPLE_EVERY=20``).
"""
import os

# ---------------------------------------------------------------- S3 layout
# Default bucket is resolved at runtime from the SageMaker session
# (sagemaker.Session().default_bucket()) unless CRITEO_BUCKET is set.
BUCKET = os.environ.get("CRITEO_BUCKET")  # None -> use SageMaker default bucket
PREFIX = os.environ.get("CRITEO_PREFIX", "criteo-ctr")

RAW_TRAIN_PREFIX = f"{PREFIX}/raw/train"            # full labelled train.txt (immutable)
RAW_TEST_PREFIX = f"{PREFIX}/raw/test_unlabeled"    # Kaggle test.txt (no labels; archived only)
SAMPLE_PREFIX = f"{PREFIX}/curated/sample"          # chronological systematic sample, Parquet
FEATURE_STORE_OFFLINE_PREFIX = f"{PREFIX}/feature-store"
SPLITS_PREFIX = f"{PREFIX}/splits"                  # train / validation / test / production
ARTIFACTS_PREFIX = f"{PREFIX}/artifacts"            # fitted encoders, manifests
ATHENA_RESULTS_PREFIX = f"{PREFIX}/athena-results"

# ------------------------------------------------------------------- Athena
ATHENA_DATABASE = os.environ.get("CRITEO_ATHENA_DB", "criteo_ctr_db")
RAW_TABLE = "raw_train"
SAMPLE_TABLE = "curated_sample"

# ----------------------------------------------------------------- Schema
LABEL = "label"
NUM_COLS = [f"I{i}" for i in range(1, 14)]   # 13 integer / count features
CAT_COLS = [f"C{i}" for i in range(1, 27)]   # 26 hashed categorical features
RAW_COLUMNS = [LABEL] + NUM_COLS + CAT_COLS  # order of fields in train.txt (tab-separated, no header)

RECORD_ID = "record_id"     # row position in the original chronological file
EVENT_TIME = "event_time"   # synthetic unix seconds derived from row position (see features.add_record_keys)
DAY_INDEX = "day_index"     # 0..6, which of the 7 collection days the row falls in

# Criteo documents the training file as chronologically ordered but ships no
# timestamp column; this anchor only makes event_time monotonic for Feature Store.
SYNTHETIC_START_EPOCH = 1_704_067_200   # 2024-01-01T00:00:00Z (nominal, NOT the real collection date)
COLLECTION_DAYS = 7

# --------------------------------------------------------------- Sampling
# Keep every k-th row of the 45.8M-row file. k=45 -> ~1.02M rows spanning all
# 7 days in order. A head() sample would cover only part of day 1 and destroy
# the chronological split, so systematic sampling is used instead.
SAMPLE_EVERY = int(os.environ.get("CRITEO_SAMPLE_EVERY", "45"))
READ_CHUNKSIZE = 1_000_000

# ------------------------------------------------------------------ Splits
# Chronological, contiguous, in this order. "production" is the most recent 40%
# and is held back to simulate live traffic for batch inference and monitoring.
SPLIT_ORDER = ["train", "validation", "test", "production"]
SPLIT_FRACTIONS = {"train": 0.40, "validation": 0.10, "test": 0.10, "production": 0.40}

# --------------------------------------------------------- Feature config
HASH_BUCKETS = 2 ** 20      # fixed-width space for hashed categorical ids
MISSING_ID = 0              # reserved id for missing categorical values
RARE_ID = 1                 # reserved id for levels rare/unseen in the training split
RARE_MIN_COUNT = int(os.environ.get("CRITEO_RARE_MIN_COUNT", "10"))

# ------------------------------------------------------------ Feature Store
FG_NUMERIC = "criteo-ctr-numeric-fg"
FG_CATEGORICAL = "criteo-ctr-categorical-fg"
FG_LABEL = "criteo-ctr-label-fg"
