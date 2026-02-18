"""Project configuration: paths, constants, and Spark settings."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Parquet file names
REVIEWS_PARQUET = RAW_DIR / "reviews.parquet"
METADATA_PARQUET = RAW_DIR / "metadata.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "reviews_clean.parquet"

# ---------------------------------------------------------------------------
# HuggingFace dataset identifiers
# ---------------------------------------------------------------------------
HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
HF_REVIEWS_CONFIG = "raw_review_Amazon_Fashion"
HF_META_CONFIG = "raw_meta_Amazon_Fashion"

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
USE_SAMPLE = True
SAMPLE_SIZE = 100_000

# ---------------------------------------------------------------------------
# Spark configuration
# ---------------------------------------------------------------------------
SPARK_APP_NAME = "fashion-reviews-pyspark"
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = "4g"

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
PRICE_QUANTILES = 4
TEXT_LENGTH_BINS = [0, 50, 150, 300, 500, 1000, float("inf")]
TEXT_LENGTH_LABELS = ["very_short", "short", "medium", "long", "very_long", "extremely_long"]
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
