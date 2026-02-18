"""Download Amazon Fashion data from HuggingFace and read/write Parquet."""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from src.config import (
    HF_META_CONFIG,
    HF_REPO,
    HF_REVIEWS_CONFIG,
    METADATA_PARQUET,
    REVIEWS_PARQUET,
    SAMPLE_SIZE,
    USE_SAMPLE,
)


def download_reviews(
    output_path: Path = REVIEWS_PARQUET,
    sample: bool = USE_SAMPLE,
    sample_size: int = SAMPLE_SIZE,
) -> Path:
    """Download fashion reviews from HuggingFace and save as Parquet."""
    from datasets import load_dataset

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(HF_REPO, HF_REVIEWS_CONFIG, split="full", trust_remote_code=True)
    if sample:
        ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    ds.to_parquet(str(output_path))
    return output_path


def download_metadata(
    output_path: Path = METADATA_PARQUET,
) -> Path:
    """Download fashion product metadata from HuggingFace and save as Parquet."""
    from datasets import load_dataset

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(HF_REPO, HF_META_CONFIG, split="full", trust_remote_code=True)
    ds.to_parquet(str(output_path))
    return output_path


def load_reviews_spark(spark: SparkSession, path: Path = REVIEWS_PARQUET) -> DataFrame:
    """Read reviews Parquet into a Spark DataFrame."""
    return spark.read.parquet(str(path))


def load_metadata_spark(spark: SparkSession, path: Path = METADATA_PARQUET) -> DataFrame:
    """Read metadata Parquet into a Spark DataFrame."""
    return spark.read.parquet(str(path))
