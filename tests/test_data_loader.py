"""Tests for data_loader module."""

import tempfile
from pathlib import Path

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from src.data_loader import load_metadata_spark, load_reviews_spark


def _write_parquet(spark, data, schema, path):
    """Helper: write a small DataFrame to Parquet."""
    df = spark.createDataFrame(data, schema=schema)
    df.write.mode("overwrite").parquet(str(path))
    return path


class TestLoadReviewsSpark:
    """Tests for reading reviews from Parquet."""

    def test_returns_dataframe_with_correct_columns(self, spark):
        schema = StructType([
            StructField("parent_asin", StringType()),
            StructField("rating", IntegerType()),
            StructField("text", StringType()),
            StructField("helpful_vote", IntegerType()),
        ])
        data = [
            ("B001", 5, "Great product", 3),
            ("B002", 1, "Terrible", 0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reviews.parquet"
            _write_parquet(spark, data, schema, path)
            df = load_reviews_spark(spark, path)
            assert df.count() == 2
            assert set(df.columns) == {"parent_asin", "rating", "text", "helpful_vote"}

    def test_preserves_values(self, spark):
        schema = StructType([
            StructField("parent_asin", StringType()),
            StructField("rating", IntegerType()),
        ])
        data = [("B001", 5), ("B002", 3)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reviews.parquet"
            _write_parquet(spark, data, schema, path)
            rows = load_reviews_spark(spark, path).collect()
            ratings = sorted([r["rating"] for r in rows])
            assert ratings == [3, 5]


class TestLoadMetadataSpark:
    """Tests for reading metadata from Parquet."""

    def test_returns_dataframe(self, spark):
        schema = StructType([
            StructField("parent_asin", StringType()),
            StructField("title", StringType()),
            StructField("price", StringType()),
        ])
        data = [("B001", "Blue Shirt", "29.99"), ("B002", "Red Hat", "15.00")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.parquet"
            _write_parquet(spark, data, schema, path)
            df = load_metadata_spark(spark, path)
            assert df.count() == 2
            assert "title" in df.columns

    def test_empty_parquet(self, spark):
        schema = StructType([
            StructField("parent_asin", StringType()),
            StructField("price", StringType()),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.parquet"
            _write_parquet(spark, [], schema, path)
            df = load_metadata_spark(spark, path)
            assert df.count() == 0
