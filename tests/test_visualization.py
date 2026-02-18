"""Tests for visualization module (smoke tests, no file I/O)."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.visualization import (
    plot_helpfulness_by_price,
    plot_helpfulness_by_text_length,
    plot_model_roc_data,
    plot_rating_distribution,
    plot_temporal_trends,
)


class TestPlotRatingDistribution:
    def test_returns_figure(self, spark):
        schema = StructType([
            StructField("rating", IntegerType()),
            StructField("review_count", IntegerType()),
            StructField("purchase_type", StringType()),
        ])
        data = [
            (5, 100, "Verified"),
            (4, 80, "Verified"),
            (5, 20, "Unverified"),
        ]
        df = spark.createDataFrame(data, schema=schema)
        fig = plot_rating_distribution(df, save=False)
        assert fig is not None


class TestPlotHelpfulnessByPrice:
    def test_returns_figure(self, spark):
        schema = StructType([
            StructField("price_quartile", IntegerType()),
            StructField("helpful_rate", DoubleType()),
        ])
        data = [(1, 0.3), (2, 0.4), (3, 0.5), (4, 0.6)]
        df = spark.createDataFrame(data, schema=schema)
        fig = plot_helpfulness_by_price(df, save=False)
        assert fig is not None


class TestPlotTemporalTrends:
    def test_returns_figure(self, spark):
        schema = StructType([
            StructField("year_month", StringType()),
            StructField("monthly_count", IntegerType()),
            StructField("rolling_3mo_avg", DoubleType()),
        ])
        data = [
            ("2023-01", 100, 100.0),
            ("2023-02", 120, 110.0),
            ("2023-03", 90, 103.3),
        ]
        df = spark.createDataFrame(data, schema=schema)
        fig = plot_temporal_trends(df, save=False)
        assert fig is not None


class TestPlotHelpfulnessByTextLength:
    def test_returns_figure(self, spark):
        schema = StructType([
            StructField("text_length_bin", StringType()),
            StructField("helpful_rate", DoubleType()),
        ])
        data = [("short", 0.2), ("medium", 0.4), ("long", 0.6)]
        df = spark.createDataFrame(data, schema=schema)
        fig = plot_helpfulness_by_text_length(df, save=False)
        assert fig is not None


class TestPlotRocCurve:
    def test_returns_figure(self):
        fig = plot_model_roc_data(
            fpr=[0.0, 0.2, 0.5, 1.0],
            tpr=[0.0, 0.6, 0.8, 1.0],
            auc_val=0.75,
            save=False,
        )
        assert fig is not None
