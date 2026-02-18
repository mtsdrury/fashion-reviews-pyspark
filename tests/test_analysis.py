"""Tests for analysis module."""

from pyspark.sql.types import (
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.analysis import (
    helpfulness_by_price_tier,
    helpfulness_by_text_length,
    rating_by_verified,
    rating_by_verified_sql,
    temporal_trends_sql,
)


def _reviews_df(spark):
    schema = StructType([
        StructField("verified_purchase", BooleanType()),
        StructField("rating", IntegerType()),
        StructField("is_helpful", IntegerType()),
        StructField("helpful_vote", IntegerType()),
    ])
    data = [
        (True, 5, 1, 2),
        (True, 5, 0, 0),
        (True, 4, 1, 1),
        (False, 1, 0, 0),
        (False, 3, 1, 3),
    ]
    return spark.createDataFrame(data, schema=schema)


class TestRatingByVerified:
    def test_returns_correct_groups(self, spark):
        df = _reviews_df(spark)
        result = rating_by_verified(df)
        assert result.count() >= 4  # at least 4 distinct (verified, rating) combos

    def test_sql_version_has_purchase_type(self, spark):
        df = _reviews_df(spark)
        result = rating_by_verified_sql(spark, df)
        assert "purchase_type" in result.columns
        types = {r["purchase_type"] for r in result.collect()}
        assert types == {"Verified", "Unverified"}


class TestHelpfulnessByPriceTier:
    def test_aggregates_by_quartile(self, spark):
        schema = StructType([
            StructField("price_quartile", IntegerType()),
            StructField("is_helpful", IntegerType()),
            StructField("helpful_vote", IntegerType()),
        ])
        data = [
            (1, 1, 3),
            (1, 0, 0),
            (2, 1, 1),
            (2, 1, 2),
        ]
        df = spark.createDataFrame(data, schema=schema)
        result = helpfulness_by_price_tier(df).collect()
        assert len(result) == 2
        q1 = [r for r in result if r["price_quartile"] == 1][0]
        assert abs(q1["helpful_rate"] - 0.5) < 0.01


class TestTemporalTrendsSql:
    def test_returns_rolling_avg(self, spark):
        schema = StructType([
            StructField("review_year", IntegerType()),
            StructField("review_month", IntegerType()),
            StructField("is_helpful", IntegerType()),
        ])
        data = [
            (2023, 1, 1),
            (2023, 1, 0),
            (2023, 2, 1),
            (2023, 3, 1),
            (2023, 3, 0),
            (2023, 3, 1),
        ]
        df = spark.createDataFrame(data, schema=schema)
        result = temporal_trends_sql(spark, df).collect()
        assert len(result) == 3
        assert result[0]["prev_month_count"] is None  # first month has no lag
        assert result[0]["rolling_3mo_avg"] is not None


class TestHelpfulnessByTextLength:
    def test_aggregates_by_bin(self, spark):
        schema = StructType([
            StructField("text_length_bin", StringType()),
            StructField("is_helpful", IntegerType()),
            StructField("helpful_vote", IntegerType()),
            StructField("text_length", IntegerType()),
        ])
        data = [
            ("short", 1, 2, 80),
            ("short", 0, 0, 90),
            ("medium", 1, 1, 200),
            ("long", 1, 5, 400),
        ]
        df = spark.createDataFrame(data, schema=schema)
        result = helpfulness_by_text_length(df).collect()
        assert len(result) == 3
        bins = {r["text_length_bin"] for r in result}
        assert bins == {"short", "medium", "long"}
