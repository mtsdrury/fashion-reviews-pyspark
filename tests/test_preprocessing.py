"""Tests for preprocessing module."""

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.preprocessing import (
    add_monthly_review_trend,
    add_price_quartile,
    add_text_length,
    add_text_length_bin,
    add_word_count,
    clean_reviews,
    join_reviews_metadata,
    parse_price,
    parse_timestamp,
)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
class TestCleanReviews:
    def test_drops_null_text(self, spark):
        schema = StructType([
            StructField("text", StringType()),
            StructField("rating", IntegerType()),
            StructField("helpful_vote", IntegerType()),
        ])
        data = [("Good", 5, 1), (None, 3, 0), ("Fine", 4, None)]
        df = spark.createDataFrame(data, schema=schema)
        result = clean_reviews(df)
        assert result.count() == 2

    def test_is_helpful_label(self, spark):
        schema = StructType([
            StructField("text", StringType()),
            StructField("rating", IntegerType()),
            StructField("helpful_vote", IntegerType()),
        ])
        data = [("Great", 5, 3), ("Bad", 1, 0)]
        df = spark.createDataFrame(data, schema=schema)
        result = clean_reviews(df).orderBy("rating").collect()
        assert result[0]["is_helpful"] == 0  # rating 1, helpful_vote 0
        assert result[1]["is_helpful"] == 1  # rating 5, helpful_vote 3

    def test_null_helpful_vote_becomes_zero(self, spark):
        schema = StructType([
            StructField("text", StringType()),
            StructField("rating", IntegerType()),
            StructField("helpful_vote", IntegerType()),
        ])
        data = [("Ok", 3, None)]
        df = spark.createDataFrame(data, schema=schema)
        result = clean_reviews(df).collect()
        assert result[0]["helpful_vote"] == 0
        assert result[0]["is_helpful"] == 0


class TestParseTimestamp:
    def test_extracts_year_month(self, spark):
        schema = StructType([StructField("timestamp", LongType())])
        # 2023-06-15 in epoch ms
        data = [(1686787200000,)]
        df = spark.createDataFrame(data, schema=schema)
        result = parse_timestamp(df).collect()[0]
        assert result["review_year"] == 2023
        assert result["review_month"] == 6


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
class TestTextFeatures:
    def test_text_length(self, spark):
        schema = StructType([StructField("text", StringType())])
        data = [("hello",), ("hi",)]
        df = spark.createDataFrame(data, schema=schema)
        result = add_text_length(df).collect()
        lengths = sorted([r["text_length"] for r in result])
        assert lengths == [2, 5]

    def test_text_length_bin(self, spark):
        schema = StructType([
            StructField("text", StringType()),
            StructField("text_length", IntegerType()),
        ])
        data = [("a", 10), ("b", 100), ("c", 200), ("d", 400), ("e", 800)]
        df = spark.createDataFrame(data, schema=schema)
        result = add_text_length_bin(df).orderBy("text_length").collect()
        assert result[0]["text_length_bin"] == "very_short"  # 10
        assert result[1]["text_length_bin"] == "short"  # 100
        assert result[2]["text_length_bin"] == "medium"  # 200
        assert result[3]["text_length_bin"] == "long"  # 400
        assert result[4]["text_length_bin"] == "very_long"  # 800

    def test_word_count(self, spark):
        schema = StructType([StructField("text", StringType())])
        data = [("one two three",), ("hello",)]
        df = spark.createDataFrame(data, schema=schema)
        result = add_word_count(df).collect()
        counts = sorted([r["word_count"] for r in result])
        assert counts == [1, 3]


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------
class TestJoins:
    def test_parse_price(self, spark):
        schema = StructType([StructField("price", StringType())])
        data = [("$29.99",), ("15.00",), (None,)]
        df = spark.createDataFrame(data, schema=schema)
        result = parse_price(df).orderBy("price_numeric").collect()
        assert result[0]["price_numeric"] is None
        assert abs(result[1]["price_numeric"] - 15.0) < 0.01
        assert abs(result[2]["price_numeric"] - 29.99) < 0.01

    def test_join_reviews_metadata(self, spark):
        reviews_schema = StructType([
            StructField("parent_asin", StringType()),
            StructField("rating", IntegerType()),
        ])
        meta_schema = StructType([
            StructField("parent_asin", StringType()),
            StructField("title", StringType()),
        ])
        reviews = spark.createDataFrame([("B001", 5), ("B002", 3)], schema=reviews_schema)
        meta = spark.createDataFrame([("B001", "Blue Shirt")], schema=meta_schema)
        result = join_reviews_metadata(reviews, meta).orderBy("parent_asin").collect()
        assert result[0]["title"] == "Blue Shirt"
        assert result[1]["title"] is None  # left join, no match


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------
class TestWindowFunctions:
    def test_price_quartile(self, spark):
        schema = StructType([StructField("price_numeric", DoubleType())])
        data = [(10.0,), (20.0,), (30.0,), (40.0,)]
        df = spark.createDataFrame(data, schema=schema)
        result = add_price_quartile(df).orderBy("price_numeric").collect()
        quartiles = [r["price_quartile"] for r in result]
        assert quartiles == [1, 2, 3, 4]

    def test_price_quartile_filters_null(self, spark):
        schema = StructType([StructField("price_numeric", DoubleType())])
        data = [(10.0,), (None,), (30.0,), (40.0,)]
        df = spark.createDataFrame(data, schema=schema)
        result = add_price_quartile(df)
        assert result.count() == 3

    def test_monthly_review_trend(self, spark):
        schema = StructType([
            StructField("review_year", IntegerType()),
            StructField("review_month", IntegerType()),
            StructField("review_id", StringType()),
        ])
        data = [
            (2023, 1, "r1"),
            (2023, 1, "r2"),
            (2023, 2, "r3"),
            (2023, 3, "r4"),
            (2023, 3, "r5"),
            (2023, 3, "r6"),
        ]
        df = spark.createDataFrame(data, schema=schema)
        result = add_monthly_review_trend(df).orderBy("year_month").collect()
        assert result[0]["monthly_count"] == 2  # Jan
        assert result[0]["prev_month_count"] is None  # no prior month
        assert result[1]["prev_month_count"] == 2  # Feb lag = Jan count
        assert result[2]["monthly_count"] == 3  # Mar
