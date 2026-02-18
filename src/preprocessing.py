"""Data cleaning, feature engineering, joins, and window functions."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from src.config import PRICE_QUANTILES, TEXT_LENGTH_BINS, TEXT_LENGTH_LABELS


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_reviews(df: DataFrame) -> DataFrame:
    """Drop nulls in key columns, cast types, add binary helpful label."""
    return (
        df.filter(F.col("text").isNotNull() & F.col("rating").isNotNull())
        .withColumn("helpful_vote", F.coalesce(F.col("helpful_vote"), F.lit(0)))
        .withColumn("is_helpful", F.when(F.col("helpful_vote") >= 1, 1).otherwise(0))
    )


def parse_timestamp(df: DataFrame, col_name: str = "timestamp") -> DataFrame:
    """Convert epoch-millisecond timestamp to date columns."""
    return (
        df.withColumn("review_date", F.from_unixtime(F.col(col_name) / 1000).cast("date"))
        .withColumn("review_year", F.year("review_date"))
        .withColumn("review_month", F.month("review_date"))
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_text_length(df: DataFrame) -> DataFrame:
    """Add review text character length."""
    return df.withColumn("text_length", F.length(F.col("text")))


def add_text_length_bin(df: DataFrame) -> DataFrame:
    """Bin text_length into categorical buckets using when/otherwise."""
    bins = TEXT_LENGTH_BINS
    labels = TEXT_LENGTH_LABELS
    expr = F.lit(None)
    for i in range(len(labels) - 1, -1, -1):
        expr = F.when(
            (F.col("text_length") >= bins[i]) & (F.col("text_length") < bins[i + 1]),
            F.lit(labels[i]),
        ).otherwise(expr)
    return df.withColumn("text_length_bin", expr)


def add_word_count(df: DataFrame) -> DataFrame:
    """Add word count feature."""
    return df.withColumn("word_count", F.size(F.split(F.col("text"), r"\s+")))


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------
def parse_price(meta_df: DataFrame) -> DataFrame:
    """Clean price string to numeric. Handles '$29.99' and null/empty."""
    return meta_df.withColumn(
        "price_numeric",
        F.regexp_replace(F.col("price"), r"[^0-9.]", "").cast(DoubleType()),
    )


def join_reviews_metadata(reviews_df: DataFrame, meta_df: DataFrame) -> DataFrame:
    """Left join reviews with metadata on parent_asin."""
    return reviews_df.join(meta_df, on="parent_asin", how="left")


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------
def add_price_quartile(df: DataFrame) -> DataFrame:
    """Add price quartile (1-4) using ntile window function."""
    window = Window.orderBy("price_numeric")
    return df.filter(F.col("price_numeric").isNotNull()).withColumn(
        "price_quartile", F.ntile(PRICE_QUANTILES).over(window)
    )


def add_monthly_review_trend(df: DataFrame) -> DataFrame:
    """Add running average of monthly review counts and lag using window functions."""
    monthly = (
        df.groupBy("review_year", "review_month")
        .agg(F.count("*").alias("monthly_count"))
        .withColumn(
            "year_month",
            F.concat_ws("-", F.col("review_year"), F.lpad(F.col("review_month"), 2, "0")),
        )
    )
    window = Window.orderBy("year_month")
    return monthly.withColumn(
        "prev_month_count", F.lag("monthly_count", 1).over(window)
    ).withColumn(
        "running_avg",
        F.avg("monthly_count").over(window.rowsBetween(Window.unboundedPreceding, 0)),
    )
