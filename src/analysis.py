"""Four exploratory analyses using PySpark and Spark SQL."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Analysis 1: Rating distribution by verified purchase status
# ---------------------------------------------------------------------------
def rating_by_verified(df: DataFrame) -> DataFrame:
    """GroupBy + agg: count reviews per rating and verified_purchase status."""
    return (
        df.groupBy("verified_purchase", "rating")
        .agg(F.count("*").alias("review_count"))
        .orderBy("verified_purchase", "rating")
    )


def rating_by_verified_sql(spark: SparkSession, df: DataFrame) -> DataFrame:
    """Spark SQL version with CASE WHEN for verified label."""
    df.createOrReplaceTempView("reviews")
    return spark.sql("""
        SELECT
            CASE WHEN verified_purchase = true THEN 'Verified'
                 ELSE 'Unverified'
            END AS purchase_type,
            rating,
            COUNT(*) AS review_count
        FROM reviews
        GROUP BY verified_purchase, rating
        ORDER BY purchase_type, rating
    """)


# ---------------------------------------------------------------------------
# Analysis 2: Helpfulness by price tier
# ---------------------------------------------------------------------------
def helpfulness_by_price_tier(df: DataFrame) -> DataFrame:
    """Aggregate helpfulness rate by price quartile (requires price_quartile col)."""
    return (
        df.groupBy("price_quartile")
        .agg(
            F.avg("is_helpful").alias("helpful_rate"),
            F.avg("helpful_vote").alias("avg_helpful_votes"),
            F.count("*").alias("review_count"),
        )
        .orderBy("price_quartile")
    )


# ---------------------------------------------------------------------------
# Analysis 3: Temporal review trends (uses CTE in Spark SQL)
# ---------------------------------------------------------------------------
def temporal_trends_sql(spark: SparkSession, df: DataFrame) -> DataFrame:
    """Spark SQL with CTE: monthly review counts with running average."""
    df.createOrReplaceTempView("reviews_dated")
    return spark.sql("""
        WITH monthly AS (
            SELECT
                review_year,
                review_month,
                CONCAT(review_year, '-', LPAD(review_month, 2, '0')) AS year_month,
                COUNT(*) AS monthly_count,
                AVG(CAST(is_helpful AS DOUBLE)) AS monthly_helpful_rate
            FROM reviews_dated
            GROUP BY review_year, review_month
        )
        SELECT
            year_month,
            monthly_count,
            monthly_helpful_rate,
            LAG(monthly_count, 1) OVER (ORDER BY year_month) AS prev_month_count,
            AVG(monthly_count) OVER (
                ORDER BY year_month
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ) AS rolling_3mo_avg
        FROM monthly
        ORDER BY year_month
    """)


# ---------------------------------------------------------------------------
# Analysis 4: Review text length vs. helpfulness
# ---------------------------------------------------------------------------
def helpfulness_by_text_length(df: DataFrame) -> DataFrame:
    """Aggregate helpfulness rate by text length bin."""
    return (
        df.groupBy("text_length_bin")
        .agg(
            F.avg("is_helpful").alias("helpful_rate"),
            F.avg("helpful_vote").alias("avg_helpful_votes"),
            F.avg("text_length").alias("avg_text_length"),
            F.count("*").alias("review_count"),
        )
        .orderBy("avg_text_length")
    )
