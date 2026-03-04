"""Export aggregated PySpark analysis results as CSVs for Tableau Public.

Runs the full pipeline (no sampling) and writes 7 small aggregated tables
to data/tableau/. These are summary tables (4-100 rows each), not raw data,
so Tableau loads them instantly.

Usage:
    python -m src.export_tableau_data
"""

from pyspark.sql import functions as F

from src.analysis import (
    helpfulness_by_price_tier,
    helpfulness_by_text_length,
    rating_by_verified,
    temporal_trends_sql,
)
from src.config import METADATA_PARQUET, REVIEWS_PARQUET, TABLEAU_DIR
from src.data_loader import (
    download_metadata,
    download_reviews,
    load_metadata_spark,
    load_reviews_spark,
)
from src.preprocessing import (
    add_price_quartile,
    add_text_length,
    add_text_length_bin,
    clean_reviews,
    join_reviews_metadata,
    parse_price,
    parse_timestamp,
)
from src.spark_session import get_spark_session


def _save_csv(df, name: str) -> None:
    """Convert a PySpark DataFrame to Pandas and save as a single CSV."""
    path = TABLEAU_DIR / f"{name}.csv"
    df.toPandas().to_csv(path, index=False)
    rows = df.count()
    print(f"  {name}.csv ({rows} rows)")


def main() -> None:
    print("Exporting Tableau CSVs (full dataset)...")
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)

    # Download data if not present.
    # NOTE: If the parquet was previously downloaded in sample mode, delete
    # data/raw/reviews.parquet and rerun to get the full ~2.5M dataset.
    if not REVIEWS_PARQUET.exists():
        print("Downloading reviews (full dataset)...")
        download_reviews(sample=False)
    if not METADATA_PARQUET.exists():
        print("Downloading metadata...")
        download_metadata()

    spark = get_spark_session()

    # Load and preprocess
    reviews_raw = load_reviews_spark(spark)
    metadata_raw = load_metadata_spark(spark)

    reviews = clean_reviews(reviews_raw)
    reviews = parse_timestamp(reviews)
    reviews = add_text_length(reviews)
    reviews = add_text_length_bin(reviews)
    reviews.cache()

    metadata = parse_price(metadata_raw)
    reviews_with_meta = join_reviews_metadata(reviews, metadata)
    reviews_with_price = add_price_quartile(reviews_with_meta)

    total = reviews.count()
    print(f"Loaded {total:,} reviews\n")

    # --- Export 1: Rating by verified purchase ---
    _save_csv(rating_by_verified(reviews), "rating_by_verified")

    # --- Export 2: Helpfulness by price tier ---
    _save_csv(helpfulness_by_price_tier(reviews_with_price), "helpfulness_by_price_tier")

    # --- Export 3: Temporal trends ---
    _save_csv(temporal_trends_sql(spark, reviews), "temporal_trends")

    # --- Export 4: Helpfulness by text length ---
    _save_csv(helpfulness_by_text_length(reviews), "helpfulness_by_text_length")

    # --- Export 5: Helpfulness by rating (from notebook section 8a) ---
    helpfulness_by_rating = (
        reviews.groupBy("rating")
        .agg(
            F.avg("is_helpful").alias("helpful_rate"),
            F.avg("helpful_vote").alias("avg_helpful_votes"),
            F.count("*").alias("review_count"),
        )
        .orderBy("rating")
    )
    _save_csv(helpfulness_by_rating, "helpfulness_by_rating")

    # --- Export 6: Photo impact (from notebook section 8b) ---
    reviews_with_images = reviews.withColumn(
        "has_images",
        F.when(F.size(F.col("images")) > 0, "With Photos").otherwise("Text Only"),
    )
    photo_impact = (
        reviews_with_images.groupBy("has_images")
        .agg(
            F.avg("is_helpful").alias("helpful_rate"),
            F.avg("helpful_vote").alias("avg_helpful_votes"),
            F.count("*").alias("review_count"),
        )
        .orderBy("has_images")
    )
    _save_csv(photo_impact, "photo_impact")

    # --- Export 7: Yearly satisfaction (from notebook section 8c) ---
    yearly_satisfaction = (
        reviews.filter(F.col("review_year") >= 2013)
        .groupBy("review_year")
        .agg(
            F.avg("rating").alias("avg_rating"),
            F.count("*").alias("review_count"),
        )
        .orderBy("review_year")
    )
    _save_csv(yearly_satisfaction, "yearly_satisfaction")

    spark.stop()
    print(f"\nDone. 7 CSVs written to {TABLEAU_DIR}")


if __name__ == "__main__":
    main()
