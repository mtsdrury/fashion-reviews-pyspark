"""Tests for text_pipeline module."""

from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from src.text_pipeline import (
    build_classification_pipeline,
    build_tfidf_pipeline,
    prepare_for_classification,
    train_test_split,
)


def _sample_reviews(spark):
    schema = StructType([
        StructField("text", StringType()),
        StructField("is_helpful", IntegerType()),
        StructField("text_length", IntegerType()),
    ])
    data = [
        ("This shirt is amazing quality and fits perfectly", 1, 47),
        ("Terrible product fell apart after one wash", 0, 42),
        ("Love the color and the material is soft", 1, 39),
        ("Not worth the price too thin", 0, 28),
        ("Great jacket keeps me warm in winter", 1, 36),
        ("Runs small order a size up", 0, 26),
        ("Beautiful dress perfect for summer", 1, 34),
        ("Cheaply made would not recommend", 0, 32),
        ("Excellent quality I ordered two more", 1, 36),
        ("Disappointed with the stitching", 0, 31),
    ]
    return spark.createDataFrame(data, schema=schema)


def _sample_reviews_double(spark):
    schema = StructType([
        StructField("text", StringType()),
        StructField("is_helpful", DoubleType()),
    ])
    data = [
        ("This shirt is amazing quality and fits perfectly", 1.0),
        ("Terrible product fell apart after one wash", 0.0),
        ("Love the color and the material is soft", 1.0),
        ("Not worth the price too thin", 0.0),
        ("Great jacket keeps me warm in winter", 1.0),
        ("Runs small order a size up", 0.0),
        ("Beautiful dress perfect for summer", 1.0),
        ("Cheaply made would not recommend", 0.0),
        ("Excellent quality I ordered two more", 1.0),
        ("Disappointed with the stitching", 0.0),
    ]
    return spark.createDataFrame(data, schema=schema)


class TestBuildTfidfPipeline:
    def test_pipeline_has_four_stages(self, spark):
        pipeline = build_tfidf_pipeline()
        assert len(pipeline.getStages()) == 4

    def test_pipeline_fits_and_transforms(self, spark):
        df = _sample_reviews(spark)
        pipeline = build_tfidf_pipeline()
        model = pipeline.fit(df)
        result = model.transform(df)
        assert "tfidf_features" in result.columns
        assert result.count() == 10


class TestBuildClassificationPipeline:
    def test_pipeline_has_six_stages(self, spark):
        pipeline = build_classification_pipeline()
        assert len(pipeline.getStages()) == 6

    def test_pipeline_with_extra_features(self, spark):
        pipeline = build_classification_pipeline(extra_feature_cols=["text_length"])
        assert len(pipeline.getStages()) == 6

    def test_pipeline_trains_and_predicts(self, spark):
        df = _sample_reviews_double(spark)
        pipeline = build_classification_pipeline()
        model = pipeline.fit(df)
        preds = model.transform(df)
        assert "prediction" in preds.columns
        assert preds.count() == 10


class TestPrepareForClassification:
    def test_filters_null_text(self, spark):
        schema = StructType([
            StructField("text", StringType()),
            StructField("is_helpful", IntegerType()),
        ])
        data = [("Good shirt", 1), (None, 0), ("", 1), ("Nice", 0)]
        df = spark.createDataFrame(data, schema=schema)
        result = prepare_for_classification(df)
        assert result.count() == 2

    def test_casts_label_to_double(self, spark):
        schema = StructType([
            StructField("text", StringType()),
            StructField("is_helpful", IntegerType()),
        ])
        data = [("Good shirt", 1)]
        df = spark.createDataFrame(data, schema=schema)
        result = prepare_for_classification(df)
        assert result.schema["is_helpful"].dataType == DoubleType()


class TestTrainTestSplit:
    def test_split_preserves_total(self, spark):
        schema = StructType([StructField("id", IntegerType())])
        data = [(i,) for i in range(100)]
        df = spark.createDataFrame(data, schema=schema)
        train, test = train_test_split(df, 0.8, 0.2, seed=42)
        assert train.count() + test.count() == 100
