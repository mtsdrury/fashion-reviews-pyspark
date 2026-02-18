"""Tests for evaluation module."""

from pyspark.sql.types import DoubleType, StructField, StructType

from src.evaluation import evaluate_binary_classification, get_confusion_counts


def _mock_predictions(spark):
    """Create a predictions DataFrame mimicking MLlib output."""
    schema = StructType([
        StructField("is_helpful", DoubleType()),
        StructField("prediction", DoubleType()),
        StructField("rawPrediction", DoubleType()),
        StructField("probability", DoubleType()),
    ])
    # 3 TP, 2 TN, 1 FP, 1 FN
    data = [
        (1.0, 1.0, 0.8, 0.8),
        (1.0, 1.0, 0.7, 0.7),
        (1.0, 1.0, 0.6, 0.6),
        (0.0, 0.0, 0.3, 0.3),
        (0.0, 0.0, 0.2, 0.2),
        (0.0, 1.0, 0.55, 0.55),  # FP
        (1.0, 0.0, 0.4, 0.4),  # FN
    ]
    return spark.createDataFrame(data, schema=schema)


def _vector_predictions(spark):
    """Create predictions with proper vector columns for BinaryClassificationEvaluator."""
    from pyspark.ml.linalg import Vectors

    data = [
        (1.0, 1.0, Vectors.dense([-0.8, 0.8]), Vectors.dense([0.2, 0.8])),
        (1.0, 1.0, Vectors.dense([-0.7, 0.7]), Vectors.dense([0.3, 0.7])),
        (1.0, 1.0, Vectors.dense([-0.6, 0.6]), Vectors.dense([0.4, 0.6])),
        (0.0, 0.0, Vectors.dense([0.7, -0.7]), Vectors.dense([0.7, 0.3])),
        (0.0, 0.0, Vectors.dense([0.8, -0.8]), Vectors.dense([0.8, 0.2])),
        (0.0, 1.0, Vectors.dense([-0.1, 0.1]), Vectors.dense([0.45, 0.55])),
        (1.0, 0.0, Vectors.dense([0.1, -0.1]), Vectors.dense([0.6, 0.4])),
    ]
    return spark.createDataFrame(
        data, ["is_helpful", "prediction", "rawPrediction", "probability"]
    )


class TestEvaluateBinaryClassification:
    def test_returns_all_metrics(self, spark):
        df = _vector_predictions(spark)
        metrics = evaluate_binary_classification(df)
        expected_keys = {
            "auc_roc", "auc_pr", "accuracy",
            "weighted_precision", "weighted_recall", "f1",
        }
        assert set(metrics.keys()) == expected_keys

    def test_metrics_in_valid_range(self, spark):
        df = _vector_predictions(spark)
        metrics = evaluate_binary_classification(df)
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of range"

    def test_accuracy_value(self, spark):
        df = _vector_predictions(spark)
        metrics = evaluate_binary_classification(df)
        # 5 correct out of 7
        assert abs(metrics["accuracy"] - 5 / 7) < 0.01


class TestGetConfusionCounts:
    def test_counts_correct(self, spark):
        df = _vector_predictions(spark)
        counts = get_confusion_counts(df)
        assert counts["tp"] == 3
        assert counts["tn"] == 2
        assert counts["fp"] == 1
        assert counts["fn"] == 1

    def test_total_equals_row_count(self, spark):
        df = _vector_predictions(spark)
        counts = get_confusion_counts(df)
        total = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
        assert total == 7
