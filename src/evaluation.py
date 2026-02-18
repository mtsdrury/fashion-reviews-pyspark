"""Model evaluation metrics for the helpfulness classifier."""

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import DataFrame


def evaluate_binary_classification(predictions: DataFrame) -> dict[str, float]:
    """Compute AUC-ROC, AUC-PR, accuracy, precision, recall, and F1."""
    binary_eval = BinaryClassificationEvaluator(labelCol="is_helpful")

    auc_roc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})
    auc_pr = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderPR"})

    multi_eval = MulticlassClassificationEvaluator(
        labelCol="is_helpful", predictionCol="prediction"
    )
    accuracy = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})
    precision = multi_eval.evaluate(
        predictions, {multi_eval.metricName: "weightedPrecision"}
    )
    recall = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})
    f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "f1": f1,
    }


def get_confusion_counts(predictions: DataFrame) -> dict[str, int]:
    """Compute TP, TN, FP, FN from predictions."""
    rows = (
        predictions.selectExpr(
            "CAST(is_helpful AS INT) AS label",
            "CAST(prediction AS INT) AS pred",
        )
        .groupBy("label", "pred")
        .count()
        .collect()
    )
    counts = {(r["label"], r["pred"]): r["count"] for r in rows}
    return {
        "tp": counts.get((1, 1), 0),
        "tn": counts.get((0, 0), 0),
        "fp": counts.get((0, 1), 0),
        "fn": counts.get((1, 0), 0),
    }
