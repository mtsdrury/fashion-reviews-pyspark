"""MLlib TF-IDF pipeline and logistic regression classifier."""

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (
    IDF,
    CountVectorizer,
    StopWordsRemover,
    Tokenizer,
    VectorAssembler,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import RANDOM_SEED, TEST_RATIO, TRAIN_RATIO


def build_tfidf_pipeline(
    input_col: str = "text",
    num_features: int = 5000,
) -> Pipeline:
    """Build an MLlib Pipeline: Tokenizer -> StopWordsRemover -> CountVectorizer -> IDF."""
    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    cv = CountVectorizer(
        inputCol="filtered_words", outputCol="raw_features", vocabSize=num_features
    )
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    return Pipeline(stages=[tokenizer, remover, cv, idf])


def build_classification_pipeline(
    extra_feature_cols: list[str] | None = None,
    num_features: int = 5000,
) -> Pipeline:
    """Full pipeline: TF-IDF + optional numeric features -> LogisticRegression."""
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    cv = CountVectorizer(
        inputCol="filtered_words", outputCol="raw_features", vocabSize=num_features
    )
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")

    assembler_inputs = ["tfidf_features"]
    if extra_feature_cols:
        assembler_inputs.extend(extra_feature_cols)

    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="is_helpful",
        maxIter=20,
        regParam=0.01,
    )
    return Pipeline(stages=[tokenizer, remover, cv, idf, assembler, lr])


def prepare_for_classification(df: DataFrame) -> DataFrame:
    """Filter to rows with non-null text and cast label to double for MLlib."""
    return (
        df.filter(F.col("text").isNotNull())
        .filter(F.length(F.col("text")) > 0)
        .withColumn("is_helpful", F.col("is_helpful").cast("double"))
    )


def train_test_split(
    df: DataFrame,
    train_ratio: float = TRAIN_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[DataFrame, DataFrame]:
    """Split DataFrame into train and test sets."""
    return df.randomSplit([train_ratio, test_ratio], seed=seed)


def extract_top_tfidf_terms(model, n: int = 20) -> dict[str, list[str]]:
    """Extract vocabulary from a fitted TF-IDF pipeline model.

    Returns the top-n terms by IDF weight (rarest, most distinctive terms).
    """
    cv_model = model.stages[2]  # CountVectorizerModel
    vocab = cv_model.vocabulary

    idf_model = model.stages[3]  # IDFModel
    idf_values = idf_model.idf.toArray()

    indexed = sorted(enumerate(idf_values), key=lambda x: x[1], reverse=True)
    top_terms = [vocab[i] for i, _ in indexed[:n] if i < len(vocab)]
    return {"top_idf_terms": top_terms}
