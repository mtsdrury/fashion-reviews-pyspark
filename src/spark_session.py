"""SparkSession factory."""

from pyspark.sql import SparkSession

from src.config import SPARK_APP_NAME, SPARK_DRIVER_MEMORY, SPARK_MASTER


def get_spark_session(
    app_name: str = SPARK_APP_NAME,
    master: str = SPARK_MASTER,
    driver_memory: str = SPARK_DRIVER_MEMORY,
) -> SparkSession:
    """Create or retrieve a SparkSession with project defaults."""
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
