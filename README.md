# Fashion Reviews PySpark Analysis

**What makes a fashion product review helpful on Amazon?**

A PySpark portfolio project analyzing ~2.5M Amazon Fashion reviews to identify what distinguishes helpful reviews from unhelpful ones. Uses the McAuley Lab Amazon Reviews 2023 dataset from HuggingFace.

## Research Question

Binary classification: reviews with 1+ helpful votes are "helpful," those with 0 are "not helpful." The analysis decomposes into exploratory analysis (what distinguishes helpful reviews?), feature engineering, and a predictive pipeline.

## Analyses

| # | Analysis | PySpark Features |
|---|----------|-----------------|
| 1 | Rating distribution by verified purchase status | `groupBy`, `agg`, Spark SQL (`GROUP BY`, `CASE WHEN`) |
| 2 | Helpfulness by price tier | Join (reviews + metadata), Window (`ntile` for price quartiles) |
| 3 | Temporal review trends | Window (`lag`, running avg), Spark SQL CTEs, date functions |
| 4 | Review text length vs. helpfulness | `withColumn`, `when`/`otherwise` binning |
| 5 | TF-IDF keyword analysis | MLlib Pipeline (Tokenizer, StopWordsRemover, CountVectorizer, IDF) |
| 6 | Helpfulness prediction model | MLlib (VectorAssembler, LogisticRegression, BinaryClassificationEvaluator) |

## Setup

```bash
pip install -r requirements.txt
```

**Java 11** is required for PySpark. Install via your package manager or [Adoptium](https://adoptium.net/).

## Usage

### Quick start (sample mode)

By default, `USE_SAMPLE = True` in `src/config.py` loads 100K rows for fast iteration:

```bash
jupyter notebook notebooks/analysis.ipynb
```

### Full dataset

Set `USE_SAMPLE = False` in `src/config.py`, then rerun the notebook. The HuggingFace download runs automatically on first execution.

### Run tests

```bash
pytest tests/ -v
```

### Lint

```bash
ruff check src/ tests/
```

## Project Structure

```
fashion-reviews-pyspark/
├── .github/workflows/ci.yml      # GitHub Actions: lint + test
├── pyproject.toml                 # Project config, ruff, pytest settings
├── requirements.txt
├── data/
│   ├── raw/README.md              # Download instructions
│   └── processed/                 # Cleaned Parquet (gitignored)
├── src/
│   ├── config.py                  # Paths, constants, Spark config
│   ├── spark_session.py           # SparkSession factory
│   ├── data_loader.py             # HuggingFace download + Parquet I/O
│   ├── preprocessing.py           # Cleaning, feature eng, joins, windows
│   ├── analysis.py                # 4 exploratory analyses
│   ├── text_pipeline.py           # MLlib TF-IDF + classification
│   ├── evaluation.py              # Classifier metrics
│   └── visualization.py           # Matplotlib/Seaborn plots
├── tests/                         # Pytest suite (~25 tests, CI-safe)
├── notebooks/
│   └── analysis.ipynb             # End-to-end analysis notebook
└── results/
    └── figures/                   # Saved plots
```

## Dataset

[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (Amazon Fashion category). Downloaded automatically via the HuggingFace `datasets` library. Parquet files are gitignored.

## Tech Stack

- **PySpark 3.5** (DataFrames, Spark SQL, MLlib)
- **Python 3.10+**
- **Matplotlib / Seaborn** (visualization)
- **Pytest** (testing)
- **GitHub Actions** (CI)
