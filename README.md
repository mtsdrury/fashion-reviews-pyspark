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
│   ├── processed/                 # Cleaned Parquet (gitignored)
│   └── tableau/                   # Aggregated CSVs for Tableau dashboard
├── src/
│   ├── config.py                  # Paths, constants, Spark config
│   ├── spark_session.py           # SparkSession factory
│   ├── data_loader.py             # HuggingFace download + Parquet I/O
│   ├── preprocessing.py           # Cleaning, feature eng, joins, windows
│   ├── analysis.py                # 4 exploratory analyses
│   ├── export_tableau_data.py     # CSV export for Tableau dashboard
│   ├── text_pipeline.py           # MLlib TF-IDF + classification
│   ├── evaluation.py              # Classifier metrics
│   └── visualization.py           # Matplotlib/Seaborn plots
├── tests/                         # Pytest suite (~25 tests, CI-safe)
├── notebooks/
│   └── analysis.ipynb             # End-to-end analysis notebook
└── results/
    └── figures/                   # Saved plots
```

## Tableau Dashboard

An interactive Tableau Public dashboard extends this analysis with 4 views: KPI scorecards with rating distribution, helpfulness drivers (text length, star rating, photos), price tier insights, and temporal trends. Built from aggregated CSVs exported by `src/export_tableau_data.py`.

**[View the dashboard on Tableau Public](https://public.tableau.com/app/profile/mackenzie.drury/viz/AmazonFashionReviewsWhatMakesaReviewHelpful/FashionReviewsIntelligenceDashboard)**

### Export CSVs for Tableau

```bash
python -m src.export_tableau_data
```

This runs the full pipeline (`USE_SAMPLE = False`) and writes 7 aggregated tables to `data/tableau/`:

| File | Rows | Description |
|------|------|-------------|
| `rating_by_verified.csv` | 10 | Rating distribution split by verified purchase |
| `helpfulness_by_price_tier.csv` | 4 | Helpful rate by price quartile |
| `temporal_trends.csv` | ~100 | Monthly review count with rolling 3-month average |
| `helpfulness_by_text_length.csv` | 6 | Helpful rate by text length bin |
| `helpfulness_by_rating.csv` | 5 | Helpful rate by star rating |
| `photo_impact.csv` | 2 | Photo vs. text-only review helpfulness |
| `yearly_satisfaction.csv` | ~11 | Average rating and volume by year (2013+) |

## Dataset

[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (Amazon Fashion category). Downloaded automatically via the HuggingFace `datasets` library. Parquet files are gitignored.

## Tech Stack

- **PySpark 3.5** (DataFrames, Spark SQL, MLlib)
- **Python 3.10+**
- **Tableau Public** (interactive dashboard)
- **Matplotlib / Seaborn** (static visualization)
- **Pytest** (testing)
- **GitHub Actions** (CI)
