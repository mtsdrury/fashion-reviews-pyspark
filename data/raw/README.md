# Data Download

This project uses the **Amazon Reviews 2023** dataset from the McAuley Lab.

## Automatic Download

Run the data loader from the project root:

```python
from src.data_loader import download_reviews, download_metadata

download_reviews()   # saves to data/raw/reviews.parquet
download_metadata()  # saves to data/raw/metadata.parquet
```

## Dataset Details

- **Source:** [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Configs used:** `raw_review_Amazon_Fashion`, `raw_meta_Amazon_Fashion`
- **Size:** ~2.5M reviews + product metadata
- **License:** See the HuggingFace dataset card for terms

Parquet files are gitignored. Each collaborator downloads their own copy.
