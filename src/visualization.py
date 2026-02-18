"""Matplotlib/Seaborn plotting functions for analysis results."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame

from src.config import FIGURES_DIR


def _save_fig(fig: plt.Figure, name: str, output_dir: Path = FIGURES_DIR) -> Path:
    """Save figure to results/figures/ and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rating_distribution(df: DataFrame, save: bool = True) -> plt.Figure:
    """Bar plot: rating distribution split by verified purchase status."""
    pdf = df.toPandas()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=pdf, x="rating", y="review_count", hue="purchase_type", ax=ax)
    ax.set_title("Rating Distribution by Purchase Verification Status")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Reviews")
    if save:
        _save_fig(fig, "rating_by_verified")
    return fig


def plot_helpfulness_by_price(df: DataFrame, save: bool = True) -> plt.Figure:
    """Bar plot: helpfulness rate by price quartile."""
    pdf = df.toPandas()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=pdf, x="price_quartile", y="helpful_rate", ax=ax, color="steelblue")
    ax.set_title("Review Helpfulness Rate by Price Tier")
    ax.set_xlabel("Price Quartile (1=lowest, 4=highest)")
    ax.set_ylabel("Proportion Marked Helpful")
    if save:
        _save_fig(fig, "helpfulness_by_price")
    return fig


def plot_temporal_trends(df: DataFrame, save: bool = True) -> plt.Figure:
    """Line plot: monthly review count with rolling average."""
    pdf = df.toPandas()
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(pdf["year_month"], pdf["monthly_count"], alpha=0.3, label="Monthly Count")
    ax1.plot(pdf["year_month"], pdf["rolling_3mo_avg"], color="red", linewidth=2, label="3-Mo Avg")
    ax1.set_title("Monthly Review Volume and Trend")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Review Count")
    ax1.legend()
    # Rotate x labels for readability
    every_n = max(1, len(pdf) // 15)
    ax1.set_xticks(range(0, len(pdf), every_n))
    ax1.set_xticklabels(pdf["year_month"].iloc[::every_n], rotation=45, ha="right")
    fig.tight_layout()
    if save:
        _save_fig(fig, "temporal_trends")
    return fig


def plot_helpfulness_by_text_length(df: DataFrame, save: bool = True) -> plt.Figure:
    """Bar plot: helpfulness rate by text length bin."""
    pdf = df.toPandas()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=pdf, x="text_length_bin", y="helpful_rate", ax=ax, color="teal")
    ax.set_title("Review Helpfulness by Text Length")
    ax.set_xlabel("Text Length Category")
    ax.set_ylabel("Proportion Marked Helpful")
    if save:
        _save_fig(fig, "helpfulness_by_text_length")
    return fig


def plot_model_roc_data(fpr: list, tpr: list, auc_val: float, save: bool = True) -> plt.Figure:
    """ROC curve for the helpfulness classifier."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"Model (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_title("ROC Curve: Review Helpfulness Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    if save:
        _save_fig(fig, "roc_curve")
    return fig
