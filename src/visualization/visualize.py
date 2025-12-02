# src/visualization/visualize.py (FIXED COMPLETE VERSION)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def _savefig(fig, name: str):
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(out)

# ---------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------
def heatmap_correlation(df: pd.DataFrame, name: str = "heatmap_correlation"):
    corr = df.corr(numeric_only=True)
    if corr.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(6, corr.shape[0] * 0.4), max(4, corr.shape[1] * 0.4)))
    cax = ax.imshow(corr, interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    return _savefig(fig, name)

# ---------------------------------------------------------
# Feature Importance Plot
# ---------------------------------------------------------
def feature_importance_plot(feature_names: List[str], importances: List[float], name: str = "feature_importance"):
    if len(importances) == 0:
        return None
    idx = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in idx]
    vals = [importances[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.25)))
    ax.barh(range(len(names)), vals[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances")
    return _savefig(fig, name)

# ---------------------------------------------------------
# Histograms per Feature
# ---------------------------------------------------------
def histograms_per_feature(df: pd.DataFrame, name: str = "histograms"):
    paths = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(df[col].dropna(), bins=30, alpha=0.8)
        ax.set_title(f"Histogram: {col}")
        ax.set_ylabel("Count")
        paths.append(_savefig(fig, f"{name}_{col}"))
    return paths

# ---------------------------------------------------------
# Pairwise Scatter
# ---------------------------------------------------------
def pairwise_scatter(df: pd.DataFrame, cols: List[str] = None, name: str = "pairwise_scatter"):
    numeric = list(df.select_dtypes(include=[np.number]).columns)
    if len(numeric) < 2:
        return None
    if cols is None:
        cols = numeric[:6]

    axes = pd.plotting.scatter_matrix(df[cols], figsize=(len(cols) * 1.5, len(cols) * 1.5), diagonal="hist")
    fig = plt.gcf()
    return _savefig(fig, name)

# ---------------------------------------------------------
# Scatter Feature vs Target
# ---------------------------------------------------------
def scatter_feature_vs_target(df: pd.DataFrame, target_col: str, features: List[str] = None, name_prefix: str = "scatter"):
    paths = []
    numeric = df.select_dtypes(include=[np.number]).columns

    if features is None:
        features = [c for c in numeric if c != target_col]

    for feat in features:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(df[feat], df[target_col], alpha=0.7, s=20)
        ax.set_xlabel(feat)
        ax.set_ylabel(target_col)
        ax.set_title(f"{feat} vs {target_col}")
        paths.append(_savefig(fig, f"{name_prefix}_{feat}_vs_{target_col}"))
    return paths

# ---------------------------------------------------------
# Distribution Plot
# ---------------------------------------------------------
def distribution_plot(df: pd.DataFrame, col: str, name: str = None):
    name = name or f"dist_{col}"
    fig, ax = plt.subplots(figsize=(6, 3))
    data = df[col].dropna()
    ax.hist(data, bins=30, density=True, alpha=0.6)

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 200)
        ax.plot(xs, kde(xs))
    except Exception:
        pass

    ax.set_title(f"Distribution: {col}")
    return _savefig(fig, name)

# ---------------------------------------------------------
# Correlation Matrix Raw
# ---------------------------------------------------------
def correlation_matrix_values(df: pd.DataFrame, name: str = "correlation_matrix"):
    corr = df.corr(numeric_only=True)
    if corr.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(cax, ax=ax)
    return _savefig(fig, name)

# ---------------------------------------------------------
# Model Comparison Bar Chart — FIXED
# ---------------------------------------------------------
def model_comparison_bar(runs: List[Dict[str, Any]], metric_key="rmse"):
    labels = []
    values = []

    for r in runs:
        label = r.get("model_key") or r.get("model") or "unknown"
        val = r.get("metrics", {}).get(metric_key)
        labels.append(label)
        values.append(val if val is not None else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"Model Comparison by {metric_key.upper()}")
    plt.tight_layout()

    return _savefig(fig, f"model_comparison_{metric_key}")

# ---------------------------------------------------------
# Full Visualization Suite
# ---------------------------------------------------------
def full_report(df: pd.DataFrame, target_col: str, train_runs: List[Dict[str, Any]]):
    out = {}
    out["heatmap"] = heatmap_correlation(df)
    out["histograms"] = histograms_per_feature(df)
    out["pairwise"] = pairwise_scatter(df)
    out["scatter_vs_target"] = scatter_feature_vs_target(df, target_col)
    out["correlation_matrix"] = correlation_matrix_values(df)

    # choose proper metric automatically
    first_metrics = train_runs[0].get("metrics", {}) if train_runs else {}
    metric_choice = "rmse" if "rmse" in first_metrics else "f1"

    out["model_comparison"] = model_comparison_bar(train_runs, metric_key=metric_choice)
    return out
