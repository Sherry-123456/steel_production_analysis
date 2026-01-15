from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loading import load_data
from data_preprocessing import split_dataset




def save_plot(filename, subdir="figures/eda", dpi=300):
    project_root = Path(__file__).resolve().parents[1]
    save_dir = project_root / subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / filename
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {save_path}")




def plot_correlation_heatmap(df):
    """Plot correlation matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", square=True)
    plt.title("Feature Correlation Matrix")


def plot_feature_histograms(X, bins=20):
    """Plot histograms for all features."""
    n_features = X.shape[1]
    cols = int(np.ceil(np.sqrt(n_features)))
    rows = int(np.ceil(n_features / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(X.columns):
        axes[i].hist(X[col], bins=bins, edgecolor="black", alpha=0.7)
        axes[i].set_title(col)
        axes[i].grid(True, linestyle="--", alpha=0.5)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])


def plot_target_distribution(y):
    """Plot distribution of target variable."""
    plt.figure(figsize=(8, 6))
    sns.histplot(y, bins=20, kde=True)
    plt.title("Target Variable Distribution")
    plt.grid(True, linestyle="--", alpha=0.5)


def plot_boxplots(X):
    """Plot boxplots for numeric features."""
    plt.figure(figsize=(10, 6))
    X.boxplot(rot=45)
    plt.title("Boxplots of Features")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)


def plot_pairwise_relationships(df, features):
    """Plot pairwise feature relationships."""
    sns.pairplot(df[features], diag_kind="hist")
    plt.suptitle("Pairwise Feature Relationships", y=1.02)




def main():
    print("Starting EDA analysis...")

    # load data
    train_data = load_data("normalized_train_data.csv")
    test_data = load_data("normalized_test_data.csv")
    data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

    # split dataset (EDA only uses training set)
    X_train, _, _, y_train, _, _ = split_dataset(data)

    train_df = pd.concat([X_train, y_train], axis=1)

    # correlation matrix
    plot_correlation_heatmap(train_df)
    save_plot("correlation_matrix.png")

    # feature distributions
    plot_feature_histograms(X_train)
    save_plot("feature_distributions.png")

    # target distribution
    plot_target_distribution(y_train)
    save_plot("target_distribution.png")

    # boxplots
    plot_boxplots(X_train)
    save_plot("boxplots.png")

    # pair plots (select subset to avoid clutter)
    selected_features = list(X_train.columns[:4]) + ["output"]
    plot_pairwise_relationships(train_df, selected_features)
    save_plot("pairplot.png")

    print("EDA completed successfully.")


if __name__ == "__main__":
    main()




