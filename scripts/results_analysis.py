import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------- 指标计算 ----------

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression performance metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


# ---------- 结果可视化 ----------

def plot_predictions_vs_actual(y_true, y_pred, model_name, save_dir="figures/results"):
    """
    Scatter plot of predicted vs actual values.
    """
    base = Path(__file__).resolve().parents[1]
    save_path = base / save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             linestyle="--", color="red")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}: Predictions vs Actual")

    file_path = save_path / f"{model_name.lower()}_pred_vs_actual.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved prediction plot: {file_path}")


def plot_model_comparison(results, save_dir="figures/results"):
    """
    Bar plot comparing model RMSE.
    """
    base = Path(__file__).resolve().parents[1]
    save_path = base / save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results).T

    plt.figure(figsize=(8, 5))
    plt.bar(df.index, df["RMSE"])
    plt.ylabel("RMSE")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=45)

    file_path = save_path / "model_comparison_rmse.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved model comparison plot: {file_path}")




