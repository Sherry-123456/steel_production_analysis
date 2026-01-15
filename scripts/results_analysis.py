import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def plot_learning_curve(model, X, y, model_name,
                        cv=5,
                        train_sizes=np.linspace(0.1, 1.0, 5),
                        save_dir="figures/results"):
    base = Path(__file__).resolve().parents[1]
    save_path = base / save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
        train_sizes=train_sizes,
        n_jobs=-1,
    )

    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    val_rmse = np.sqrt(-val_scores.mean(axis=1))

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_rmse, marker="o", label="Training RMSE")
    plt.plot(train_sizes, val_rmse, marker="s", label="Validation RMSE")
    plt.xlabel("Training Samples")
    plt.ylabel("RMSE")
    plt.title(f"Learning Curve - {model_name}")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path / f"{model_name.lower()}_learning_curve.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_predictions_vs_actual(y_true, y_pred, model_name,
                               save_dir="figures/results"):
    base = Path(__file__).resolve().parents[1]
    save_path = base / save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], "--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}: Predicted vs Actual")

    plt.savefig(save_path / f"{model_name.lower()}_pred_vs_actual.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, model_name,
                   save_dir="figures/results"):
    base = Path(__file__).resolve().parents[1]
    save_path = base / save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot - {model_name}")
    plt.grid(True)

    plt.savefig(save_path / f"{model_name.lower()}_residuals.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_comparison_with_error(results,
                                     save_dir="figures/results"):
    base = Path(__file__).resolve().parents[1]
    save_path = base / save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results).T

    plt.figure(figsize=(8, 5))
    plt.bar(df.index, df["RMSE_mean"],
            yerr=df["RMSE_std"], capsize=6)
    plt.ylabel("RMSE")
    plt.title("Model Comparison with Error Bars")
    plt.xticks(rotation=45)

    plt.savefig(save_path / "model_comparison_with_error.png",
                dpi=300, bbox_inches="tight")
    plt.close()




