import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from data_loading import load_data
from data_preprocessing import split_dataset, preprocess_data, scale_features
from results_analysis import (
    calculate_metrics,
    plot_learning_curve,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_model_comparison_with_error,
)


def ensure_dirs():
    base = Path(__file__).resolve().parents[1]
    (base / "results/models").mkdir(parents=True, exist_ok=True)
    (base / "results/logs").mkdir(parents=True, exist_ok=True)
    (base / "figures/results").mkdir(parents=True, exist_ok=True)
    return base


def log_results(text, filename="training_log.txt"):
    base = Path(__file__).resolve().parents[1]
    with open(base / "results/logs" / filename, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def get_model_configs():
    return {
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 20],
                "min_samples_split": [2, 5],
            },
        ),
        "SVR": (
            SVR(),
            {
                "C": [0.5, 1.0, 10],
                "gamma": ["scale", 0.1],
            },
        ),
        "MLP": (
            MLPRegressor(max_iter=2000, random_state=42),
            {
                "hidden_layer_sizes": [(256, 128), (420, 210)],
                "alpha": [0.001, 0.01],
            },
        ),
        "GPR": (
            GaussianProcessRegressor(kernel=RBF(length_scale=10)),
            {
                "alpha": [1e-2, 5e-2],
            },
        ),
    }


def train_and_validate(name, model, params, X_train, y_train, X_val, y_val):
    print(f"\nTraining {name}...")

    search = GridSearchCV(
        model,
        param_grid=params,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    start = time.perf_counter()
    search.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    best_model = search.best_estimator_

    y_val_pred = best_model.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)

    log_results(
        f"{name} | Best params: {search.best_params_} | "
        f"Val metrics: {val_metrics} | Train time: {train_time:.3f}s"
    )

    return best_model


def main():
    base = ensure_dirs()

    print("Loading data...")
    df = load_data("cleaned_train_data.csv")

    print("Preprocessing data...")
    df_clean = preprocess_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df_clean)
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

    models = get_model_configs()
    best_models = {}
    comparison_results = {}

    # ========= 模型训练 =========
    for name, (model, params) in models.items():
        best_model = train_and_validate(
            name, model, params, X_train, y_train, X_val, y_val
        )
        best_models[name] = best_model

    # ========= 学习曲线 =========
    X_curve = np.vstack([X_train, X_val])
    y_curve = np.hstack([y_train, y_val])

    print("\nGenerating learning curves...")
    for name, model in best_models.items():
        plot_learning_curve(model, X_curve, y_curve, name)

    # ========= 测试 & 可视化 =========
    print("\nEvaluating on test set...")
    for name, model in best_models.items():
        model.fit(X_curve, y_curve)

        y_test_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_test_pred)

        # 交叉验证误差条
        cv_scores = cross_val_score(
            model,
            X_curve,
            y_curve,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        rmse_scores = np.sqrt(-cv_scores)

        comparison_results[name] = {
            "RMSE_mean": rmse_scores.mean(),
            "RMSE_std": rmse_scores.std(),
        }

        plot_predictions_vs_actual(y_test, y_test_pred, name)
        plot_residuals(y_test, y_test_pred, name)

        joblib.dump(model, base / "results/models" / f"{name.lower()}_final.pkl")

    # ========= 模型对比柱状图 =========
    plot_model_comparison_with_error(comparison_results)

    print("\nAll models trained and visualizations generated successfully.")


if __name__ == "__main__":
    main()




