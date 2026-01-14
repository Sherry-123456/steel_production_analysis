import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from data_loading import load_data
from data_preprocessing import split_dataset, preprocess_data, scale_features
from results_analysis import calculate_metrics


# ---------- 工具函数 ----------

def ensure_dirs():
    base = Path(__file__).resolve().parents[1]
    (base / "results/models").mkdir(parents=True, exist_ok=True)
    (base / "results/logs").mkdir(parents=True, exist_ok=True)
    return base


def log_results(text, filename="training_log.txt"):
    base = Path(__file__).resolve().parents[1]
    log_path = base / "results/logs" / filename
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ---------- 模型 + 参数空间 ----------

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


# ---------- 训练与验证 ----------

def train_and_validate(name, model, params, X_train, y_train, X_val, y_val):
    print(f"\nTraining {name} with cross-validation...")

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

    return best_model, val_metrics


# ---------- 主流程 ----------

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
    val_results = {}

    # 1. 模型选择（基于验证集）
    for name, (model, params) in models.items():
        best_model, metrics = train_and_validate(
            name, model, params, X_train, y_train, X_val, y_val
        )
        best_models[name] = best_model
        val_results[name] = metrics

    print("\nValidation results:")
    for k, v in val_results.items():
        print(k, v)

    # 2. 使用 Train + Val 重新训练 final model
    X_final = np.vstack([X_train, X_val])
    y_final = np.hstack([y_train, y_val])

    test_results = {}

    print("\nTraining final models and evaluating on test set...")
    for name, model in best_models.items():
        start = time.perf_counter()
        model.fit(X_final, y_final)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        y_test_pred = model.predict(X_test)
        infer_time = time.perf_counter() - start

        metrics = calculate_metrics(y_test, y_test_pred)
        metrics["Training_Time"] = train_time
        metrics["Inference_Time"] = infer_time

        test_results[name] = metrics

        joblib.dump(model, base / "results/models" / f"{name.lower()}_final.pkl")

    results_df = pd.DataFrame(test_results).T
    results_df.to_csv(base / "results/test_performance.csv")

    print("\nFinal test results:")
    print(results_df)
    print("\nModel training completed successfully.")


if __name__ == "__main__":
    main()



