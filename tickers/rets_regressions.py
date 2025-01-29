import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import lightgbm as lgb

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")


def load_feature_data(feature_path):
    df_features = pd.read_csv(feature_path, parse_dates=["datetime"])
    df_features = df_features[
        [
            col
            for col in df_features.columns
            if not col.startswith("umap") or col == "datetime"
        ]
    ]

    df_features.dropna(axis=1, how="all", inplace=True)

    return df_features


def load_returns_data(rets_path):
    df_rets = pd.read_csv(rets_path, parse_dates=["datetime"])

    if "datetime" not in df_rets.columns:
        raise ValueError("rets.csv must contain a 'datetime' column.")

    return df_rets


def compute_correlations(df, strategy, feature_cols):
    correlations = df[feature_cols].corrwith(df[strategy]).dropna()
    correlations_sorted = correlations.abs().sort_values(ascending=False)
    return correlations_sorted


def perform_linear_regression(df, strategy, feature_cols, output_dir):
    X = df[feature_cols].values
    y = df[strategy].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    train_size = len(y_train)
    test_size = len(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = Ridge(alpha=7.0)
    lr.fit(X_train_scaled, y_train)

    y_train_pred = lr.predict(X_train_scaled)
    r2_train = r2_score(y_train, y_train_pred)
    print("TRAIN R2: ", r2_train)

    y_pred = lr.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression Performance:")
    # print(f"  Mean Squared Error (MSE): {mse:.6f}")
    # print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  R^2 Score: {r2:.6f}")

    metrics_path = os.path.join(output_dir, f"linear_regression_{strategy}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Linear Regression Performance for {strategy}\n")
        f.write(f"Train Size: {train_size}\n")
        f.write(f"Test Size: {test_size}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
        f.write(f"R^2 Score: {r2:.6f}\n")

    return lr, scaler, mse, mae, r2


def perform_random_forest_regression(df, strategy, feature_cols, output_dir):
    """
    Perform Random Forest regression to predict strategy returns based on features.
    Split data into train and test sets, train the model, evaluate performance.
    Save the evaluation metrics and feature importances to files.
    """

    X = df[feature_cols].values
    y = df[strategy].values

    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    train_size = len(y_train)
    test_size = len(y_test)

    # Initialize the model with hyperparameters
    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    # Train the model
    rf_reg.fit(X_train, y_train)

    # Predict
    y_pred = rf_reg.predict(X_test)

    y_train_pred = rf_reg.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    print("TRAIN R2: ", r2_train)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest Regression Performance:")
    # print(f"  Mean Squared Error (MSE): {mse:.6f}")
    # print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  R^2 Score: {r2:.6f}")

    # Save Random Forest Regression Metrics
    metrics_path = os.path.join(
        output_dir, f"random_forest_regression_{strategy}_metrics.txt"
    )
    with open(metrics_path, "w") as f:
        f.write(f"Random Forest Regression Performance for {strategy}\n")
        f.write(f"Train Size: {train_size}\n")
        f.write(f"Test Size: {test_size}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
        f.write(f"R^2 Score: {r2:.6f}\n")

    return rf_reg, mse, mae, r2


def perform_random_forest_classifier(df, strategy, feature_cols, output_dir):
    """
    Perform Random Forest classification to predict if strategy returns are >0 or <=0.
    Split data into train and test sets, train the model, evaluate performance.
    Save the evaluation metrics and feature importances to files.
    """
    print(f"\nPerforming Random Forest Classification for strategy: {strategy}")

    X = df[feature_cols].values
    y = (df[strategy].values > 0).astype(int)  # Binary target

    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    train_size = len(y_train)
    test_size = len(y_test)

    # Initialize the model with hyperparameters
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    # Train the model
    rf_clf.fit(X_train, y_train)

    # Predict
    y_pred = rf_clf.predict(X_test)
    y_proba = rf_clf.predict_proba(X_test)[:, 1]

    y_train_pred = rf_clf.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    print("TRAIN ACCURACY: ", accuracy_train)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Random Forest Classification Performance:")
    print(f"  Accuracy: {accuracy:.6f}")
    # print(f"  Precision: {precision:.6f}")
    # print(f"  Recall: {recall:.6f}")
    # print(f"  F1 Score: {f1:.6f}")
    # print(f"  ROC AUC Score: {auc:.6f}")

    # Save Random Forest Classification Metrics
    metrics_path = os.path.join(
        output_dir, f"random_forest_classification_{strategy}_metrics.txt"
    )
    with open(metrics_path, "w") as f:
        f.write(f"Random Forest Classification Performance for {strategy}\n")
        f.write(f"Train Size: {train_size}\n")
        f.write(f"Test Size: {test_size}\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
        f.write(f"F1 Score: {f1:.6f}\n")
        f.write(f"ROC AUC Score: {auc:.6f}\n")
    print(f"Random Forest classification metrics saved to '{metrics_path}'.")

    return rf_clf, accuracy, precision, recall, f1, auc


def perform_lightgbm_regression(df, strategy, feature_cols, output_dir):
    X = df[feature_cols]
    y = df[strategy]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    train_size = len(y_train)
    test_size = len(y_test)

    lgbm = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=7,
        num_leaves=31,
        lambda_l1=0.1,
        lambda_l2=0.1,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
    )

    lgbm.fit(X_train, y_train)

    y_pred = lgbm.predict(X_test)

    y_train_pred = lgbm.predict(X_train)
    accuracy_train = r2_score(y_train, y_train_pred)
    print("TRAIN R2: ", accuracy_train)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"LightGBM Regression Performance:")
    # print(f"  Mean Squared Error (MSE): {mse:.6f}")
    # print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  R^2 Score: {r2:.6f}")

    # Feature Importance
    feature_importances = pd.Series(lgbm.feature_importances_, index=feature_cols)
    feature_importances_sorted = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=feature_importances_sorted.values[:20],
        y=feature_importances_sorted.index[:20],
    )
    plt.title(f"LightGBM Feature Importances for {strategy}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"feature_importances_{strategy}_lightgbm.png")
    plt.savefig(plot_path)
    plt.close()

    fi_df = feature_importances_sorted.reset_index()
    fi_df.columns = ["Feature", "Importance"]
    fi_csv_path = os.path.join(
        output_dir, f"feature_importances_{strategy}_lightgbm.csv"
    )
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"Feature importances saved to '{fi_csv_path}'.")

    metrics_path = os.path.join(output_dir, f"lightgbm_{strategy}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"LightGBM Regression Performance for {strategy}\n")
        f.write(f"Train Size: {train_size}\n")
        f.write(f"Test Size: {test_size}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
        f.write(f"R^2 Score: {r2:.6f}\n")
    print(f"LightGBM regression metrics saved to '{metrics_path}'.")

    return lgbm, mse, mae, r2, feature_importances_sorted


def perform_lightgbm_classification(df, strategy, feature_cols, output_dir):
    X = df[feature_cols].values
    y = (df[strategy].values > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    train_size = len(y_train)
    test_size = len(y_test)

    lgbm_clf = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=1,
        num_leaves=31,
        learning_rate=0.01,
        random_state=42,
        verbose=-1,
        lambda_l1=0.1,
        lambda_l2=0.2,
        min_child_samples=10,
        is_unbalance=True,
    )

    lgbm_clf.fit(X_train, y_train)

    y_pred = lgbm_clf.predict(X_test)
    y_proba = lgbm_clf.predict_proba(X_test)[:, 1]

    y_train_pred = lgbm_clf.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    print("TRAIN accuracy: ", accuracy_train)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"LightGBM Classification Performance:")
    print(f"  Accuracy: {accuracy:.6f}")
    # print(f"  Precision: {precision:.6f}")
    # print(f"  Recall: {recall:.6f}")
    # print(f"  F1 Score: {f1:.6f}")
    # print(f"  ROC AUC Score: {auc:.6f}")

    # Save LightGBM Classification Metrics
    metrics_path = os.path.join(
        output_dir, f"lightgbm_classification_{strategy}_metrics.txt"
    )
    with open(metrics_path, "w") as f:
        f.write(f"LightGBM Classification Performance for {strategy}\n")
        f.write(f"Train Size: {train_size}\n")
        f.write(f"Test Size: {test_size}\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
        f.write(f"F1 Score: {f1:.6f}\n")
        f.write(f"ROC AUC Score: {auc:.6f}\n")

    return lgbm_clf, accuracy, precision, recall, f1, auc


def compute_correlations(df, strategy, feature_cols):
    correlations = df[feature_cols].corrwith(df[strategy]).dropna()
    correlations_sorted = correlations.abs().sort_values(ascending=False)
    return correlations_sorted


def main():
    feature_dataset_path = "feature_datasets/combined_features_umap.csv"
    returns_path = "rets.csv"
    output_dir = "strategy_analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)

    df_features = load_feature_data(feature_dataset_path)
    df_rets = load_returns_data(returns_path)

    strategy_cols = [col for col in df_rets.columns if col != "datetime"]

    base_feature_cols = [col for col in df_features.columns if col != "datetime"]

    for strategy in tqdm(strategy_cols, desc="Analyzing Strategies"):
        print(f"\n===== Strategy: {strategy} =====")

        df_strategy_rets = df_rets[["datetime", strategy]].copy()
        df_strategy_rets.dropna(inplace=True)
        df_strategy_rets.sort_values("datetime", inplace=True)
        initial_count = len(df_strategy_rets)

        df_strategy_rets[f"{strategy}_lag_1"] = df_strategy_rets[strategy].shift(1)

        df_strategy_rets[f"{strategy}_avg_5"] = (
            df_strategy_rets[strategy].shift(1).rolling(window=5).mean()
        )

        engineered_features = [f"{strategy}_lag_1", f"{strategy}_avg_5"]
        feature_cols = base_feature_cols + engineered_features

        df_merged = pd.merge(df_strategy_rets, df_features, on="datetime", how="inner")
        df_merged.dropna(inplace=True)

        final_count = len(df_merged)
        dropped_percentage = ((initial_count - final_count) / initial_count) * 100
        # print(f"Dropped {dropped_percentage:.2f}% for {strategy}")

        if df_merged.empty:
            print(f"No data dropping NaNs for {strategy}")
            continue

        correlations = compute_correlations(df_merged, strategy, feature_cols)
        correlation_df = pd.DataFrame(
            {"Feature": correlations.index, "Correlation": correlations.values}
        )

        corr_out_path = os.path.join(output_dir, f"correlation_{strategy}.csv")
        correlation_df.to_csv(corr_out_path, index=False)

        lr_model, scaler, lr_mse, lr_mae, lr_r2 = perform_linear_regression(
            df_merged, strategy, feature_cols, output_dir
        )

        (
            lgbm_model,
            lgbm_mse,
            lgbm_mae,
            lgbm_r2,
            feature_importances,
        ) = perform_lightgbm_regression(df_merged, strategy, feature_cols, output_dir)

        (
            lgbm_clf_model,
            lgbm_clf_accuracy,
            lgbm_clf_precision,
            lgbm_clf_recall,
            lgbm_clf_f1,
            lgbm_clf_auc,
        ) = perform_lightgbm_classification(
            df_merged, strategy, feature_cols, output_dir
        )

        (
            rf_reg_model,
            rf_reg_mse,
            rf_reg_mae,
            rf_reg_r2,
        ) = perform_random_forest_regression(
            df_merged, strategy, feature_cols, output_dir
        )

        (
            rf_clf_model,
            rf_clf_accuracy,
            rf_clf_precision,
            rf_clf_recall,
            rf_clf_f1,
            rf_clf_auc,
        ) = perform_random_forest_classifier(
            df_merged, strategy, feature_cols, output_dir
        )


if __name__ == "__main__":
    main()
