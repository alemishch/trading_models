import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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
            if col.startswith("pca") or col.startswith("umap") or col == "datetime"
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    y_pred = lr.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression Performance:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  R^2 Score: {r2:.6f}")

    metrics_path = os.path.join(output_dir, f"linear_regression_{strategy}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Linear Regression Performance for {strategy}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
        f.write(f"R^2 Score: {r2:.6f}\n")

    return lr, scaler, mse, mae, r2


def perform_lightgbm_regression(df, strategy, feature_cols, output_dir):
    X = df[feature_cols]
    y = df[strategy]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    lgbm = lgb.LGBMRegressor(random_state=42)

    lgbm.fit(X_train, y_train)

    y_pred = lgbm.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"LightGBM Regression Performance:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
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
        f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
        f.write(f"R^2 Score: {r2:.6f}\n")
    print(f"LightGBM regression metrics saved to '{metrics_path}'.")

    return lgbm, mse, mae, r2, feature_importances_sorted


def compute_correlations(df, strategy, feature_cols):
    correlations = df[feature_cols].corrwith(df[strategy]).dropna()
    correlations_sorted = correlations.abs().sort_values(ascending=False)
    return correlations_sorted


def main():
    feature_dataset_path = "feature_datasets/combined_features_pca.csv"
    returns_path = "rets.csv"
    output_dir = "strategy_analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)

    df_features = load_feature_data(feature_dataset_path)
    df_rets = load_returns_data(returns_path)

    strategy_cols = [col for col in df_rets.columns if col != "datetime"]

    feature_cols = [col for col in df_features.columns if col != "datetime"]

    for strategy in tqdm(strategy_cols, desc="Analyzing Strategies"):
        print(f"\n===== Strategy: {strategy} =====")

        df_strategy_rets = df_rets[["datetime", strategy]].copy()

        df_merged = pd.merge(df_strategy_rets, df_features, on="datetime", how="inner")
        df_merged.dropna(inplace=True)
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


if __name__ == "__main__":
    main()
