import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

market_features = pd.read_csv(
    "all_features_BTCUSDT.csv", index_col=0, parse_dates=[0], low_memory=False
)
market_features = market_features.resample("D").ffill()
market_features = market_features.loc[:, market_features.isna().mean() <= 0.7]

strategies_data = {}
returns_features_path = "returns_features"
for file in os.listdir(returns_features_path):
    if file.endswith("_returns_features.csv"):
        strategy_name = file.replace("_returns_features.csv", "")
        df = pd.read_csv(
            os.path.join(returns_features_path, file),
            index_col=0,
            parse_dates=[0],
            low_memory=False,
        )
        df = df.resample("D").ffill()
        # df['Sharpe_Ratio_Target'] = ((df['Sharpe_Ratio_30'].shift(-1) > 0) | ((df['Sharpe_Ratio_30'].shift(-1) > df['Sharpe_Ratio_30']) & (df['Sharpe_Ratio_30']<0))).astype(int)
        df["Sharpe_Ratio_Target"] = (
            df["Sharpe_Ratio_30"].shift(-1) > df["Sharpe_Ratio_30"]
        ).astype(int)
        # df['Sharpe_Ratio_Target'] = (df['Mean_Returns_5'].shift(-1) >= df['Mean_Returns_5']).astype(int)
        # test = df[['Sharpe_Ratio_10','Sharpe_Ratio_Target']]
        df = df[:-1]
        strategies_data[strategy_name] = df

merged_data = {}
for strategy, df in strategies_data.items():
    merged_df = pd.merge(
        market_features, df, left_index=True, right_index=True, how="inner"
    )
    merged_df = merged_df.dropna(axis=1, thresh=0.5 * len(merged_df))
    merged_df = merged_df.dropna()
    merged_data[strategy] = merged_df

# frequency for walk-forward steps
train_end = "2023-12-31"
predict_start = "2024-01-01"
predict_end = "2024-01-31"

step = pd.DateOffset(months=1)


def compute_average_metrics(results):
    average_metrics = {}
    for strategy, models_dict in results.items():
        average_metrics[strategy] = {}
        for model, metrics_list in models_dict.items():
            if metrics_list:
                df_metrics = pd.DataFrame(metrics_list)
                avg_metrics = (
                    df_metrics[["Accuracy", "Precision", "Recall", "F1_Score"]]
                    .mean()
                    .to_dict()
                )
            else:
                avg_metrics = {
                    "Accuracy": None,
                    "Precision": None,
                    "Recall": None,
                    "F1_Score": None,
                }
            average_metrics[strategy][model] = avg_metrics

    return average_metrics


def plot_average_metrics(average_metrics, metric="F1_Score"):
    plot_data = []
    for strategy, models_dict in average_metrics.items():
        for model, metrics in models_dict.items():
            if metrics[metric] is not None:
                plot_data.append(
                    {"Strategy": strategy, "Model": model, metric: metrics[metric]}
                )

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x="Strategy", y=metric, hue="Model")
    plt.title(f"Average {metric} by Model and Strategy")
    plt.xlabel("Strategy")
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.show()


rf = RandomForestClassifier(n_estimators=100, random_state=42)
cat = CatBoostClassifier(verbose=0, random_state=42)
lgbm = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

target = "Sharpe_Ratio_Target"

strategies = merged_data.keys()
dfs = {}


def leave_important(X_train, y_train, X_test, n):
    # if n=0 leave all features
    if n > 0:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        importances.sort_values(ascending=False, inplace=True)

        top_features = importances.nlargest(n).index
        X_train_selected = X_train[top_features]
        X_test_selected = X_test[top_features]
    else:
        X_train_selected = X_train
        X_test_selected = X_test

    return X_train_selected, X_test_selected, importances


models = {
    # 'RandomForest': rf,
    "CatBoost": cat,
    "LightGBM": lgbm,
}
results_sharpe = {
    strategy: {model: [] for model in models.keys()} for strategy in strategies
}
feature_importances_dict = {}

strategy_model_dfs = {
    strategy: {model: merged_data[strategy].copy() for model in models.keys()}
    for strategy in strategies
}
for strategy in strategies:
    for model_name in models.keys():
        strategy_model_dfs[strategy][model_name][f"Preds_{model_name}"] = np.nan

for strategy in strategies:
    df = merged_data[strategy].copy()

    current_train_end = pd.to_datetime(train_end)
    current_predict_start = pd.to_datetime(predict_start)
    current_predict_end = pd.to_datetime(predict_end)

    strategy_importances = []

    while current_predict_start <= df.index.max():
        train = df.loc[:current_train_end]
        predict = df.loc[current_predict_start:current_predict_end]

        X_train = train.drop(target, axis=1)
        y_train = train[f"{target}"]

        X_predict = predict.drop(target, axis=1)
        y_true = predict[f"{target}"]
        if len(X_train) > 0:
            X_train_selected, X_predict_selected, importances = leave_important(
                X_train, y_train, X_predict, n=15
            )
            strategy_importances.append(importances)

        for model_name, model in models.items():
            try:
                if model_name == "RandomForest":
                    model = RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight="balanced"
                    )
                elif model_name == "CatBoost":
                    model = CatBoostClassifier(
                        verbose=0, random_state=42, auto_class_weights="Balanced"
                    )
                elif model_name == "LightGBM":
                    model = LGBMClassifier(
                        n_estimators=100,
                        random_state=42,
                        class_weight="balanced",
                        verbose=-1,
                    )

                model.fit(X_train_selected, y_train)

                y_pred = model.predict(X_predict_selected)
                y_pred_shifted = (
                    pd.Series(y_pred, index=predict.index).shift().fillna(1)
                )

                acc = accuracy_score(y_true, y_pred_shifted)
                prec = precision_score(y_true, y_pred_shifted, zero_division=0)
                rec = recall_score(y_true, y_pred_shifted, zero_division=0)
                f1 = f1_score(y_true, y_pred_shifted, zero_division=0)

                results_sharpe[strategy][model_name].append(
                    {
                        "Model": model_name,
                        "Train_End": current_train_end.date(),
                        "Predict_Start": current_predict_start.date(),
                        "Predict_End": current_predict_end.date(),
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1_Score": f1,
                    }
                )
                strategy_model_dfs[strategy][model_name].loc[
                    predict.index, f"Preds_{model_name}"
                ] = y_pred_shifted
            except:
                continue
        current_train_end = current_predict_end
        current_predict_start = current_train_end + pd.Timedelta(days=1)
        current_predict_end = current_predict_start + step - pd.Timedelta(days=1)

        if current_predict_end > df.index.max():
            current_predict_end = df.index.max()


def calculate_sharpe_ratio(returns):
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0:
        return 0
    return mean_return / std_return * np.sqrt(365)


for strategy in strategies:
    try:
        df = strategy_model_dfs[strategy]["CatBoost"].copy()
        if strategy in ["G59_V1", "G59_V2", "G90_V1", "G24", "G58_V1"]:
            test_period = df.loc["2024"]
            start = "2024-01-01"
        else:
            start = "2024-07-01"
            test_period = df.loc["2024-07-01":"2024-12-31"]

        cumulative_returns = test_period["Mean_Returns_1"].cumsum()

        pred_zero_cb = df.loc[
            (df.index >= start)
            & (df.index <= "2024-12-31")
            & (df["Preds_CatBoost"] == 0)
        ].index
        pred_zero_lgbm = df.loc[
            (strategy_model_dfs[strategy]["LightGBM"].index >= start)
            & (strategy_model_dfs[strategy]["LightGBM"].index <= "2024-12-31")
            & (strategy_model_dfs[strategy]["LightGBM"]["Preds_LightGBM"] == 0)
        ].index

        sim_returns_cb = test_period["Mean_Returns_1"].copy()
        sim_returns_cb.loc[test_period.index.isin(pred_zero_cb)] = 0
        cumulative_sim_cb = sim_returns_cb.cumsum()

        sim_returns_lgbm = test_period["Mean_Returns_1"].copy()
        sim_returns_lgbm.loc[test_period.index.isin(pred_zero_lgbm)] = 0
        cumulative_sim_lgbm = sim_returns_lgbm.cumsum()

        sharpe_negative_dates = test_period.loc[
            test_period["Sharpe_Ratio_Target"] == 0
        ].index

        sharpe_ratio = calculate_sharpe_ratio(test_period["Mean_Returns_1"])
        autocorrelation = test_period["Sharpe_Ratio_30"].apply(np.log).autocorr()

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 10), sharex=True)

        ax1.plot(
            test_period.index,
            cumulative_returns,
            label="Cumulative Returns",
            color="blue",
        )
        ax1.scatter(
            pred_zero_cb,
            cumulative_returns.loc[pred_zero_cb],
            color="yellow",
            label="Predicted 0",
            zorder=5,
        )
        ax1.scatter(
            pred_zero_lgbm,
            cumulative_returns.loc[pred_zero_lgbm],
            color="green",
            label="Predicted 0",
            zorder=5,
            marker="v",
        )

        ax1.plot(
            test_period.index,
            cumulative_sim_cb,
            label="Simulated (CatBoost)",
            linestyle=":",
            color="orange",
        )
        ax1.plot(
            test_period.index,
            cumulative_sim_lgbm,
            label="Simulated (LightGBM)",
            linestyle="--",
            color="purple",
        )

        ax1.scatter(
            sharpe_negative_dates,
            cumulative_returns.loc[sharpe_negative_dates],
            color="red",
            label="Sharpe Ratio < 0",
            zorder=5,
            marker="x",
        )
        ax1.set_title(f"{strategy} - Cumulative Returns with Predictions (2024)")
        ax1.set_ylabel("Cumulative Returns")
        ax1.grid(True)
        ax1.legend()
        ax1.text(
            0.95,
            0.95,
            f"Autocorrelation: {autocorrelation:.2f}\nSharpe Ratio: {sharpe_ratio:.2f}",
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.6),
        )

        ax2.plot(
            test_period.index,
            test_period["Sharpe_Ratio_30"],
            label="Sharpe Ratio (30)",
            color="green",
        )
        ax2.set_title(f"{strategy} - Sharpe Ratio (2024)")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_xlabel("Date")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"graph/ensemble/{strategy}.png")
        plt.show()
    except Exception as e:
        continue
