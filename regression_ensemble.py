import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
from collections import defaultdict

market_features = pd.read_csv(
    "all_features_BTCUSDT.csv", index_col=0, parse_dates=[0], low_memory=False
)
market_features = market_features.resample("D").ffill()
market_features = market_features.loc[:, market_features.isna().mean() <= 0.7]
market_columns = market_features.columns.tolist()

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

        df["Sharpe_Target_10"] = df["Sharpe_Ratio_10"].shift(-1)
        df["Sharpe_Target_30"] = df["Sharpe_Ratio_30"].shift(-1)

        df["Mean_Returns_1_Target"] = df["Mean_Returns_1"].shift(-1)

        df["Mean_Returns_5_Target"] = df["Mean_Returns_5"].shift(-1)
        strategies_data[strategy_name] = df

targets = [
    "Sharpe_Target_10",
    "Sharpe_Target_30",
    "Mean_Returns_5_Target",
    "Mean_Returns_1_Target",
]

trues = {
    "Sharpe_Target_10": "Sharpe_Ratio_10",
    "Sharpe_Target_30": "Sharpe_Ratio_30",
    "Mean_Returns_1_Target": "Mean_Returns_1",
    "Mean_Returns_5_Target": "Mean_Returns_5",
}

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


cat = CatBoostRegressor(verbose=0, random_state=42)
lgbm = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

models = {
    # "logit": logit,
    "CatBoost": cat,
    "LightGBM": lgbm,
}

strategies = [
    "G44",
    "G85_V1",
    "G24",
    "G59_V2",
    "G70_V1",
    "G58_V1",
    "G33_V1",
    "G73",
    "G43",
    "G59_V1",
    "G69_V2",
    "G90_V1",
    "G53_V1",
    "G19_V5",
]

feature_sets = {"With_MF": True, "Without_MF": False}

results_dict = {
    strategy: {
        model: {target: {fs: [] for fs in feature_sets.keys()} for target in targets}
        for model in models.keys()
    }
    for strategy in strategies
}

feature_importances_dict = {
    strategy: {
        model: {target: {fs: {} for fs in feature_sets.keys()} for target in targets}
        for model in models.keys()
    }
    for strategy in strategies
}

strategy_model_dfs = {
    strategy: {
        model: {fs: merged_data[strategy].copy() for fs in feature_sets.keys()}
        for model in models.keys()
    }
    for strategy in strategies
}

for strategy in strategies:
    for model_name in models.keys():
        for fs in feature_sets.keys():
            for target in targets:
                strategy_model_dfs[strategy][model_name][fs][
                    f"Preds_{model_name}_{target}"
                ] = np.nan


def get_feature_columns(df, include_market):
    if include_market:
        feature_cols = [col for col in df.columns if col not in targets]
    else:
        feature_cols = [
            col
            for col in df.columns
            if not col in market_columns and col not in targets
        ]
    return feature_cols


def leave_important(X_train, y_train, X_test, n):
    # if n=0 leave all features
    if n > 0:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
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


# backup_dir = "regression_backup"
# os.makedirs(backup_dir, exist_ok=True)

# for strategy in tqdm(strategies):
#     df = merged_data[strategy].copy()
#     df = df.sort_index()

#     for target in tqdm(targets):
#         for model_name, model in models.items():
#             for fs_name, include_market in feature_sets.items():
#                 feature_cols = get_feature_columns(df, include_market)
#                 current_train_end = pd.to_datetime(train_end)
#                 current_predict_start = pd.to_datetime(predict_start)
#                 current_predict_end = pd.to_datetime(predict_end)

#                 while current_predict_start <= df.index.max():
#                     train = df.loc[:current_train_end]
#                     predict = df.loc[current_predict_start:current_predict_end]

#                     X_train = train[feature_cols]
#                     y_train = train[target]

#                     X_predict = predict[feature_cols]
#                     y_true = predict[target]

#                     if len(X_train) == 0:
#                         current_train_end = current_predict_end
#                         current_predict_start = current_train_end + pd.Timedelta(days=1)
#                         current_predict_end = (
#                             current_predict_start + step - pd.Timedelta(days=1)
#                         )
#                         if current_predict_end > df.index.max():
#                             current_predict_end = df.index.max()
#                         continue

#                     try:
#                         (
#                             X_train_selected,
#                             X_predict_selected,
#                             importances,
#                         ) = leave_important(X_train, y_train, X_predict, n=15)
#                     except Exception as e:
#                         print(e)
#                         current_train_end = current_predict_end
#                         current_predict_start = current_train_end + pd.Timedelta(days=1)
#                         current_predict_end = (
#                             current_predict_start + step - pd.Timedelta(days=1)
#                         )
#                         if current_predict_end > df.index.max():
#                             current_predict_end = df.index.max()
#                         continue

#                     try:
#                         model.fit(X_train_selected, y_train)

#                         try:
#                             y_pred = model.predict(X_predict_selected)
#                             y_pred_shifted = (
#                                 pd.Series(y_pred, index=predict.index).shift().fillna(1)
#                             )
#                             strategy_model_dfs[strategy][model_name][fs_name].loc[
#                                 y_pred_shifted.index, f"Preds_{model_name}_{target}"
#                             ] = y_pred_shifted
#                         except Exception as e:
#                             print(
#                                 f"Error for {strategy}, Model: {model_name}, Target: {target}, Feature Set: {fs_name}. Error: {e}"
#                             )
#                             # Fill with 1
#                             strategy_model_dfs[strategy][model_name][fs_name].loc[
#                                 predict.index, f"Preds_{model_name}_{target}"
#                             ] = 0

#                     except Exception as e:
#                         print(e)
#                         current_train_end = current_predict_end
#                         current_predict_start = current_train_end + pd.Timedelta(days=1)
#                         current_predict_end = (
#                             current_predict_start + step - pd.Timedelta(days=1)
#                         )
#                         if current_predict_end > df.index.max():
#                             current_predict_end = df.index.max()
#                         continue

#                     current_train_end = current_predict_end
#                     current_predict_start = current_train_end + pd.Timedelta(days=1)
#                     current_predict_end = (
#                         current_predict_start + step - pd.Timedelta(days=1)
#                     )

#                     if current_predict_end > df.index.max():
#                         current_predict_end = df.index.max()


# def calculate_sharpe_ratio(returns):
#     mean_return = returns.mean()
#     std_return = returns.std()
#     if std_return == 0:
#         return 0
#     return mean_return / std_return * np.sqrt(365)


backup_dir = "regression_backup"
# os.makedirs(backup_dir, exist_ok=True)

# for strategy, models_dict in strategy_model_dfs.items():
#     for model_name, feature_sets_dict in models_dict.items():
#         for feature_set, df in feature_sets_dict.items():
#             file_name = f"{strategy}_{model_name}_{feature_set}.xlsx"
#             file_path = os.path.join(backup_dir, file_name)

#             try:
#                 df.to_excel(file_path, index=True)
#                 print(f"Saved {file_name} successfully.")
#             except Exception as e:
#                 print(f"Error saving {file_name}: {e}")

# print(strategy_model_dfs["G19_V5"]["CatBoost"]["With_Market_Features"].head(100))
# print(strategy_model_dfs["G19_V5"]["LightGBM"]["With_Market_Features"].head(100))


strategy_model_dfs = defaultdict(lambda: defaultdict(dict))
mf = {"With_Market_Features: With_MF, Without_Market_Features": "Without_MF"}


def load_data_to_dict(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(".xlsx"):
            try:
                file_parts = file_name.replace(".xlsx", "").split("_")
                if len(file_parts) == 4:
                    strategy = file_parts[0]
                    model_name = file_parts[1]
                    fs_name = "_".join(file_parts[2:])

                    file_path = os.path.join(directory, file_name)

                    df = pd.read_excel(file_path, index_col=0)
                    print(strategy, model_name, fs_name)
                    strategy_model_dfs[strategy][model_name][fs_name] = df
                    print(f"Loaded {file_name}")
                elif len(file_parts) >= 5:
                    strategy = f"{file_parts[0]}_{file_parts[1]}"
                    model_name = file_parts[2]
                    fs_name = "_".join(file_parts[3:])

                    file_path = os.path.join(directory, file_name)

                    df = pd.read_excel(file_path, index_col=0)
                    print(strategy, model_name, fs_name)
                    strategy_model_dfs[strategy][model_name][fs_name] = df
                    print(f"Loaded {file_name}")
                else:
                    print(f" invalid file name {file_name}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    return strategy_model_dfs


strategy_model_dfs = load_data_to_dict(backup_dir)
for strategy in strategies:
    for target in targets:
        plt.figure(figsize=(10, 6))
        for i, model in enumerate(models):
            for j, features in enumerate(feature_sets):
                df = strategy_model_dfs[strategy][model][features].copy()
                if strategy in ["G59_V1", "G59_V2", "G90_V1", "G24", "G58_V1"]:
                    test_period = df.loc["2024"]
                    start = "2024-01-01"
                else:
                    start = "2024-07-01"
                    test_period = df.loc["2024-07-01":"2024-12-31"]

                cumulative_returns = test_period["Mean_Returns_1"].cumsum()

                pred_zero_abs = test_period.loc[
                    test_period[f"Preds_{model}_{target}"] < 0
                ].index

                if target in ["Sharpe_Target_10", "Sharpe_Target_30"]:
                    shifted_preds = df[f"Preds_{model}_{target}"].shift(-1).fillna(1)
                    pred_zero_cur = df.loc[
                        (df.index >= start)
                        & (df.index <= "2024-12-31")
                        & (shifted_preds < df[trues[target]])
                    ].index
                    pred_zero_cur = pred_zero_cur.shift(1, freq="D").dropna()

                sim_returns_abs = test_period["Mean_Returns_1"].copy()
                sim_returns_abs.loc[test_period.index.isin(pred_zero_abs)] = 0
                cumulative_sim_abs = sim_returns_abs.cumsum()

                sim_returns_cur = test_period["Mean_Returns_1"].copy()
                sim_returns_cur.loc[test_period.index.isin(pred_zero_cur)] = 0
                cumulative_sim_cur = sim_returns_cur.cumsum()

                if i == 0 and j == 0:
                    plt.plot(
                        test_period.index,
                        cumulative_returns,
                        label=f"True {target}",
                        color="blue",
                    )

                plt.plot(
                    test_period.index,
                    cumulative_sim_abs,
                    label=f"Pred {target} by {model} {features} (abs)",
                    linestyle=":",
                )

                if target in ["Sharpe_Target_10", "Sharpe_Target_30"]:
                    plt.plot(
                        test_period.index,
                        cumulative_sim_cur,
                        label=f"Pred {target} by {model} {features} (current)",
                        linestyle=":",
                    )

        plt.title(f"{strategy} - regression")
        plt.ylabel(f"Сг")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"graph/regression/{strategy}_{target}.png")
        # plt.show()

simulation_results = []
end_date = "2024-12-31"

for strategy in strategies:
    for model_name in models.keys():
        for feature_set, include_market in feature_sets.items():
            for target in targets:
                try:
                    if strategy in ["G59_V1", "G59_V2", "G90_V1", "G24", "G58_V1"]:
                        start_date = "2024-01-01"
                    else:
                        start_date = "2024-07-01"

                    df = strategy_model_dfs[strategy][model_name][feature_set].copy()

                    df_sim = df.loc[start_date:end_date].copy()

                    pred_column = f"Preds_{model_name}_{target}"

                    if pred_column not in df_sim.columns:
                        print(
                            f"Prediction column {pred_column} not found in strategy {strategy}. Skipping."
                        )
                        continue

                    sim_returns = df_sim["Mean_Returns_1"].copy()
                    sim_returns[df_sim[pred_column] == 0] = 0

                    cumulative_sim = sim_returns.sum()

                    simulation_results.append(
                        {
                            "Model": model_name,
                            "Feature_Set": feature_set,
                            "Target": target,
                            "Strategy": strategy,
                            "Cumulative_Returns": cumulative_sim,
                        }
                    )

                except Exception as e:
                    print(
                        f"Simulation error for Strategy: {strategy}, Model: {model_name}, Feature Set: {feature_set}, Target: {target}. Error: {e}"
                    )
                    continue


# benchmark_results = []

# for strategy in strategies:
#     try:
#         if strategy in ["G59_V1", "G59_V2", "G90_V1", "G24", "G58_V1"]:
#             start_date = "2024-01-01"
#         else:
#             start_date = "2024-07-01"

#         df = merged_data[strategy].copy()

#         df_benchmark = df.loc[start_date:end_date].copy()

#         cumulative_benchmark = df_benchmark["Mean_Returns_1"].sum()

#         benchmark_results.append(
#             {
#                 "Model": "Benchmark",
#                 "Feature_Set": "N/A",
#                 "Target": "N/A",
#                 "Strategy": strategy,
#                 "Cumulative_Returns": cumulative_benchmark,
#             }
#         )

#     except Exception as e:
#         print(f"error for Strategy: {strategy}. Error: {e}")
#         continue

# all_results = simulation_results + benchmark_results
# df_all_results = pd.DataFrame(all_results)

# df_all_results["Experiment"] = df_all_results.apply(
#     lambda row: f"{row['Model']} | {row['Feature_Set']} | {row['Target']}", axis=1
# )

# df_pivot = df_all_results.pivot_table(
#     index=["Model", "Feature_Set", "Target"],
#     columns="Strategy",
#     values="Cumulative_Returns",
#     aggfunc="first",
# ).reset_index()


# df_pivot.set_index(["Model", "Feature_Set", "Target"], inplace=True)

# cmap = "RdYlGn"  # Red for poor returns, Green for good returns

# vmin = df_pivot.min().min()
# vmax = df_pivot.max().max()

# styled_table = df_pivot.style.background_gradient(
#     cmap=cmap, axis=None, vmin=vmin, vmax=vmax
# ).format("{:.4f}")

# styled_table.to_html("simulation_results.html")