import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
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

        # df["Sharpe_Target_10"] = df["Sharpe_Ratio_10"].shift(-1)
        # df["Sharpe_Target_30"] = df["Sharpe_Ratio_30"].shift(-1)

        df["Mean_Returns_1_Target"] = df["Mean_Returns_1"].shift(-1)

        # df["Mean_Returns_5_Target"] = df["Mean_Returns_5"].shift(-1)
        strategies_data[strategy_name] = df


merged_data = {}
for strategy, df in strategies_data.items():
    merged_df = pd.merge(
        market_features, df, left_index=True, right_index=True, how="inner"
    )
    merged_df = merged_df.dropna(axis=1, thresh=0.5 * len(merged_df))
    merged_df = merged_df.dropna()
    merged_data[strategy] = merged_df

backup_dir = "regression_backup"
os.makedirs(backup_dir, exist_ok=True)

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

feature_sets = {"With_MF": True, "Without_MF": False}

strategy_model_dfs = defaultdict(lambda: defaultdict(dict))


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


simulation_results = []
end_date = "2024-12-31"
allowed_cur_targets = ["Sharpe_Target_10", "Sharpe_Target_30"]


for source, label in [
    (strategy_model_dfs, "top 15 features"),
    # (strategy_model_dfs_all_features, "all features"),
]:
    for strategy in strategies:
        for model in models.keys():
            for features, include_market in feature_sets.items():
                for target in targets:
                    try:
                        # if (
                        #     label == "all features"
                        #     and target != "Mean_Returns_1_Target"
                        # ):
                        #     print(
                        #         f"Skipping target '{target}' for {label}, Strategy={strategy}, Model={model}, Features={features}"
                        #     )
                        #     continue
                        df = source[strategy][model][features].copy()

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
                            shifted_preds = (
                                df[f"Preds_{model}_{target}"].shift(-1).fillna(1)
                            )
                            pred_zero_cur = df.loc[
                                (df.index >= start)
                                & (df.index <= "2024-12-31")
                                & (shifted_preds < df[trues[target]])
                            ].index
                            pred_zero_cur = pred_zero_cur.shift(1, freq="D").dropna()
                            sim_returns_cur = test_period["Mean_Returns_1"].copy()
                            sim_returns_cur.loc[
                                test_period.index.isin(pred_zero_cur)
                            ] = 0
                            cumulative_sim_cur = sim_returns_cur.sum()
                        else:
                            cumulative_sim_cur = 0

                        sim_returns_abs = test_period["Mean_Returns_1"].copy()
                        sim_returns_abs.loc[test_period.index.isin(pred_zero_abs)] = 0
                        cumulative_sim_abs = sim_returns_abs.sum()

                        if target in allowed_cur_targets:
                            simulation_results.append(
                                {
                                    "Model": model,
                                    "Feature_Set": features,
                                    "Selected features": label,
                                    "Target": target,
                                    "Strategy": strategy,
                                    "Metric": "Cumulative_Returns Cur",
                                    "Cumulative_Returns": cumulative_sim_cur,
                                }
                            )

                        simulation_results.append(
                            {
                                "Model": model,
                                "Feature_Set": features,
                                "Selected features": label,
                                "Target": target,
                                "Strategy": strategy,
                                "Metric": "Cumulative_Returns abs",
                                "Cumulative_Returns": cumulative_sim_abs,
                            }
                        )

                    except Exception as e:
                        print(
                            f"Simulation error for {label},Strategy: {strategy}, Model: {model}, Feature Set: {features}, Target: {target}. Error: {e}"
                        )
                        continue

regression_backup_all_features = "regression_backup_all_features"
strategy_model_dfs_all_features = load_data_to_dict(regression_backup_all_features)
for source, label in [
    # (strategy_model_dfs, "top 15 features"),
    (strategy_model_dfs_all_features, "all features"),
]:
    for strategy in strategies:
        for model in models.keys():
            for features, include_market in feature_sets.items():
                for target in targets:
                    try:
                        if (
                            label == "all features"
                            and target != "Mean_Returns_1_Target"
                        ):
                            print(
                                f"Skipping target '{target}' for {label}, Strategy={strategy}, Model={model}, Features={features}"
                            )
                            continue
                        df = source[strategy][model][features].copy()

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
                            shifted_preds = (
                                df[f"Preds_{model}_{target}"].shift(-1).fillna(1)
                            )
                            pred_zero_cur = df.loc[
                                (df.index >= start)
                                & (df.index <= "2024-12-31")
                                & (shifted_preds < df[trues[target]])
                            ].index
                            pred_zero_cur = pred_zero_cur.shift(1, freq="D").dropna()
                            sim_returns_cur = test_period["Mean_Returns_1"].copy()
                            sim_returns_cur.loc[
                                test_period.index.isin(pred_zero_cur)
                            ] = 0
                            cumulative_sim_cur = sim_returns_cur.sum()
                        else:
                            cumulative_sim_cur = 0

                        sim_returns_abs = test_period["Mean_Returns_1"].copy()
                        sim_returns_abs.loc[test_period.index.isin(pred_zero_abs)] = 0
                        cumulative_sim_abs = sim_returns_abs.sum()

                        if target in allowed_cur_targets:
                            simulation_results.append(
                                {
                                    "Model": model,
                                    "Feature_Set": features,
                                    "Selected features": label,
                                    "Target": target,
                                    "Strategy": strategy,
                                    "Metric": "Cumulative_Returns Cur",
                                    "Cumulative_Returns": cumulative_sim_cur,
                                }
                            )

                        simulation_results.append(
                            {
                                "Model": model,
                                "Feature_Set": features,
                                "Selected features": label,
                                "Target": target,
                                "Strategy": strategy,
                                "Metric": "Cumulative_Returns abs",
                                "Cumulative_Returns": cumulative_sim_abs,
                            }
                        )

                    except Exception as e:
                        print(
                            f"Simulation error for {label},Strategy: {strategy}, Model: {model}, Feature Set: {features}, Target: {target}. Error: {e}"
                        )
                        continue

benchmark_results = []

for strategy in strategies:
    try:
        if strategy in ["G59_V1", "G59_V2", "G90_V1", "G24", "G58_V1"]:
            start_date = "2024-01-01"
        else:
            start_date = "2024-07-01"

        try:
            model = next(iter(strategy_model_dfs_all_features[strategy].keys()))
            features = next(
                iter(strategy_model_dfs_all_features[strategy][model].keys())
            )
            df = strategy_model_dfs_all_features[strategy][model][features].copy()
        except StopIteration:
            print(f"No data available for Strategy: {strategy} in all_features.")
            continue

        df_benchmark = df.loc[start_date:end_date].copy()

        cumulative_benchmark = df_benchmark["Mean_Returns_1"].sum()

        for metric in ["Cumulative_Returns Cur", "Cumulative_Returns abs"]:
            benchmark_results.append(
                {
                    "Model": "Benchmark",
                    "Feature_Set": "N/A",
                    "Selected features": "N/A",
                    "Target": "N/A",
                    "Strategy": strategy,
                    "Metric": metric,  # <-- Use actual metrics to match simulation rows
                    "Cumulative_Returns": cumulative_benchmark,
                }
            )

    except Exception as e:
        print(f"error for Strategy: {strategy}. Error: {e}")
        continue

all_results = simulation_results + benchmark_results
df_all_results = pd.DataFrame(all_results)

# Pivot the DataFrame
df_pivot = df_all_results.pivot_table(
    index=["Model", "Feature_Set", "Selected features", "Target", "Metric"],
    columns="Strategy",
    values="Cumulative_Returns",
    aggfunc="first",
).reset_index()

# Separate benchmark rows
benchmark_df = df_pivot[
    (df_pivot["Model"] == "Benchmark")
    & (df_pivot["Feature_Set"] == "N/A")
    & (df_pivot["Selected features"] == "N/A")
    & (df_pivot["Target"] == "N/A")
].copy()

simulation_df = df_pivot[
    ~(
        (df_pivot["Model"] == "Benchmark")
        & (df_pivot["Feature_Set"] == "N/A")
        & (df_pivot["Selected features"] == "N/A")
        & (df_pivot["Target"] == "N/A")
    )
].copy()

strategy_cols = [
    col
    for col in simulation_df.columns
    if col not in ["Model", "Feature_Set", "Selected features", "Target", "Metric"]
]

benchmark_melted = benchmark_df.melt(
    id_vars=["Metric"],
    value_vars=strategy_cols,
    var_name="Strategy",
    value_name="Benchmark_Return",
)

simulation_melted = simulation_df.melt(
    id_vars=["Model", "Feature_Set", "Selected features", "Target", "Metric"],
    value_vars=strategy_cols,
    var_name="Strategy",
    value_name="Cumulative_Returns",
)

merged_df = simulation_melted.merge(
    benchmark_melted, on=["Metric", "Strategy"], how="left"
)

merged_df["Better_than_Benchmark"] = (
    merged_df["Cumulative_Returns"] > merged_df["Benchmark_Return"]
)

better_than_benchmark_counts = (
    merged_df.groupby(
        ["Model", "Feature_Set", "Selected features", "Target", "Metric"]
    )["Better_than_Benchmark"]
    .sum()
    .reset_index()
)

combined_df = pd.concat([benchmark_df, simulation_df], ignore_index=True)

final_df = combined_df.merge(
    better_than_benchmark_counts,
    on=["Model", "Feature_Set", "Selected features", "Target", "Metric"],
    how="left",
)

final_df.rename(
    columns={"Better_than_Benchmark": "Better_Than_Benchmark"}, inplace=True
)

final_df.set_index(
    ["Model", "Feature_Set", "Selected features", "Target", "Metric"], inplace=True
)

final_df["Better_Than_Benchmark"].fillna(0, inplace=True)

cmap = "RdYlGn"  # Red for poor returns, Green for good returns

gradient_cols = [col for col in final_df.columns if col != "Better_Than_Benchmark"]

# Determine min and max for gradient
vmin = final_df[gradient_cols].min().min()
vmax = final_df[gradient_cols].max().max()

styled_table = (
    final_df.style.format("{:.4f}")
    .background_gradient(
        cmap=cmap,
        axis=None,
        subset=gradient_cols,
        vmin=vmin,
        vmax=vmax,
    )
    .set_table_styles(
        [
            # Add border to table headers
            {"selector": "th", "props": [("border", "1px solid black")]},
            # Add border to table data cells
            {"selector": "td", "props": [("border", "1px solid black")]},
            # Optionally, add border to the table itself
            {"selector": "table", "props": [("border-collapse", "collapse")]},
        ]
    )
    .format({"Better_Than_Benchmark": "{:.0f}"})
)

styled_table.to_html("regression_results.html")
