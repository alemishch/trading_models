import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from dtaidistance import dtw
from scipy.stats import wasserstein_distance, skew, kurtosis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

var = "Sharpe_Ratio"


def select_columns(df, metrics, windows):
    selected_columns = []
    for metric in metrics:
        for window in windows:
            if f"Sharpe_Ratio_{window}" in df.columns:
                selected_columns.append(f"Sharpe_Ratio_{window}")
            if f"Mean_Returns_{window}" in df.columns:
                selected_columns.append(f"Mean_Returns_{window}")
            if f"ACF_Lag_1_{window}" in df.columns:
                selected_columns.append(f"ACF_Lag_1_{window}")
    return df[selected_columns]


windows = [30]
# metrics = [
#     "Realized_Volatility",
#     "Garman_Klass_Volatility",
#     "OU_Theta",
#     "Hurst",
#     "Momentum",
#     "RSI",
#     "ADX",
# ]

# market_features = pd.read_csv(
#     "all_features_BTCUSDT.csv", index_col=0, parse_dates=[0], low_memory=False
# )
# market_features = select_columns(market_features, metrics, windows)
# market_features = market_features.resample("D").ffill()

strategies_data = {}
merged_data = {}
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
        # Select relevant columns for all windows
        selected_columns = []
        for window in windows:
            selected_columns.extend(
                [
                    f"Mean_Returns_{window}",
                    # f"ACF_Lag_1_{window}",
                    f"Sharpe_Ratio_30",
                ]
            )
        df_selected = df[selected_columns]
        df_selected = df_selected.resample("D").ffill()
        strategies_data[strategy_name] = df_selected
        df_selected = df_selected.dropna()  ##
        if len(df_selected > 0):  ##
            merged_data[strategy_name] = df_selected  ##

# for strategy, df in strategies_data.items():
#     merged_df = pd.merge(
#         market_features, df, left_index=True, right_index=True, how="inner"
#     )
#     if strategy in ["G59_V1", "G59_V2", "G90_V1", "G24"]:
#         merged_df = merged_df.dropna(axis=1, thresh=0.7 * len(merged_df))
#         merged_df = merged_df.dropna()
#         merged_data[strategy] = merged_df


def split_train_test(merged_data):
    train_data = {}
    test_data = {}

    for strategy, df in merged_data.items():
        if strategy in ["G59_V1", "G59_V2", "G90_V1", "G24", "G58_V1"]:
            df = df.sort_index()

            train_df = df[df.index.year != 2024]
            test_df = df[df.index.year == 2024]

            train_data[strategy] = train_df
            test_data[strategy] = test_df
        else:
            df = df.sort_index()

            train_df = df.copy()
            test_df = df.copy()

            train_data[strategy] = train_df
            test_data[strategy] = test_df

    return train_data, test_data


train_data, test_data = split_train_test(merged_data)
strategy_names = train_data.keys()


REGIME_COLOR_MAPPING = {"Low": "red", "Normal": "blue", "High": "green"}


def get_regime_colors(regime_labels):
    return [REGIME_COLOR_MAPPING.get(label, "gray") for label in regime_labels]


def visualize_regimes(
    test_data,
    test_regime_labels,
    strategy,
    window,
    data_type="test",
    method_name="Clustering",
):
    df_test = test_data.copy()
    regimes = test_regime_labels[window][strategy]
    df_test["Regime_Label"] = regimes
    df_test["datetime"] = df_test.index  # Ensure datetime is available for plotting

    plt.figure(figsize=(15, 7))

    plt.plot(
        df_test["datetime"],
        df_test[f"{var}_{window}"],
        label=f"{var} {window}D",
        color="black",
        linewidth=1,
    )

    colors = get_regime_colors(df_test["Regime_Label"])

    plt.scatter(
        df_test["datetime"],
        df_test[f"{var}_{window}"],
        c=colors,
        label="Regime",
        alpha=0.6,
        marker="o",
    )

    handles = []
    labels = []
    for regime, color in REGIME_COLOR_MAPPING.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=regime,
                markerfacecolor=color,
                markersize=10,
            )
        )
        labels.append(regime)

    if "Noise" in df_test["Regime_Label"].unique():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="w",
                label="Noise",
                markerfacecolor="gray",
                markersize=10,
            )
        )
        labels.append("Noise")

    plt.legend(handles, labels)
    plt.title(
        f"{var} Over Time with {method_name} Market Regimes for {strategy} ({window}-Day) ({data_type} Set)"
    )
    plt.xlabel("Date")
    plt.ylabel(f"{var}")
    plt.tight_layout()
    plt.savefig(f"graph/vlstar/sharpe_{window}/{data_type}_{strategy}.png")
    # plt.show()


def sortino_ratio(returns, target=0):
    downside = returns[returns < target]
    expected_return = returns.mean() - target
    downside_std = downside.std()
    return expected_return / downside_std if downside_std != 0 else np.nan


def pad_truncate_sequences(sequences, max_length):
    """
    Makes sequences of same length
    """
    fixed_length_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            # Pad with zeros
            padded_seq = np.pad(seq, (0, max_length - len(seq)), "constant")
        else:
            # Truncate to max_length
            padded_seq = seq[:max_length]
        fixed_length_sequences.append(padded_seq)
    return np.array(fixed_length_sequences)


def scale_sequences_fit_transform(sequences):
    # For train
    scaler = StandardScaler()
    scaler.fit(sequences)
    scaled_sequences = scaler.transform(sequences)
    return scaled_sequences, scaler


def scale_sequences_transform(sequences, scaler):
    # For test
    scaled_sequences = scaler.transform(sequences)
    return scaled_sequences


def fit_vlstar(train_data, window, n_clusters=3):
    kmedoids_models = {}
    scalers = {}
    train_labels = {}
    X_train_scaled_dict = {}

    for strategy in train_data.keys():
        X_train = train_data[strategy][f"{var}_{window}"].values.tolist()
        X_train = np.array(X_train)

        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        # pad and scale
        X_train_padded = pad_truncate_sequences(
            X_train, X_train.shape[1] if X_train.ndim > 1 else 1
        )
        X_train_scaled, scaler = scale_sequences_fit_transform(X_train_padded)
        scalers[strategy] = scaler
        X_train_scaled_dict[strategy] = X_train_scaled

        # distance matrix for train
        n_train = X_train_scaled.shape[0]
        distance_matrix = np.zeros((n_train, n_train))
        for i in range(n_train):
            for j in range(i + 1, n_train):
                distance = dtw.distance(X_train_scaled[i], X_train_scaled[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        # k-medoids with distance matrix
        kmedoids = KMedoids(
            n_clusters=n_clusters,
            metric="precomputed",
            init="k-medoids++",
            random_state=42,
        )
        kmedoids.fit(distance_matrix)

        train_labels[strategy] = kmedoids.labels_
        kmedoids_models[strategy] = kmedoids

    return kmedoids_models, scalers, train_labels, X_train_scaled_dict


def assign_test_labels(
    kmedoids_models, scalers, test_data, X_train_scaled_dict, window, n_clusters=3
):
    test_labels = {}

    for strategy in test_data.keys():
        X_test = test_data[strategy][f"{var}_{window}"].values.tolist()
        X_test = np.array(X_test)

        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        X_test_padded = pad_truncate_sequences(
            X_test,
            (
                X_train_scaled_dict[strategy].shape[1]
                if X_train_scaled_dict[strategy].ndim > 1
                else 1
            ),
        )

        scaler = scalers[strategy]
        X_test_scaled = scale_sequences_transform(X_test_padded, scaler)

        kmedoids = kmedoids_models[strategy]
        medoid_indices = kmedoids.medoid_indices_
        medoids = X_train_scaled_dict[strategy][medoid_indices]

        n_test = X_test_scaled.shape[0]
        distances_test = np.zeros((n_test, n_clusters))
        for i in range(n_test):
            for j in range(n_clusters):
                distances_test[i, j] = dtw.distance(X_test_scaled[i], medoids[j])

        test_labels[strategy] = distances_test.argmin(axis=1)

    return test_labels


def map_clusters_to_regimes(
    train_data,
    train_labels,
    test_labels,
    window,
    regime_labels=["Low", "Normal", "High"],
):
    train_regime_labels = {}
    test_regime_labels = {}

    for strategy in train_data.keys():
        df_train = train_data[strategy].copy()
        labels = train_labels[strategy]
        df_train["Cluster"] = labels

        cluster_sharpe = df_train.groupby("Cluster")[f"{var}_{window}"].mean()
        sorted_clusters = cluster_sharpe.sort_values().index.tolist()

        regime_mapping = {}
        num_regimes = len(regime_labels)
        if len(sorted_clusters) >= num_regimes:
            for cluster, label in zip(sorted_clusters[:num_regimes], regime_labels):
                regime_mapping[cluster] = label
            for cluster in sorted_clusters[num_regimes:]:
                regime_mapping[cluster] = f"Label_{cluster}"
        else:
            for cluster, label in zip(sorted_clusters, regime_labels):
                regime_mapping[cluster] = label

        df_train["Regime_Label"] = df_train["Cluster"].map(regime_mapping)
        train_regime_labels[strategy] = df_train["Regime_Label"]

        labels_test = test_labels[strategy]
        regime_label_test = []
        for lbl in labels_test:
            regime = regime_mapping.get(lbl, "Noise")  # 'Noise' if label not found
            regime_label_test.append(regime)

        test_regime_labels[strategy] = regime_label_test

    return train_regime_labels, test_regime_labels


def simulate_returns(
    test_sharpe,
    test_regime_labels,
    strategy,
    rets_df,
    multiplier_low=0.1,
    multiplier_high=2.0,
):
    df_test = test_sharpe.copy().to_frame()

    regimes = np.array(test_regime_labels)

    shifted_regimes = np.roll(regimes, 1)
    shifted_regimes[0] = "Normal"

    multipliers = np.ones_like(shifted_regimes, dtype=float)
    multipliers[shifted_regimes == "Low"] = multiplier_low
    multipliers[shifted_regimes == "High"] = multiplier_high

    # always trade
    df_test["Returns_Always_Trade"] = rets_df.loc[df_test.index, strategy].values
    # apply multipliers
    df_test["Returns_Conditional_Trade"] = (
        rets_df.loc[df_test.index, strategy].values * multipliers
    )
    # yield curve
    df_test["Cumulative_Always_Trade"] = df_test["Returns_Always_Trade"].cumsum()
    df_test["Cumulative_Conditional_Trade"] = df_test[
        "Returns_Conditional_Trade"
    ].cumsum()

    # Sharpe
    sharpe_always = (
        df_test["Returns_Always_Trade"].mean()
        / df_test["Returns_Always_Trade"].std()
        * np.sqrt(365)
        if df_test["Returns_Always_Trade"].std() != 0
        else np.nan
    )
    sharpe_conditional = (
        df_test["Returns_Conditional_Trade"].mean()
        / df_test["Returns_Conditional_Trade"].std()
        * np.sqrt(365)
        if df_test["Returns_Conditional_Trade"].std() != 0
        else np.nan
    )

    df_test["Regime"] = shifted_regimes

    # max drawdown
    peak_always = df_test["Cumulative_Always_Trade"].cummax()
    drawdown_always = (df_test["Cumulative_Always_Trade"] - peak_always) / peak_always
    max_drawdown_always = drawdown_always.min()

    peak_conditional = df_test["Cumulative_Conditional_Trade"].cummax()
    drawdown_conditional = (
        df_test["Cumulative_Conditional_Trade"] - peak_conditional
    ) / peak_conditional
    max_drawdown_conditional = drawdown_conditional.min()

    # Sortino
    sortino_always = sortino_ratio(df_test["Returns_Always_Trade"])
    sortino_conditional = sortino_ratio(df_test["Returns_Conditional_Trade"])

    # 3. Skewness and Kurtosis
    skew_always = skew(df_test["Returns_Always_Trade"].dropna())
    skew_conditional = skew(df_test["Returns_Conditional_Trade"].dropna())

    kurtosis_always = kurtosis(df_test["Returns_Always_Trade"].dropna())
    kurtosis_conditional = kurtosis(df_test["Returns_Conditional_Trade"].dropna())

    df_test["Returns_Lost"] = np.where(
        df_test["Regime"] == "Low",
        df_test["Returns_Always_Trade"] - df_test["Returns_Conditional_Trade"],
        0.0,
    )
    df_test["Returns_Gained"] = np.where(
        df_test["Regime"] == "High",
        df_test["Returns_Conditional_Trade"] - df_test["Returns_Always_Trade"],
        0.0,
    )

    total_returns_lost = df_test["Returns_Lost"].sum()
    total_returns_gained = df_test["Returns_Gained"].sum()

    return (
        df_test,
        sharpe_always,
        sharpe_conditional,
        max_drawdown_always,
        max_drawdown_conditional,
        sortino_always,
        sortino_conditional,
        skew_always,
        skew_conditional,
        kurtosis_always,
        kurtosis_conditional,
        total_returns_lost,
        total_returns_gained,
    )


def plot_yield_curves(
    test_sharpe, test_regime_labels, strategy, windows, rets_df, apply_mode=False
):
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    overall_sharpes = {}
    max_drawdowns = {}
    sortinos = {}
    skews = {}
    kurtoses = {}
    autocorrelations = {}
    returns_lost = {}
    returns_gained = {}
    for window in windows:
        sharpe_series = test_sharpe[f"{var}_{window}"]
        regimes = test_regime_labels

        regimes_default = np.array(regimes)

        regimes_mode1 = np.copy(regimes)
        regimes_mode1[sharpe_series < 0] = "Low"

        regimes_mode2 = np.copy(regimes)
        regimes_mode2[sharpe_series < 0] = "Low"
        regimes_mode2[(regimes_mode1 == "Low") & (sharpe_series > 0)] = "Normal"

        sharpe_text = f""
        autocorr_text = f""

        for idx, regime_mode in zip([0, 2], [regimes_default, regimes_mode2]):
            for multiplier in [1.0, 2.0]:
                # Simulate returns
                (
                    df_sim,
                    sharpe_always,
                    sharpe_conditional,
                    max_dd_always,
                    max_dd_conditional,
                    sortino_always,
                    sortino_conditional,
                    skew_always,
                    skew_conditional,
                    kurtosis_always,
                    kurtosis_conditional,
                    total_returns_lost,
                    total_returns_gained,
                ) = simulate_returns(
                    test_sharpe=sharpe_series,
                    test_regime_labels=regime_mode,
                    strategy=strategy,
                    rets_df=rets_df,
                    multiplier_high=multiplier,
                )

                overall_sharpes[window] = {
                    "Always": sharpe_always,
                    "Conditional": sharpe_conditional,
                }
                log_sharpe = np.log(
                    df_sim[f"{var}_{window}"].replace(0, np.nan)
                ).dropna()
                autocorr = log_sharpe.autocorr(lag=1)
                sharpe_cond = overall_sharpes[window]["Conditional"]
                sharpe_text += (
                    f"Sharpe mode{idx}, multiplier={multiplier}: {sharpe_cond:.2f}\n"
                )
                # autocorr_text += f"Autocorr mode{idx}, multiplier={multiplier} {var} (lag=1): {autocorr:.2f}\n"

                log_sharpe = np.log(sharpe_series.replace(0, np.nan)).dropna()
                autocorr = log_sharpe.autocorr(lag=1)
                autocorrelations[window] = autocorr

                # conditional trade cumulative returns
                ax1.plot(
                    df_sim.index,
                    df_sim["Cumulative_Conditional_Trade"],
                    label=f"Conditional Trade {window}-Day, mode{idx}, multiplier={multiplier}",
                    linestyle="--",
                )

                # Highlight Low
                low_days = df_sim[regime_mode == "Low"].index
                ax1.scatter(
                    low_days,
                    df_sim.loc[low_days, "Cumulative_Conditional_Trade"],
                    color="red",
                    marker="v",
                    alpha=0.6,
                )

        ax2.scatter(
            df_sim.index,
            df_sim[f"{var}_{window}"],
            c=get_regime_colors(df_sim["Regime"]),
            label="Regime",
            alpha=0.6,
            marker="o",
        )

    reference_window = windows[0]
    (
        df_always,
        sharpe_always_overall,
        _,
        _,
        _,
        sortino_always_overall,
        _,
        skew_always_overall,
        skew_conditional_overall,
        kurtosis_always_overall,
        kurtosis_conditional_overall,
        _,
        _,
    ) = simulate_returns(
        test_sharpe=test_sharpe[f"{var}_{reference_window}"],
        test_regime_labels=["Normal"] * len(test_sharpe[f"{var}_{reference_window}"]),
        strategy=strategy,
        rets_df=rets_df,
    )
    ax1.plot(
        df_always.index,
        df_always["Cumulative_Always_Trade"],
        label="Always Trade",
        color="blue",
    )

    ax1.set_title(f"Yield Curves for {strategy} - {var}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Returns")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title(f"Market Regimes for {strategy} - {var}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.grid(True)

    # for window in windows:
    #     sharpe_cond = overall_sharpes[window]["Conditional"]
    #     autocorr = autocorrelations[window]
    #     sharpe_text += f"Sharpe Conditional {window}-Day: {sharpe_cond:.2f}\n"
    #     autocorr_text += f"Autocorr {window}-Day {var} (lag=1): {autocorr:.2f}\n"

    mode_status = "Enabled" if apply_mode else "Disabled"
    mode_text = f"Mode: {mode_status}\n"
    sharpe_text += f"Sharpe Always: {sharpe_always_overall:.2f}\n"
    autocorr_text = f"Autocorr initial: {autocorr:.2f}\n"
    full_text = (
        # mode_text
        sharpe_text
        + autocorr_text
        # + returns_lost_text
        # + returns_gained_text
    )

    ax1.text(
        0.95,
        0.95,
        full_text,
        transform=ax1.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.6),
        verticalalignment="top",
        horizontalalignment="right",
    )

    plt.tight_layout()
    plt.savefig(f"graph/vlstar/mode_multiplier/{strategy}_{var}_{window}.png")
    plt.show()


def simulate_and_sum_returns(
    strategy_names,
    windows,
    test_sharpe_dict,
    vlstar_test_regime_labels,
    rets_df,
    year="2024",
    apply_mode=False,
):
    sum_returns_always = rets_df[strategy_names].sum(axis=1)
    sum_returns_always_yr = sum_returns_always.loc[f"{year}-01-01":f"{year}-12-31"]

    cumulative_always = sum_returns_always_yr.cumsum()

    sharpe_always_overall = (
        sum_returns_always_yr.mean() / sum_returns_always_yr.std() * np.sqrt(365)
        if sum_returns_always_yr.std() != 0
        else np.nan
    )

    autocorrelation_always = sum_returns_always_yr.pct_change()

    sum_returns_conditional = {
        window: pd.Series(0.0, index=rets_df.index) for window in windows
    }

    for window in windows:
        for strategy in strategy_names:
            sharpe_series = test_sharpe_dict[strategy].get(f"{var}_{window}", None)
            if sharpe_series is None:
                continue

            regimes = vlstar_test_regime_labels[window][strategy]

            if apply_mode:
                regimes = np.array(regimes)
                regimes[sharpe_series < 0] = "Low"

            df_sim, _, sharpe_conditional = simulate_returns(
                test_sharpe=sharpe_series,
                test_regime_labels=regimes,
                strategy=strategy,
                rets_df=rets_df,
            )

            df_sim = df_sim.reindex(rets_df.index, fill_value=0.0)

            sum_returns_conditional[window] += df_sim["Returns_Conditional_Trade"]

    # Filter for year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    sum_returns_always_yr = sum_returns_always.loc[start_date:end_date]
    sum_returns_conditional_yr = {
        window: sum_returns_conditional[window].loc[start_date:end_date]
        for window in windows
    }

    # cumulative sums
    cumulative_always = sum_returns_always_yr.cumsum()
    cumulative_conditional = {
        window: sum_returns_conditional_yr[window].cumsum() for window in windows
    }

    # Compute Sharpe ratios and autocorrelations
    sharpe_always_overall = (
        sum_returns_always_yr.mean() / sum_returns_always_yr.std() * np.sqrt(365)
        if sum_returns_always_yr.std() != 0
        else np.nan
    )
    sharpe_conditional_overall = {}
    autocorrelations = {}
    for window in windows:
        sum_returns_cond = sum_returns_conditional_yr[window]
        sharpe_cond = (
            sum_returns_cond.mean() / sum_returns_cond.std() * np.sqrt(365)
            if sum_returns_cond.std() != 0
            else np.nan
        )
        sharpe_conditional_overall[window] = sharpe_cond
        autocorr = sum_returns_cond.pct_change().autocorr(lag=1)
        autocorrelations[window] = autocorr

    return (
        cumulative_always,
        cumulative_conditional,
        sharpe_always_overall,
        sharpe_conditional_overall,
        autocorrelations,
    )


def plot_combined_yield_curves(
    cumulative_always,
    cumulative_conditional,
    sharpe_always_overall,
    sharpe_conditional_overall,
    autocorrelations,
    windows,
    strategy_names,
    year="2024",
    apply_mode=False,
):
    plt.figure(figsize=(15, 7))

    for window in windows:
        plt.plot(
            cumulative_conditional[window],
            label=f"Conditional Trade {window}-Day",
            linestyle="--",
        )

    plt.plot(
        cumulative_always,
        label="Always Trade",
        color="blue",
    )

    plt.title(f"Combined Yield Curves for All Strategies - {year}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)

    sharpe_text = f"{var} Always: {sharpe_always_overall:.2f}\n"
    autocorr_text = ""
    for window in windows:
        sharpe_cond = sharpe_conditional_overall[window]
        autocorr = autocorrelations[window]
        sharpe_text += f"{var} Conditional {window}-Day: {sharpe_cond:.2f}\n"
        autocorr_text += f"Autocorr {window}-Day Sharpe (lag=1): {autocorr:.2f}\n"

    mode_status = "Enabled" if apply_mode else "Disabled"
    mode_text = f"Mode: {mode_status}\n"

    full_text = mode_text + sharpe_text + autocorr_text

    plt.text(
        0.05,
        0.95,
        full_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6),
    )

    plt.tight_layout()
    plt.savefig(
        f"graph/vlstar/combined_simulation_2024_{'mode' if apply_mode else 'nomode'}.png"
    )
    plt.show()


rets = pd.read_csv(
    "rets.csv", index_col="datetime", parse_dates=["datetime"], low_memory=False
)
rets = rets.resample("D").ffill()


def analyze_performance(
    train_sharpe, train_regime_labels, strategy, rets_df, threshold=0.6
):
    (
        df_train_sim,
        sharpe_always,
        sharpe_conditional,
        max_dd_always,
        max_dd_conditional,
        sortino_always,
        sortino_conditional,
        skew_always,
        skew_conditional,
        kurtosis_always,
        kurtosis_conditional,
    ) = simulate_returns(
        test_sharpe=train_sharpe,
        test_regime_labels=train_regime_labels,
        strategy=strategy,
        rets_df=rets_df,
    )

    # find if better then always trade
    comparison = (
        df_train_sim["Cumulative_Conditional_Trade"]
        >= df_train_sim["Cumulative_Always_Trade"]
    )
    percentage_better = comparison.mean()

    works_good = percentage_better >= threshold

    # metrics
    log_sharpe = np.log(train_sharpe.replace(0, np.nan)).dropna()
    autocorr = log_sharpe.autocorr(lag=1)

    result = {
        "strategy": strategy,
        "sharpe_always": sharpe_always,
        "sharpe_conditional": sharpe_conditional,
        "max_drawdown_always": max_dd_always,
        "max_drawdown_conditional": max_dd_conditional,
        "sortino_always": sortino_always,
        "sortino_conditional": sortino_conditional,
        "skew_always": skew_always,
        "skew_conditional": skew_conditional,
        "kurtosis_always": kurtosis_always,
        "kurtosis_conditional": kurtosis_conditional,
        "percentage_better": percentage_better,
        "works_good": works_good,
        "autocorr_log_sharpe": autocorr,
    }

    return result


kmedoids_models_vlstar = {window: {} for window in windows}
scalers_vlstar = {window: {} for window in windows}
vlstar_train_labels = {window: {} for window in windows}
X_train_scaled_dict = {window: {} for window in windows}

for window in windows:
    (
        kmedoids_models_vlstar[window],
        scalers_vlstar[window],
        vlstar_train_labels[window],
        X_train_scaled_dict[window],
    ) = fit_vlstar(train_data, window, n_clusters=3)

vlstar_test_labels = {window: {} for window in windows}

for window in windows:
    vlstar_test_labels[window] = assign_test_labels(
        kmedoids_models_vlstar[window],
        scalers_vlstar[window],
        test_data,
        X_train_scaled_dict[window],
        window,
        n_clusters=3,
    )

vlstar_train_regime_labels = {window: {} for window in windows}
vlstar_test_regime_labels = {window: {} for window in windows}

for window in windows:
    train_regime_labels, test_regime_labels = map_clusters_to_regimes(
        train_data,
        vlstar_train_labels[window],
        vlstar_test_labels[window],
        window,
        regime_labels=["Low", "Normal", "High"],
    )
    vlstar_train_regime_labels[window] = train_regime_labels
    vlstar_test_regime_labels[window] = test_regime_labels

# for strategy in strategy_names:
#     for window in windows:
#         print(f"Processing for {strategy}, {window}")
#         visualize_regimes(
#             test_data=train_data[strategy][[f"{var}_{window}"]],
#             test_regime_labels=vlstar_train_regime_labels,
#             strategy=strategy,
#             data_type="Train",
#             window=window,
#         )
#         visualize_regimes(
#             test_data=test_data[strategy][[f"{var}_{window}"]],
#             test_regime_labels=vlstar_test_regime_labels,
#             strategy=strategy,
#             data_type="Test",
#             window=window,
#         )
#         plot_yield_curves(
#             test_sharpe=test_data[strategy][[f"{var}_{window}"]],
#             test_regime_labels=vlstar_test_regime_labels[window],
#             strategy=strategy,
#             window=window,
#             rets_df=rets,
#         )

# for strategy in strategy_names:
#     for apply_mode in [False, True]:
#         mode_text = "with_mode" if apply_mode else "no_mode"
#         print(f"Processing for {strategy}, Mode: {mode_text}")
#         plot_yield_curves(
#             test_sharpe=test_data[strategy],
#             test_regime_labels=vlstar_test_regime_labels,
#             strategy=strategy,
#             windows=windows,
#             rets_df=rets,
#             apply_mode=apply_mode,
#         )


# for apply_mode in [True, False]:
#     (
#         cumulative_always,
#         cumulative_conditional,
#         sharpe_always_overall,
#         sharpe_conditional_overall,
#         autocorrelations,
#     ) = simulate_and_sum_returns(
#         strategy_names=strategy_names,
#         windows=windows,
#         test_sharpe_dict={strategy: test_data[strategy] for strategy in strategy_names},
#         vlstar_test_regime_labels=vlstar_test_regime_labels,
#         rets_df=rets,
#         year="2024",
#         apply_mode=apply_mode,
#     )

#     plot_combined_yield_curves(
#         cumulative_always=cumulative_always,
#         cumulative_conditional=cumulative_conditional,
#         sharpe_always_overall=sharpe_always_overall,
#         sharpe_conditional_overall=sharpe_conditional_overall,
#         autocorrelations=autocorrelations,
#         windows=windows,
#         strategy_names=strategy_names,
#         year="2024",
#         apply_mode=apply_mode,
#     )

results = []
strategy_names = ["G59_V1", "G59_V2", "G90_V1", "G24", "G58_V1"]
# for strategy in strategy_names:
#     print(f"Strategy {strategy}")
#     windows = [1, 5]
#     var = "Mean_Returns"
#     for window in windows:
#         test_sharpe_series = test_data[strategy][f"{var}_{window}"]
#         test_regimes = vlstar_test_regime_labels[window][strategy]

#         plot_yield_curves(
#             test_sharpe=test_sharpe_series.to_frame(),
#             test_regime_labels=test_regimes,
#             strategy=strategy,
#             windows=[window],
#             rets_df=rets,
#             apply_mode=False,
#         )

for strategy in strategy_names:
    print(f"Strategy {strategy}")
    window = 30
    test_sharpe_series = test_data[strategy][f"{var}_{window}"]
    test_regimes = vlstar_test_regime_labels[window][strategy]
    plot_yield_curves(
        test_sharpe=test_sharpe_series.to_frame(),
        test_regime_labels=test_regimes,
        strategy=strategy,
        windows=[30],
        rets_df=rets,
        apply_mode=False,
    )


#     analysis_result = analyze_performance(
#         train_sharpe=test_sharpe_series,
#         train_regime_labels=test_regimes,
#         strategy=strategy,
#         rets_df=rets,
#         threshold=0.6,
#     )

#     results.append(analysis_result)


# results_df = pd.DataFrame(results)

# good_cases = results_df[results_df["works_good"]]
# poor_cases = results_df[~results_df["works_good"]]
# print(good_cases["strategy"])
# avg_metrics_good = good_cases.mean(numeric_only=True)
# avg_metrics_poor = poor_cases.mean(numeric_only=True)

# print("\nAvg metrics for good performance:\n")
# print(avg_metrics_good)

# print("\nAvg metrics for poor performance:\n")
# print(avg_metrics_poor)
