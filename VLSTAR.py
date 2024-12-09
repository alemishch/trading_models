import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from dtaidistance import dtw
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os


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
metrics = [
    "Realized_Volatility",
    "Garman_Klass_Volatility",
    "OU_Theta",
    "Hurst",
    "Momentum",
    "RSI",
    "ADX",
]

market_features = pd.read_csv(
    "all_features_BTCUSDT.csv", index_col=0, parse_dates=[0], low_memory=False
)
market_features = select_columns(market_features, metrics, windows)
market_features = market_features.resample("D").ffill()

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
        # Select relevant columns for all windows
        selected_columns = []
        for window in windows:
            selected_columns.extend(
                [
                    f"Mean_Returns_{window}",
                    f"ACF_Lag_1_{window}",
                    f"Sharpe_Ratio_{window}",
                ]
            )
        df_selected = df[selected_columns]
        df_selected = df_selected.resample("D").ffill()
        strategies_data[strategy_name] = df_selected

merged_data = {}
for strategy, df in strategies_data.items():
    merged_df = pd.merge(
        market_features, df, left_index=True, right_index=True, how="inner"
    )
    if strategy in ["G59_V1", "G59_V2", "G90_V1", "G24"]:
        merged_df = merged_df.dropna(axis=1, thresh=0.7 * len(merged_df))
        merged_df = merged_df.dropna()
        merged_data[strategy] = merged_df


def split_train_test(merged_data):
    train_data = {}
    test_data = {}

    for strategy, df in merged_data.items():
        df = df.sort_index()

        train_df = df[df.index.year != 2024]
        test_df = df[df.index.year == 2024]

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
        df_test[f"Sharpe_Ratio_{window}"],
        label=f"Sharpe Ratio {window}D",
        color="black",
        linewidth=1,
    )

    colors = get_regime_colors(df_test["Regime_Label"])

    plt.scatter(
        df_test["datetime"],
        df_test[f"Sharpe_Ratio_{window}"],
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
        f"Sharpe Ratio Over Time with {method_name} Market Regimes for {strategy} ({window}-Day) ({data_type} Set)"
    )
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.tight_layout()
    plt.savefig(f"graph/vlstar/sharpe_{window}/{data_type}_{strategy}.png")
    plt.show()


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
        X_train = train_data[strategy][f"Sharpe_Ratio_{window}"].values.tolist()
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
        X_test = test_data[strategy][f"Sharpe_Ratio_{window}"].values.tolist()
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

        cluster_sharpe = df_train.groupby("Cluster")[f"Sharpe_Ratio_{window}"].mean()
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


def simulate_returns(test_sharpe, test_regime_labels, strategy, rets_df):
    df_test = test_sharpe.copy()
    regimes = np.array(test_regime_labels[strategy])

    # always trade
    df_test["Returns_Always_Trade"] = rets_df.loc[df_test.index, strategy].values

    # don't trade when Low
    df_test["Returns_Conditional_Trade"] = rets_df.loc[df_test.index, strategy].values
    df_test.loc[regimes == "Low", "Returns_Conditional_Trade"] = 0

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

    return df_test, sharpe_always, sharpe_conditional


def plot_yield_curves(test_sharpe, test_regime_labels, strategy, window, rets_df):
    df_sim, sharpe_always, sharpe_conditional = simulate_returns(
        test_sharpe, test_regime_labels, strategy, rets_df
    )

    plt.figure(figsize=(15, 7))
    plt.plot(
        df_sim.index,
        df_sim["Cumulative_Always_Trade"],
        label="Always Trade",
        color="blue",
    )
    plt.plot(
        df_sim.index,
        df_sim["Cumulative_Conditional_Trade"],
        label="Conditional Trade",
        color="orange",
    )

    plt.title(f"Yield Curves for {strategy} - {window}-Day Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)

    text_x = df_sim.index[int(len(df_sim) * 0.05)]  # 5% from the start
    text_y = (
        df_sim["Cumulative_Always_Trade"].max() * 0.95
    )  # 95% of the max cumulative return
    plt.text(
        text_x,
        text_y,
        f"Sharpe Always: {sharpe_always:.2f}\nSharpe Conditional: {sharpe_conditional:.2f}",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.6),
    )

    plt.tight_layout()
    plt.savefig(f"graph/vlstar/simulation/{strategy}_{window}.png")
    plt.show()


rets = pd.read_csv(
    "rets.csv", index_col="datetime", parse_dates=["datetime"], low_memory=False
)
rets = rets.resample("D").ffill()

test_period_dates = next(iter(test_data.values())).index
rets = rets.loc[rets.index.isin(test_period_dates)]


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

for strategy in strategy_names:
    for window in windows:
        print(f"Processing for {strategy}, {window}")
        visualize_regimes(
            test_data=train_data[strategy][[f"Sharpe_Ratio_{window}"]],
            test_regime_labels=vlstar_train_regime_labels,
            strategy=strategy,
            data_type="Train",
            window=window,
        )
        visualize_regimes(
            test_data=test_data[strategy][[f"Sharpe_Ratio_{window}"]],
            test_regime_labels=vlstar_test_regime_labels,
            strategy=strategy,
            data_type="Test",
            window=window,
        )
        plot_yield_curves(
            test_sharpe=test_data[strategy][[f"Sharpe_Ratio_{window}"]],
            test_regime_labels=vlstar_test_regime_labels[window],
            strategy=strategy,
            window=window,
            rets_df=rets,
        )
