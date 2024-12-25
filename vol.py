import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from dtaidistance import dtw
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


CSV_FILE = "all_features_BTCUSDT.csv"

VAR_NAMES = [
    "Historical_Volatility",
    "Realized_Volatility",
    "EWMA_Volatility",
    "Parkinson_Volatility",
    "Garman_Klass_Volatility",
    "Volatility_of_Volatility",
]

WINDOWS = [10, 30, 60, 90]

N_CLUSTERS = 3

REGIME_LABELS = ["Low", "Normal", "High"]

FREQUENCY = "6M"

START_TEST_DATE = "2021-01-01"


def main():
    df = pd.read_csv(
        CSV_FILE,
        index_col=0,
        parse_dates=[0],
    )
    df = df.dropna(axis=1, thresh=0.5 * len(df))
    df = df.dropna()
    df = df.sort_index()

    rolling_splits = create_rolling_windows(df, START_TEST_DATE, freq=FREQUENCY)

    for VAR_NAME in VAR_NAMES:
        for window in WINDOWS:
            col_name = f"{VAR_NAME}_{window}"
            if col_name not in df.columns:
                print(f"{col_name} not found")
                continue

            print(f"\n Feature: {col_name}, window: {window}\n")
            combined_test_df = pd.DataFrame()
            combined_test_labels = []

            for train_end_date, test_start, test_end in rolling_splits:
                train_df = df[df.index < test_start].copy()
                test_df_chunk = df[
                    (df.index >= test_start) & (df.index < test_end)
                ].copy()

                if test_df_chunk.empty:
                    continue

                kmedoids_model, scaler, train_labels, X_train_scaled = fit_vlstar(
                    train_df, VAR_NAME, window, n_clusters=N_CLUSTERS
                )

                if kmedoids_model is None:
                    print("Model fitting failed")
                    continue

                regime_mapping = map_clusters_to_regimes(
                    train_df,
                    train_labels,
                    VAR_NAME,
                    window,
                    regime_labels=REGIME_LABELS,
                )

                test_labels = assign_test_labels(
                    kmedoids_model,
                    scaler,
                    X_train_scaled,
                    test_df_chunk,
                    VAR_NAME,
                    window,
                    n_clusters=N_CLUSTERS,
                )

                combined_test_df = pd.concat([combined_test_df, test_df_chunk])
                combined_test_labels.extend(test_labels)

                combined_test_df = combined_test_df.sort_index()

            visualize_combined_regimes(
                combined_test_df,
                VAR_NAME,
                window,
                combined_test_labels,
                regime_mapping,
            )


def create_rolling_windows(df, start_test="2021-01-01", freq="6M"):
    max_date = df.index.max()
    date_ranges = pd.date_range(start=start_test, end=max_date, freq=freq)

    window_splits = []
    for i in range(len(date_ranges) - 1):
        test_start = date_ranges[i]
        test_end = date_ranges[i + 1]
        window_splits.append((test_start, test_start, test_end))

    if len(date_ranges) > 0 and date_ranges[-1] < max_date:
        test_start = date_ranges[-1]
        test_end = max_date + pd.Timedelta(days=1)  # up to next day
        window_splits.append((test_start, test_start, test_end))

    return window_splits


def pad_truncate_sequences(sequence, max_length):
    if len(sequence) < max_length:
        return np.pad(sequence, (0, max_length - len(sequence)), "constant")
    else:
        return sequence[:max_length]


def scale_sequences_fit_transform(X):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    return scaled_data, scaler


def scale_sequences_transform(X, scaler):
    return scaler.transform(X)


def fit_vlstar(train_df, var, window, n_clusters=3):
    col_name = f"{var}_{window}"
    if col_name not in train_df.columns or train_df.empty:
        return None, None, np.array([]), np.array([])

    X_train = train_df[col_name].values
    X_train = X_train.reshape(-1, 1)

    X_train_scaled, scaler = scale_sequences_fit_transform(X_train)

    n_train = X_train_scaled.shape[0]
    distance_matrix = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(i + 1, n_train):
            distance = dtw.distance(X_train_scaled[i], X_train_scaled[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    kmedoids = KMedoids(
        n_clusters=n_clusters,
        metric="precomputed",
        init="k-medoids++",
        random_state=42,
    )
    kmedoids.fit(distance_matrix)

    train_labels = kmedoids.labels_

    return kmedoids, scaler, train_labels, X_train_scaled


def assign_test_labels(
    kmedoids_model, scaler, train_scaled, test_df, var, window, n_clusters=3
):
    if kmedoids_model is None or test_df.empty:
        return np.array([])

    col_name = f"{var}_{window}"
    if col_name not in test_df.columns:
        return np.array([])

    X_test = test_df[col_name].values.reshape(-1, 1)
    X_test_scaled = scale_sequences_transform(X_test, scaler)

    # get medoids from the training set
    medoid_indices = kmedoids_model.medoid_indices_
    medoids = train_scaled[medoid_indices]  # shape (n_clusters, 1)

    # DTW distance to each medoid
    n_test = X_test_scaled.shape[0]
    distances_test = np.zeros((n_test, n_clusters))
    for i in range(n_test):
        for j in range(n_clusters):
            distances_test[i, j] = dtw.distance(X_test_scaled[i], medoids[j])

    test_labels = distances_test.argmin(axis=1)
    return test_labels


def map_clusters_to_regimes(
    train_df, train_labels, var, window, regime_labels=["Low", "Normal", "High"]
):
    col_name = f"{var}_{window}"
    if col_name not in train_df.columns or train_df.empty or len(train_labels) == 0:
        return {}

    temp_df = train_df.copy()
    temp_df["Cluster"] = train_labels

    cluster_means = temp_df.groupby("Cluster")[col_name].mean()
    sorted_clusters = cluster_means.sort_values().index.tolist()

    regime_mapping = {}
    num_regimes = len(regime_labels)

    if len(sorted_clusters) >= num_regimes:
        for cluster_id, label in zip(sorted_clusters[:num_regimes], regime_labels):
            regime_mapping[cluster_id] = label
        for cluster_id in sorted_clusters[num_regimes:]:
            regime_mapping[cluster_id] = f"Extra_{cluster_id}"
    else:
        for cluster_id, label in zip(sorted_clusters, regime_labels):
            regime_mapping[cluster_id] = label

    return regime_mapping


REGIME_COLOR_MAPPING = {
    "Low": "red",
    "Normal": "blue",
    "High": "green",
}


def get_regime_colors(label_series):
    return [REGIME_COLOR_MAPPING.get(lbl, "gray") for lbl in label_series]


def visualize_combined_regimes(
    combined_test_df,
    var,
    window,
    combined_test_labels,
    regime_mapping,
):
    if combined_test_df.empty or len(combined_test_labels) == 0:
        print("No test data or labels to visualize.")
        return

    col_name = f"{var}_{window}"
    combined_test_regime_labels = [
        regime_mapping.get(lbl, "Noise") for lbl in combined_test_labels
    ]

    df_plot = combined_test_df.copy()
    df_plot["Regime_Label"] = combined_test_regime_labels
    df_plot["datetime"] = df_plot.index

    plt.figure(figsize=(15, 7))
    plt.plot(
        df_plot["datetime"],
        df_plot[col_name],
        color="black",
        linewidth=1,
        label=f"{col_name}",
    )

    colors = get_regime_colors(df_plot["Regime_Label"])
    plt.scatter(
        df_plot["datetime"],
        df_plot[col_name],
        c=colors,
        alpha=0.6,
        marker="o",
        label="Regime",
    )

    unique_labels = df_plot["Regime_Label"].unique()
    handles = []
    labels = []
    for regime in unique_labels:
        color = REGIME_COLOR_MAPPING.get(regime, "gray")
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color, markersize=8
            )
        )
        labels.append(regime)

    plt.legend(handles, labels, title="Regimes")
    plt.title(f"{col_name} Over Time with window={window} (Test)")
    plt.xlabel("Date")
    plt.ylabel(col_name)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"graph/volatility/{col_name}.png")


if __name__ == "__main__":
    main()
