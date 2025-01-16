import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df = df.drop(columns=[col for col in df.columns if col.endswith("1000")])
    df.dropna(axis=1, how="all", inplace=True)

    df.dropna(axis=0, how="any", inplace=True)

    df.sort_index(inplace=True)

    return df


def split_train_test(df):
    train_start = "2023-01-01"
    train_end = "2023-01-15"
    test_end = "2023-01-25"

    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end)
    test_end = pd.to_datetime(test_end)

    # Create train and test DataFrames based on the specified date ranges
    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    test_df = df[(df.index > train_end) & (df.index <= test_end)]

    return train_df, test_df


def scale_fit_transform(df):
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df.values)
    return scaled_array, scaler


def scale_transform(df, scaler):
    scaled_array = scaler.transform(df.values)
    return scaled_array


def fit_hmm(
    X_train, n_components=3, covariance_type="full", n_iter=1000, random_state=42
):
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X_train)
    return model


def map_hmm_states_to_regimes(
    X_train, hidden_states, df_train, key_feature="mr_strength_ar_100"
):
    n_samples = len(df_train)
    if len(hidden_states) != n_samples:
        raise ValueError("Mismatch between hidden_states length and df_train rows.")

    state_to_values = {}
    for i, state in enumerate(hidden_states):
        val = df_train.iloc[i][key_feature]
        if state not in state_to_values:
            state_to_values[state] = []
        state_to_values[state].append(val)

    state_means = []
    for state, vals in state_to_values.items():
        state_means.append((state, np.mean(vals)))

    sorted_states = sorted(state_means, key=lambda x: x[1])

    regime_names = ["Low", "Normal", "High"]
    state_to_regime = {}
    for i, (st, _) in enumerate(sorted_states):
        if i < len(regime_names):
            state_to_regime[st] = regime_names[i]
        else:
            state_to_regime[st] = f"Extra_{st}"

    return state_to_regime


def predict_hmm(model, X):
    hidden_states = model.predict(X)
    return hidden_states


def fit_vlstar_kmedoids(X_train, n_clusters=3, random_state=42):
    kmed = KMedoids(
        n_clusters=n_clusters,
        metric="euclidean",
        init="k-medoids++",
        random_state=random_state,
    )
    kmed.fit(X_train)
    return kmed


def map_clusters_to_regimes_kmedoids(kmed, df_train, key_feature="mr_strength_ar_100"):
    labels = kmed.labels_
    df_tmp = df_train.copy()
    df_tmp["Cluster"] = labels

    cluster_means = df_tmp.groupby("Cluster")[key_feature].mean().sort_values()
    sorted_clusters = cluster_means.index.tolist()

    regime_names = ["Low", "Normal", "High"]
    cluster_to_regime = {}
    for i, cl in enumerate(sorted_clusters):
        if i < len(regime_names):
            cluster_to_regime[cl] = regime_names[i]
        else:
            cluster_to_regime[cl] = f"Extra_{cl}"

    return cluster_to_regime, labels


def predict_vlstar_kmedoids(kmed, X_test):
    return kmed.predict(X_test)


REGIME_COLOR_MAPPING = {"Low": "red", "Normal": "blue", "High": "green"}


def visualize_regimes(df, regime_labels, title="Regime Visualization"):
    if len(regime_labels) != len(df):
        raise ValueError("Mismatch between regime_labels length and df rows.")

    start_date = "2023-01-17"
    end_date = "2023-01-18"

    regime_labels = np.array(regime_labels)

    zoomed_df = df[
        (df.index >= pd.to_datetime(start_date))
        & (df.index <= pd.to_datetime(end_date))
    ]

    zoomed_labels = regime_labels[
        (df.index >= pd.to_datetime(start_date))
        & (df.index <= pd.to_datetime(end_date))
    ]

    plt.figure(figsize=(12, 6))
    plt.plot(zoomed_df.index, zoomed_df["spread"], color="black", label="Spread")

    colors = [REGIME_COLOR_MAPPING.get(r, "gray") for r in zoomed_labels]
    plt.scatter(zoomed_df.index, zoomed_df["spread"], c=colors, alpha=0.6, marker="o")

    unique_regs = list(set(zoomed_labels))
    handles = []
    labels = []
    for r in unique_regs:
        color = REGIME_COLOR_MAPPING.get(r, "gray")
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                label=r,
                markersize=8,
            )
        )
        labels.append(r)

    plt.legend(handles, labels)
    plt.title(f"{title} ({start_date} to {end_date})")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.tight_layout()
    plt.show()


def main():
    csv_path = "AAVEUSDT.csv"

    df = load_and_preprocess(csv_path)

    train_df, test_df = split_train_test(df)

    X_train, scaler = scale_fit_transform(train_df[feature_cols])
    X_test = scale_transform(test_df[feature_cols], scaler)

    # ----------------- HMM ------------------
    hmm_model = fit_hmm(
        X_train, n_components=3, covariance_type="full", n_iter=1000, random_state=42
    )
    hidden_states_train = predict_hmm(hmm_model, X_train)
    state_to_regime = map_hmm_states_to_regimes(
        X_train,
        hidden_states_train,
        train_df,
        key_feature="mr_strength_ar_100",
    )

    hidden_states_test = predict_hmm(hmm_model, X_test)
    regime_labels_test_hmm = [state_to_regime[s] for s in hidden_states_test]
    kmed = fit_vlstar_kmedoids(X_train, n_clusters=3, random_state=42)
    cluster_to_regime, cluster_labels_train = map_clusters_to_regimes_kmedoids(
        kmed, train_df, key_feature="mr_strength_ar_100"
    )
    cluster_labels_test = predict_vlstar_kmedoids(kmed, X_test)
    regime_labels_test_vlstar = [cluster_to_regime[c] for c in cluster_labels_test]

    visualize_regimes(
        test_df, regime_labels_test_hmm, title="HMM Mean Reversion Regimes (Test Set)"
    )

    visualize_regimes(
        test_df,
        regime_labels_test_vlstar,
        title="VLSTAR (KMedoids) Mean Reversion Regimes (Test Set)",
    )

    print("All done.")


if __name__ == "__main__":
    main()
