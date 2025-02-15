import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn_extra.cluster import KMedoids

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# GLOBALS: Adjust them as needed
# ---------------------------------------------------------------------
period = 100
initial_train_start = "2023-03-01"
train_window_length = 10
test_window_length = 5
final_test_end = "2023-03-30"

plot_start_date = "2023-03-21"
plot_end_date = "2023-03-22"

# ---------------------------------------------------------------------
# SUPPORTING FUNCTIONS (same as your current code)
# ---------------------------------------------------------------------


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(axis=0, how="any", inplace=True)
    df.sort_index(inplace=True)
    return df


def generate_rolling_windows(df, initial_start, train_len, test_len, final_end):
    windows = []
    current_train_start = pd.to_datetime(initial_start)
    current_train_end = current_train_start + pd.Timedelta(days=train_len - 1)
    current_test_start = current_train_end + pd.Timedelta(days=1)
    current_test_end = current_test_start + pd.Timedelta(days=test_len - 1)

    while current_test_end <= pd.to_datetime(final_end):
        windows.append(
            (
                current_train_start,
                current_train_end,
                current_test_start,
                current_test_end,
            )
        )
        current_train_start += pd.Timedelta(days=test_len)
        current_train_end = current_train_start + pd.Timedelta(days=train_len - 1)
        current_test_start = current_train_end + pd.Timedelta(days=1)
        current_test_end = current_test_start + pd.Timedelta(days=test_len - 1)

    return windows


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


def predict_hmm(model, X):
    hidden_states = model.predict(X)
    return hidden_states


def map_hmm_states_to_regimes(
    X_train, hidden_states, df_train, key_feature="mr_strength_ar_100"
):
    n_samples = len(df_train)
    if len(hidden_states) != n_samples:
        raise ValueError("Mismatch between hidden_states length and df_train rows.")

    state_to_values = {}
    for i, state in enumerate(hidden_states):
        val = df_train.iloc[i][key_feature]
        state_to_values.setdefault(state, []).append(val)

    state_means = [(state, np.mean(vals)) for state, vals in state_to_values.items()]
    sorted_states = sorted(state_means, key=lambda x: x[1])

    regime_names = ["Low", "Normal", "High"]
    state_to_regime = {}
    for i, (st, _) in enumerate(sorted_states):
        if i < len(regime_names):
            state_to_regime[st] = regime_names[i]
        else:
            state_to_regime[st] = f"Extra_{st}"

    return state_to_regime


def fit_vlstar_kmedoids(X_train, n_clusters=3, random_state=42):
    kmed = KMedoids(
        n_clusters=n_clusters,
        metric="euclidean",
        init="k-medoids++",
        random_state=random_state,
    )
    kmed.fit(X_train)
    return kmed


def map_clusters_to_regimes_kmedoids(kmed, df_train, key_feature=f"mr_strength_ar_100"):
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
    # identical to your existing code for plotting
    regime_labels = np.array(regime_labels)
    zoom_mask = (df.index >= pd.to_datetime(plot_start_date)) & (
        df.index <= pd.to_datetime(plot_end_date)
    )
    zoomed_df = df.loc[zoom_mask].copy()
    zoomed_labels = regime_labels[zoom_mask]

    if len(zoomed_df) == 0:
        print(f"No data found for date range {plot_start_date} to {plot_end_date}")
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{title} ({plot_start_date} to {plot_end_date})", fontsize=13, y=0.98)

    # Subplot A: Spread
    axes[0].plot(zoomed_df.index, zoomed_df["spread"], color="black", label="Spread")
    colors = [REGIME_COLOR_MAPPING.get(r, "gray") for r in zoomed_labels]
    axes[0].scatter(
        zoomed_df.index, zoomed_df["spread"], c=colors, alpha=0.6, marker="o"
    )
    unique_regs = list(set(zoomed_labels))
    handles, labels_ = [], []
    for reg in unique_regs:
        color = REGIME_COLOR_MAPPING.get(reg, "gray")
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                label=reg,
                markersize=8,
            )
        )
        labels_.append(reg)
    axes[0].legend(handles, labels_, loc="best", fontsize=9)
    axes[0].set_ylabel("Spread", fontsize=9)
    axes[0].set_title("Spread (Regime-Colored)", fontsize=10)

    # Subplot B: AR Strength
    if f"mr_strength_ar_{period}" in zoomed_df.columns:
        axes[1].plot(
            zoomed_df.index,
            zoomed_df[f"mr_strength_ar_{period}"],
            label=f"AR Strength ({period})",
            color="blue",
        )
        axes[1].set_title(f"AR Strength ({period})", fontsize=10)
        axes[1].legend(loc="best", fontsize=9)
    else:
        axes[1].text(
            0.5, 0.5, f"mr_strength_ar_{period} not found", ha="center", va="center"
        )
    axes[1].set_ylabel(f"AR({period})", fontsize=9)

    # Subplot C: Hurst
    if f"hurst_{period}" in zoomed_df.columns:
        axes[2].plot(
            zoomed_df.index,
            zoomed_df[f"hurst_{period}"],
            label=f"Hurst ({period})",
            color="orange",
        )
        axes[2].set_title(f"Hurst Exponent ({period})", fontsize=10)
        axes[2].legend(loc="best", fontsize=9)
    else:
        axes[2].text(0.5, 0.5, f"hurst_{period} not found", ha="center", va="center")
    axes[2].set_ylabel(f"Hurst({period})", fontsize=9)

    # Subplot D: Variance Ratio
    if f"var_ratio_{period}" in zoomed_df.columns:
        axes[3].plot(
            zoomed_df.index,
            zoomed_df[f"var_ratio_{period}"],
            label=f"Variance Ratio ({period})",
            color="green",
        )
        axes[3].set_title(f"Variance Ratio ({period})", fontsize=10)
        axes[3].legend(loc="best", fontsize=9)
    else:
        axes[3].text(
            0.5, 0.5, f"var_ratio_{period} not found", ha="center", va="center"
        )
    axes[3].set_ylabel(f"VR({period})", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def append_predictions_to_df(df, test_indices, regime_hmm, regime_vlstar):
    if "regime_hmm" not in df.columns:
        df["regime_hmm"] = np.nan
    if "regime_vlstar" not in df.columns:
        df["regime_vlstar"] = np.nan

    df.loc[test_indices, "regime_hmm"] = regime_hmm
    df.loc[test_indices, "regime_vlstar"] = regime_vlstar
    return df


# ---------------------------------------------------------------------
# FUNCTION TO PROCESS A SINGLE CSV (replacement for the old main())
# ---------------------------------------------------------------------
def process_csv(
    csv_path,
    initial_train_start=initial_train_start,
    train_window_length=train_window_length,
    test_window_length=test_window_length,
    final_test_end=final_test_end,
    period=period,
    do_plots=False,
):
    """
    Processes a single CSV file: loads data, runs rolling windows,
    fits HMM and KMedoids, appends predictions, optionally does plots.
    Saves the resulting file as {original_name}_output.csv in the same folder.
    """
    print(f"Processing file: {csv_path}")
    df = load_and_preprocess(csv_path)

    # Generate rolling windows
    windows = generate_rolling_windows(
        df,
        initial_start=initial_train_start,
        train_len=train_window_length,
        test_len=test_window_length,
        final_end=final_test_end,
    )

    all_test_indices = []
    all_regime_hmm = []
    all_regime_vlstar = []

    # Rolling windows
    for idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"  --- Rolling Window {idx + 1} ---")
        print(f"    Train: {train_start.date()} to {train_end.date()}")
        print(f"    Test:  {test_start.date()} to {test_end.date()}")

        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]

        feature_cols = [
            f"mr_strength_ar_{period}",
            f"hurst_{period}",
            f"var_ratio_{period}",
        ]
        key_feature = f"mr_strength_ar_{period}"

        # If any feature is missing, skip this window
        if any(fc not in train_df.columns for fc in feature_cols):
            print("  -> Some features are missing in train set. Skipping...")
            continue
        if any(fc not in test_df.columns for fc in feature_cols):
            print("  -> Some features are missing in test set. Skipping...")
            continue

        train_df_features = train_df[feature_cols].copy()
        test_df_features = test_df[feature_cols].copy()

        X_train, scaler = scale_fit_transform(train_df_features)
        X_test = scale_transform(test_df_features, scaler)

        # HMM
        hmm_model = fit_hmm(
            X_train,
            n_components=3,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
        )
        hidden_states_train = predict_hmm(hmm_model, X_train)
        state_to_regime = map_hmm_states_to_regimes(
            X_train,
            hidden_states_train,
            train_df,
            key_feature=key_feature,
        )
        hidden_states_test = predict_hmm(hmm_model, X_test)
        regime_labels_test_hmm = [
            state_to_regime.get(s, "Unknown") for s in hidden_states_test
        ]

        # KMedoids
        kmed = fit_vlstar_kmedoids(X_train, n_clusters=3, random_state=42)
        cluster_to_regime, _ = map_clusters_to_regimes_kmedoids(
            kmed, train_df, key_feature=key_feature
        )
        cluster_labels_test = predict_vlstar_kmedoids(kmed, X_test)
        regime_labels_test_vlstar = [
            cluster_to_regime.get(c, "Unknown") for c in cluster_labels_test
        ]

        test_indices = test_df.index
        all_test_indices.extend(test_indices)
        all_regime_hmm.extend(regime_labels_test_hmm)
        all_regime_vlstar.extend(regime_labels_test_vlstar)

    # Build a DataFrame of predictions
    predictions_df = pd.DataFrame(
        {"regime_hmm": all_regime_hmm, "regime_vlstar": all_regime_vlstar},
        index=all_test_indices,
    )
    predictions_df = predictions_df[~predictions_df.index.duplicated(keep="last")]

    # Append to df
    df = append_predictions_to_df(
        df,
        predictions_df.index,
        predictions_df["regime_hmm"],
        predictions_df["regime_vlstar"],
    )

    # Save to new CSV in same folder
    csv_path = Path(csv_path)
    output_csv = csv_path.parent / f"{csv_path.stem}_output.csv"
    df.to_csv(output_csv)
    print(f"  -> Saved results to {output_csv}")

    # Visualization (optional)
    if do_plots:
        # Combine test DataFrame
        combined_predictions = predictions_df.copy()
        combined_predictions.sort_index(inplace=True)
        combined_test_df = df.loc[combined_predictions.index]

        visualize_regimes(
            combined_test_df,
            combined_predictions["regime_hmm"],
            title="HMM Mean Reversion Regimes (Test Set)",
        )
        visualize_regimes(
            combined_test_df,
            combined_predictions["regime_vlstar"],
            title="VLSTAR (KMedoids) Mean Reversion Regimes (Test Set)",
        )


# ---------------------------------------------------------------------
# PROCESS ALL CSV FILES IN A FOLDER (IN PARALLEL)
# ---------------------------------------------------------------------
def process_all_files_in_folder(
    folder_path,
    pattern="*.csv",
    do_plots=False,
    max_workers=4,
):
    """
    Finds all CSV files matching `pattern` in `folder_path`,
    processes them in parallel using `process_csv`.
    """
    folder = Path(folder_path)
    csv_files = list(folder.glob(pattern))
    if not csv_files:
        print(f"No CSV files found in {folder_path} matching pattern '{pattern}'")
        return

    # We'll pass the same hyperparams for all files,
    # but you could make them configurable:
    task_args = {
        "initial_train_start": initial_train_start,
        "train_window_length": train_window_length,
        "test_window_length": test_window_length,
        "final_test_end": final_test_end,
        "period": period,
        "do_plots": do_plots,
    }

    print(f"Found {len(csv_files)} file(s) in {folder_path}. Processing in parallel...")

    # Using ProcessPoolExecutor to parallelize
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for csv_path in csv_files:
            futures.append(executor.submit(process_csv, csv_path, **task_args))

        # Optionally gather results (if any). We'll just wait for them all to finish.
        for f in futures:
            f.result()  # re-raise any exceptions that happened during processing

    print("All done processing all files in parallel.")


# ---------------------------------------------------------------------
# Example usage (if you run this file directly)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose your folder is "data/"
    process_all_files_in_folder("data", pattern="*.csv", do_plots=False, max_workers=4)
