import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import matplotlib

matplotlib.use("TkAgg")


def detect_anomalies_agg_returns(
    df, L=10, rolling_window=50, threshold=2, method="mean"
):
    returns = df["close"].pct_change().fillna(0)
    if method == "sum":
        S = returns.rolling(window=L).sum()
    else:
        S = returns.rolling(window=L).mean()
    S = S.dropna()
    med = S.rolling(window=rolling_window, center=True).median()
    mad = S.rolling(window=rolling_window, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    robust_z = (S - med) / (1.4826 * mad)
    agg_anomalies = S.index[np.abs(robust_z) > threshold]
    return agg_anomalies


def plot_anomalies(df, anomalies, num_plots=1, duration="1D"):
    df = df.copy()
    df["anomaly"] = df.index.isin(anomalies)
    t = df.index.isin(anomalies)
    min_time, max_time = df.index.min(), df.index.max() - pd.Timedelta(duration)
    start_times = [
        min_time
        + pd.Timedelta(
            seconds=random.randint(0, int((max_time - min_time).total_seconds()))
        )
        for _ in range(num_plots)
    ]

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=False)
    if num_plots == 1:
        axes = [axes]

    for ax, start_time in zip(axes, start_times):
        end_time = start_time + pd.Timedelta(duration)
        segment = df.loc[start_time:end_time]

        if segment.empty:
            continue

        ax.plot(segment.index, segment["close"], label="Close Price", color="blue")

        anomalies_segment = segment[segment["anomaly"]]
        ax.scatter(
            anomalies_segment.index,
            anomalies_segment["close"],
            color="red",
            label="Anomaly",
            marker="o",
        )

        ax.set_title(
            f"Random Period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()


def main():
    file_path = "/Users/alexanderdemachev/PycharmProjects/strategy/data/futures/1min/BTCUSDT.csv"
    data = pd.read_csv(file_path)

    data["datetime"] = pd.to_datetime(data["datetime"])
    data.set_index("datetime", inplace=True)

    last_six_months = data.index.max() - pd.DateOffset(months=6)
    data = data[data.index >= last_six_months]

    anomalies = detect_anomalies_agg_returns(data, L=13, rolling_window=289)

    plot_anomalies(data, anomalies, num_plots=5, duration="1D")


if __name__ == "__main__":
    main()
