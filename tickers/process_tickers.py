import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

import umap.umap_ as umap

from tqdm import tqdm


def estimate_ar1_params(series):
    X = np.asarray(series)
    if len(X) < 2:
        return np.nan, np.nan
    Y = X[1:]
    Z = X[:-1]
    if len(Z) == 0:
        return np.nan, np.nan
    A = np.column_stack((Z, np.ones_like(Z)))
    coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
    phi = coeffs[0]
    c = coeffs[1]
    return phi, c


def discrete_mean_reversion_strength(phi):
    if phi >= 1 or np.isnan(phi):
        return 0.0
    return -np.log(phi)


def estimate_hurst_dfa(y, q=2, order=1, fit_range=(4, 20)):
    try:
        from MFDFA import MFDFA
    except ImportError:
        return np.nan

    try:
        lag = np.unique((np.logspace(0.7, 2, 30)).astype(int))
        if len(y) < max(lag):
            return np.nan
        lag, dfa = MFDFA(y, lag=lag, q=q, order=order)
        dfa = dfa.flatten()
        log_lag = np.log(lag)
        log_dfa = np.log(dfa)
        start, end = fit_range
        end = min(end, len(lag))
        slope, _ = np.polyfit(log_lag[start:end], log_dfa[start:end], 1)
        return slope
    except:
        return np.nan


import ta
from scipy.stats import linregress


def calculate_rsi(df, window=14):
    rsi = ta.momentum.RSIIndicator(close=df["close"], window=window)
    return rsi.rsi()


def calculate_momentum(df, window=10):
    """
    Uses ROC from ta library as a momentum proxy.
    """
    roc = ta.momentum.ROCIndicator(close=df["close"], window=window)
    return roc.roc()


def calculate_macd(df, window_slow=26, window_fast=12, window_sign=9):
    macd = ta.trend.MACD(
        close=df["close"],
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
    )
    macd_df = pd.DataFrame(
        {
            "macd": macd.macd(),
            "macd_signal": macd.macd_signal(),
            "macd_diff": macd.macd_diff(),
        },
        index=df.index,
    )
    return macd_df


def rolling_hurst_dfa(series, window):
    hurst_vals = []
    for i in range(len(series)):
        if i < window:
            hurst_vals.append(np.nan)
        else:
            window_data = series[i - window : i]
            h = estimate_hurst_dfa(window_data)
            hurst_vals.append(h)
    return pd.Series(hurst_vals, index=series.index)


def rolling_mr_strength_ar(series, window):
    """
    For each index >= window, estimate AR(1) and get discrete_mean_reversion_strength from phi.
    """
    mr_vals = []
    for i in range(len(series)):
        if i < window:
            mr_vals.append(np.nan)
        else:
            window_data = series[i - window : i]
            phi, _ = estimate_ar1_params(window_data)
            mr_strength = discrete_mean_reversion_strength(phi)
            mr_vals.append(mr_strength)
    return pd.Series(mr_vals, index=series.index)


def compute_features_for_ticker(ticker, data_dir, periods):
    csv_path = os.path.join(data_dir, f"{ticker}.csv")
    if not os.path.isfile(csv_path):
        print(f"CSV for {ticker} not found at {csv_path}. Skipping.")
        return None

    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=[0],
            skiprows=[1],
        )
    except Exception as e:
        print(f"Error reading {ticker}: {e}")
        return None

    df.rename(columns={df.columns[0]: "datetime", "Close": "close"}, inplace=True)
    df.set_index("datetime", inplace=True, drop=True)
    df.sort_index(inplace=True)

    numeric_cols = ["close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)

    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["log_returns"] = df["log_returns"].replace([np.inf, -np.inf], np.nan)

    for p in periods:
        df[f"log_returns_{p}"] = np.log(df["close"] / df["close"].shift(p))
        df[f"log_returns_{p}"] = df[f"log_returns_{p}"].replace(
            [np.inf, -np.inf], np.nan
        )

        df[f"Hurst_{p}"] = rolling_hurst_dfa(df["close"], p)
        df[f"MR_Strength_{p}"] = rolling_mr_strength_ar(df["close"], p)

        df[f"RSI_{p}"] = calculate_rsi(df, window=p)

        df[f"Momentum_{p}"] = calculate_momentum(df, window=p)

        macd_df = calculate_macd(
            df, window_slow=p, window_fast=max(1, p // 2), window_sign=9
        )
        df[f"MACD_{p}"] = macd_df["macd"]
        df[f"MACD_Signal_{p}"] = macd_df["macd_signal"]
        df[f"MACD_Diff_{p}"] = macd_df["macd_diff"]

        # Rolling volatility of log_returns
        df[f"Volatility_logreturns_{p}"] = df["log_returns"].rolling(p).std()

    return df


def add_pca_features(df, n_components=2, prefix="pca"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols]

    data_to_reduce = df[numeric_cols].fillna(0)

    if data_to_reduce.empty:
        print("no valid numeric data")
        for i in range(n_components):
            df[f"{prefix}_{i+1}"] = np.nan
        return df

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data_to_reduce)

    for i in range(n_components):
        df.loc[data_to_reduce.index, f"{prefix}_{i+1}"] = transformed[:, i]

    return df


def add_umap_features(df, n_components=2, prefix="umap"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    numeric_cols = [col for col in numeric_cols]

    data_to_reduce = df[numeric_cols].fillna(0)

    if data_to_reduce.empty:
        print("no valid numeric data for UMAP")
        for i in range(n_components):
            df[f"{prefix}_{i+1}"] = np.nan
        return df

    try:
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        transformed = reducer.fit_transform(data_to_reduce)

        for i in range(n_components):
            df.loc[data_to_reduce.index, f"{prefix}_{i+1}"] = transformed[:, i]
    except Exception as e:
        print(f"Error during UMAP computation: {e}")
        for i in range(n_components):
            df[f"{prefix}_{i+1}"] = np.nan

    return df


def main():
    tickers = ["SPX", "MDY", "QQQ", "VIX", "BTC-USD", "ETH-USD", "CBOE", "TNX"]

    data_dir = "market_data_csv"
    output_dir = "feature_datasets"
    os.makedirs(output_dir, exist_ok=True)

    periods = [13, 34, 55]

    for ticker in tqdm(tickers, desc="Processing Tickers"):
        df = compute_features_for_ticker(ticker, data_dir, periods)
        if df is None or df.empty:
            continue

        df_pca = df.copy()
        df_pca = add_pca_features(df_pca, n_components=7, prefix="pca")

        df_umap = df.copy()
        df_umap = add_umap_features(df_umap, n_components=7, prefix="umap")

        pca_out_path = os.path.join(output_dir, f"{ticker}_features_pca.csv")
        df_pca.to_csv(pca_out_path, index=True)
        umap_out_path = os.path.join(output_dir, f"{ticker}_features_umap.csv")
        df_umap.to_csv(umap_out_path, index=True)

        print(f"Saved {pca_out_path} and {umap_out_path}")

    print("\nAll tickers processed successfully.")


if __name__ == "__main__":
    main()
