import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import random
from datetime import timedelta
from pandas.tseries.offsets import DateOffset
from scipy.stats import percentileofscore, zscore
import ta
import os
from typing import List, Tuple
from models import *
from backtest_optimizer.main import (
    ParameterOptimizer,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

file_path = "BTCUSDT.csv"
data = pd.read_csv(file_path)

data["datetime"] = pd.to_datetime(data["datetime"])
data.set_index("datetime", inplace=True)

sequence_length = 100


def detect_anomalies(mse, num_std):
    """Detect anomalies in test data using thresholding."""
    threshold = np.mean(mse) + num_std * np.std(mse)
    # print("Threshold = ", np.round(threshold, 2))
    anomalies = np.where(mse > np.round(threshold, 2))[0]
    return anomalies


def deviation_from_mean(data, value):
    # z_scores = zscore(data)
    deviation = (value - np.mean(data)) / np.std(data)
    return deviation


def simulate_trading_std(
    model_name,
    price_data,
    scaled_prices,
    predicted_prices,
    ma_data,
    rsi_data,
    scores,
    rolling_anomalies,
    num_std,
    max_entries,
    exit_val,
    distr_len=99,
    num_std_exit=1,
    rsi_entry=[40, 60],
    rsi_exit=[60, 40],
):
    price_array = price_data.to_numpy(dtype=np.float64)
    ma_array = ma_data.to_numpy(dtype=np.float64)
    rsi_data = rsi_data.to_numpy(dtype=np.float64)
    anomalies_array = np.asarray(rolling_anomalies, dtype=bool)

    index_array = price_data.index.to_numpy()

    n = len(price_array)
    if len(rsi_data) != n or len(anomalies_array) != n:
        raise ValueError(
            "price_data, ma_data, and rolling_anomalies must have the same length."
        )

    rolling_std_series = price_data.rolling(distr_len, min_periods=1).std()
    rolling_std = rolling_std_series.values

    initial_capital = 1.0
    capital = initial_capital
    capital_history_array = np.full(n, np.nan, dtype=np.float64)
    capital_history_array[0] = capital

    open_trades = []
    trade_entries = set()
    closed_trades = []

    base_price = None
    k = 0
    base_trade_type = None

    for i in range(n):
        timestamp = index_array[i]
        current_price = price_array[i]
        current_ma = ma_array[i]
        current_rsi = rsi_data[i]

        local_std_exit = rolling_std[i]

        if open_trades:
            trades_to_close = []

            for trade in open_trades:
                trade_type = trade["type"]
                entry_price = trade["entry_price"]

                if exit_val == "rsi":
                    if (trade_type == "buy" and current_rsi >= rsi_exit[0]) or (
                        trade_type == "sell" and current_rsi <= rsi_exit[1]
                    ):
                        trades_to_close.append(trade)
                        continue
                if exit_val == "ma":
                    if (trade_type == "buy" and current_price >= current_ma) or (
                        trade_type == "sell" and current_price <= current_ma
                    ):
                        trades_to_close.append(trade)
                        continue

                if trade_type == "buy":
                    if current_price >= (entry_price + local_std_exit * num_std_exit):
                        trades_to_close.append(trade)
                elif trade_type == "sell":
                    if current_price <= (entry_price - local_std_exit * num_std_exit):
                        trades_to_close.append(trade)

            for trade in trades_to_close:
                trade_type = trade["type"]
                entry_price = trade["entry_price"]
                exit_price = current_price
                trade_volume = trade["volume"]

                if trade_type == "buy":
                    profit = (exit_price - entry_price) * (trade_volume / entry_price)
                else:
                    profit = (entry_price - exit_price) * (trade_volume / entry_price)

                capital += profit
                capital_history_array[i] = capital

                print(
                    f"Trade executed: {trade_type} at {trade['entry_time']} "
                    f"(Entry Price: {entry_price:.2f}), exited at {timestamp} "
                    f"(Exit Price: {exit_price:.2f}), Volume: {trade_volume:.4f}, "
                    f"P&L: {profit:.4f}"
                )

                closed_trades.append(
                    {
                        "type": trade_type,
                        "entry_time": trade["entry_time"],
                        "entry_price": entry_price,
                        "exit_time": timestamp,
                        "exit_price": exit_price,
                        "profit": profit,
                    }
                )

            open_trades = [t for t in open_trades if t not in trades_to_close]
            k = len(open_trades)

        if anomalies_array[i] and len(open_trades) < max_entries:
            if k == 0:
                if timestamp not in trade_entries:
                    trade_entries.add(timestamp)

                    trade_type = None
                    scaled_price = scaled_prices[i]
                    predicted_price = predicted_prices[i]

                    if exit_val == "rsi":
                        if (
                            current_rsi < rsi_entry[0]
                            and predicted_price > scaled_price
                        ):
                            trade_type = "buy"
                        elif (
                            current_rsi > rsi_entry[1]
                            and predicted_price < scaled_price
                        ):
                            trade_type = "sell"
                    elif exit_val == "ma":
                        if (
                            current_price > current_ma
                            and predicted_price < scaled_price
                        ):
                            trade_type = "sell"
                        elif (
                            current_price < current_ma
                            and predicted_price > scaled_price
                        ):
                            trade_type = "buy"

                    if trade_type is not None:
                        trade_volume = capital / max_entries
                        open_trades.append(
                            {
                                "type": trade_type,
                                "volume": trade_volume,
                                "entry_price": current_price,
                                "entry_time": timestamp,
                            }
                        )
                        base_price = current_price
                        base_trade_type = trade_type
                        k = 1
            else:
                required_distance = k * num_std * local_std_exit

                if base_trade_type == "sell":
                    condition = current_price >= base_price + required_distance
                elif base_trade_type == "buy":
                    condition = current_price <= base_price - required_distance
                else:
                    condition = False

                if condition:
                    if timestamp not in trade_entries:
                        trade_entries.add(timestamp)
                        trade_type = base_trade_type
                        trade_volume = capital / max_entries
                        open_trades.append(
                            {
                                "type": trade_type,
                                "volume": trade_volume,
                                "entry_price": current_price,
                                "entry_time": timestamp,
                            }
                        )
                        k += 1

        capital_history_array[i] = capital

    capital_history = pd.Series(capital_history_array, index=price_data.index)
    capital_history.ffill(inplace=True)
    capital_history.fillna(capital, inplace=True)

    return capital_history, closed_trades


def calculate_rsi(df, window=14):
    rsi = ta.momentum.RSIIndicator(close=df["close"], window=window)
    rsi_values = rsi.rsi()
    rsi_values = np.maximum(rsi_values, 0)  # Ensure non-negative
    return rsi_values


def calc_pl(data_dict, params):
    results_dict = {}

    num_std = params.get("num_std", 1)
    num_std_exit = params.get("num_std_exit", 1)
    percentile = params.get("percentile", 99.5)
    exit_val = params.get("exit_val", "rsi")
    max_entries = params.get("max_entries", 5)
    plot_pl = params.get("plot_pl", False)
    distr_len = params.get("distr_len", 99)
    window_minutes = params.get("window_minutes", 500)
    window_size_minutes = params.get("window_size_minutes", 500)
    rsi_entry = params.get("rsi_entry", [40, 60])
    rsi_exit = params.get("rsi_exit", [60, 40])
    max_entries = params.get("max_entries", 5)

    for ticker, df in data_dict.items():
        print(f"\n=== Processing Ticker: {ticker} ===")
        price_data = df["close"]
        scaled_prices = df["scaled_price"]
        predicted_prices = df["predicted_price"]
        rsi_data = calculate_rsi(df, window=window_minutes)
        ma_data = df["close"].rolling(window=window_minutes, min_periods=1).mean()
        scores = df["scores"].values  # array

        rolling_anomalies = np.zeros_like(scores, dtype=bool)

        for i in range(window_size_minutes, len(scores)):
            start_idx = i - window_minutes
            end_idx = i
            window_slice = scores[start_idx:end_idx]
            current_score = scores[i]

            rank = percentileofscore(window_slice, current_score, kind="rank")
            if rank >= percentile:
                rolling_anomalies[i] = True
        print(rolling_anomalies.sum())
        capital_history, closed_trades = simulate_trading_std(
            model_name="kan",
            price_data=price_data,
            scaled_prices=scaled_prices,
            predicted_prices=predicted_prices,
            ma_data=ma_data,
            rsi_data=rsi_data,
            scores=scores,
            rolling_anomalies=rolling_anomalies,
            num_std=num_std,
            max_entries=max_entries,
            exit_val=exit_val,
            distr_len=distr_len,
            num_std_exit=num_std_exit,
            rsi_entry=rsi_entry,
            rsi_exit=rsi_exit,
        )

        capital_history = capital_history - 1

        if plot_pl:
            plt.figure(figsize=(12, 5))
            plt.plot(capital_history.index, capital_history, label="returns")
            plt.title(f"P&L Curve for {ticker} (exit_val={exit_val})")
            plt.xlabel("Date")
            plt.ylabel("Returns")
            plt.legend()
            plt.show()

        daily_rets = capital_history.resample("D").sum()

        results_dict[ticker] = daily_rets

    combined_returns = pd.concat(results_dict.values(), axis=1).sum(axis=1)
    return combined_returns


model_names = ["kan"]
model_params = {
    "autoencoder": {"input_size": sequence_length * 1, "hidden_dims": [64, 32, 16]},
    "lstm": {"input_size": 1, "hidden_size": 32, "num_layers": 1},
    "cnn": {"input_size": sequence_length, "out_channels": 16, "kernel_size": 3},
    "stackvaeg": {
        "input_size": sequence_length * 1,
        "hidden_dims": [64, 32],
        "latent_dim": 16,
    },
    "omnianomaly": {
        "input_size": 1,
        "hidden_size": 32,
        "latent_dim": 16,
        "num_layers": 1,
    },
    "egads": {"window": 30},
    "dtw": {"window": 30, "n_references": 10, "seed": 42},
    "cnnautoencoder": {
        "input_channels": 1,
        "sequence_length": sequence_length,
        "hidden_dims": [32, 16],
    },
    "kan": {
        "in_feat": 1,
        "hidden_feat": 64,
        "out_feat": 1,
        "num_layers": 2,
        "use_bias": True,
        "dropout": 0.3,
    },
}


train_initial_length = pd.DateOffset(months=6)
train_expand_step = pd.DateOffset(months=3)
test_length = pd.DateOffset(months=3)

start_date = pd.Timestamp("2020-01-01 00:00:00")
end_date = pd.Timestamp("2021-12-31 23:59:00")

combined_test_dates = []
combined_scores = {}
combined_predictions = {}
combined_anomalies = {}
original_prices_buffer = []
scaled_prices = []

for model_name in model_names:
    combined_scores[model_name] = []
    combined_anomalies[model_name] = []
    combined_predictions[model_name] = []

current_train_start = start_date
current_train_end = current_train_start + train_initial_length
current_test_start = current_train_end
current_test_end = current_test_start + test_length


window_idx = 0

while current_test_end <= end_date or current_test_start < end_date:
    if current_test_end > end_date:
        current_test_end = end_date
    print(
        f"\n=== Processing Window: Train {current_train_start} to {current_train_end}, "
        f"Test {current_test_start} to {current_test_end} ==="
    )

    train_data = data.loc[current_train_start:current_train_end]
    test_data = data.loc[current_test_start:current_test_end]

    if test_data.empty:
        break

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data[["close"]])
    test_scaled = scaler.transform(test_data[["close"]])

    original_prices_buffer.extend(test_data["close"].values[sequence_length:])
    scaled_prices.extend(test_scaled[sequence_length:, 0])

    train_torch = torch.tensor(train_scaled, dtype=torch.float32)
    test_torch = torch.tensor(test_scaled, dtype=torch.float32)

    train_sequences = create_sequences(train_torch, sequence_length)
    test_sequences = create_sequences(test_torch, sequence_length)
    test_dates = test_data.index[sequence_length:]
    combined_test_dates.extend(test_dates)

    for model_name in model_names:
        print(f"\n=== Processing Model: {model_name} ===")

        # params = model_params[model_name]
        # detector = AnomalyDetector(
        #     model_name=model_name,
        #     model_params=params,
        #     sequence_length=sequence_length,
        #     device=device
        # )

        # if model_name not in ['egads', 'dtw']:
        #     detector.train(
        #         train_sequences=train_sequences,
        #         num_epochs=10,
        #         batch_size=32,
        #         learning_rate=1e-3
        #     )

        # torch.save(
        #         detector.model.state_dict(),
        #         f"{model_name}_window_{window_idx}.pth"
        #     )

        # if model_name == 'egads':
        #     detector.train(
        #         train_sequences=None,
        #         train_raw=train_torch
        #     )
        # elif model_name == 'dtw':
        #     detector.train(
        #         train_sequences=train_sequences,
        #         num_epochs=0,
        #         batch_size=0,
        #         learning_rate=0,
        #         train_raw=None
        #     )

        # if model_name not in ['egads', 'dtw']:
        #     scores, preds = detector.predict(test_sequences=test_sequences)
        # elif model_name == 'egads':
        #     scores, _ = detector.predict(test_raw=test_torch)
        #     preds = np.zeros_like(scores)
        #     scores = scores[sequence_length:]
        #     preds = preds[sequence_length:]
        # elif model_name == 'dtw':
        #     scores, _ = detector.predict(test_sequences=test_sequences)
        #     preds = np.zeros_like(scores)

        # combined_scores[model_name].extend(scores)
        # combined_predictions[model_name].extend(preds)

    window_idx += 1

    current_train_end += train_expand_step
    current_test_start += train_expand_step
    current_test_end = current_test_start + test_length

for filename in os.listdir():
    if filename.endswith("_combined_scores_3m_20202021full.npy"):
        model_name = filename.replace("_combined_scores_3m_20202021full.npy", "")
        combined_scores[model_name] = np.load(filename, allow_pickle=True)
        print(f"Loaded combined_scores for {model_name} from {filename}")

for filename in os.listdir():
    if filename.endswith("_combined_preds_3m.npy"):
        model_name = filename.replace("_combined_preds_3m.npy", "")
        combined_predictions[model_name] = np.load(filename, allow_pickle=True)
        print(f"Loaded combined_predictions for {model_name} from {filename}")

dates = pd.to_datetime(combined_test_dates)
df = pd.DataFrame(index=dates)
df.index.name = "datetime"

df["close"] = original_prices_buffer
df["scaled_price"] = scaled_prices
df["scores"] = combined_scores["kan"]

df["predicted_price"] = combined_predictions["kan"]

data_dict = {"BTCUSDT": df}

params = {
    "num_std": [1, 2],
    "num_std_exit": [1, 2, 3],
    "percentile": [99, 99.5],
    "exit_val": ["rsi", "ma"],  # or "ma"
    "max_entries": [5, 10],
    "plot_pl": False,
    "distr_len": 99,
    "window_minutes": [50, 500],
    "window_size_minutes": [1000, 5000],
    "rsi_entry": [(30, 70), (40, 60)],
    "rsi_exit": [(70, 30), (60, 40), (50, 50)],
}


save_path = "results/"
file_prefix = f"anomaly"

optimizer = ParameterOptimizer(
    calc_pl, save_path=save_path, save_file_prefix=file_prefix, n_jobs=1
)

optimizer.split_data(data_dict, "2021-06-01")
optimizer.optimize(
    data_dict=data_dict,
    params=params,
    n_runs=64,
    best_trials_pct=0.1,
    n_splits=3,
    n_test_splits=1,
)
