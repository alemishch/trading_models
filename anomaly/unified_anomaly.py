import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import random
from datetime import timedelta
from pandas.tseries.offsets import DateOffset
from scipy.stats import percentileofscore, zscore
import ta
import os
from typing import List, Tuple, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator
from models import *


from backtest_optimizer.main import (
    ParameterOptimizer,
)


def mark_extremes_advanced(
    prices: List[float],
    threshold_abs: float = 0.0,
    threshold_pct: float = 0.03,
    use_pct: bool = True,
    dynamic_threshold: bool = False,
    vol_window: int = 10,
    vol_multiplier: float = 1.0,
    confirmation_period: int = 1,
) -> List[Tuple[int, str]]:
    """
    Identify turning points (peaks and troughs) in a price series using an advanced zigzag approach.

    In this version, a turning point is detected only after a reversal condition is confirmed
    for a number of consecutive bars (confirmation_period). In addition, the threshold for a reversal
    can be based on either a fixed percentage/absolute level or (optionally) be dynamically adjusted
    based on recent volatility.

    Parameters:
        prices (List[float]): List of price values.
        threshold_abs (float): Fixed absolute threshold (if use_pct is False).
        threshold_pct (float): Percentage threshold (e.g. 0.03 for 3%) used if use_pct is True.
        use_pct (bool): If True, the base threshold is computed as threshold_pct * current_price.
        dynamic_threshold (bool): If True, adjust the threshold based on recent volatility.
        vol_window (int): Number of bars to compute volatility if dynamic_threshold is True.
        vol_multiplier (float): Multiplier for volatility to form a dynamic threshold.
        confirmation_period (int): Number of consecutive bars that must confirm a reversal before marking it.

    Returns:
        List[Tuple[int, str]]: A list of tuples where each tuple is (index, type) and type is either 'peak' or 'trough'.
    """
    if not prices:
        return []

    turning_points: List[Tuple[int, str]] = []

    # Initialize extremes with the first price.
    last_min = prices[0]
    last_max = prices[0]
    last_min_idx = 0
    last_max_idx = 0

    # Trend indicator: True means an uptrend (so the last extreme was a minimum)
    # False means a downtrend (last extreme was a maximum). None means not yet established.
    trend = None

    i = 1
    n = len(prices)

    while i < n:
        price = prices[i]

        if dynamic_threshold:
            if i < vol_window:
                i += 1
                continue
            recent_changes = np.diff(prices[i - vol_window : i + 1])
            volatility = np.std(recent_changes)
            dynamic_thresh = vol_multiplier * volatility
            threshold = max(threshold_abs, dynamic_thresh)
        else:
            threshold = threshold_abs if not use_pct else threshold_pct * prices[i]
        # --- Update current extremes ---
        if price > last_max:
            last_max = price
            last_max_idx = i
        if price < last_min:
            last_min = price
            last_min_idx = i
        # --- Establish the initial trend if not set ---
        if trend is None:
            if last_max - prices[0] > threshold:
                trend = True  # Uptrend is established (price rose from the first bar)
            elif prices[0] - last_min > threshold:
                trend = False  # Downtrend is established
            # Else, trend remains undefined.
        # --- Check for reversal conditions with confirmation ---
        # In an uptrend, a sufficient drop from the last maximum signals a potential reversal.
        if trend is True and (last_max - price > threshold):
            confirm = True
            for j in range(i, min(i + confirmation_period, n)):
                if last_max - prices[j] <= threshold:
                    confirm = False
                    break
            if confirm:
                # In an uptrend, a drop signals that the previous maximum is a peak.
                turning_points.append((last_max_idx, "peak"))
                # Reset the minimum using the current price.
                last_min = prices[i]
                last_min_idx = i
                trend = False
                i += confirmation_period  # Skip confirmation period.
                continue

        # In a downtrend, a sufficient rise from the last minimum signals a potential reversal.
        if trend is False and (price - last_min > threshold):
            confirm = True
            for j in range(i, min(i + confirmation_period, n)):
                if prices[j] - last_min <= threshold:
                    confirm = False
                    break
            if confirm:
                turning_points.append((last_min_idx, "trough"))
                # Reset the maximum using the current price.
                last_max = prices[i]
                last_max_idx = i
                trend = True
                i += confirmation_period
                continue

        i += 1

    # Always mark the final point as a turning point if it wasn’t already included.
    if not turning_points or turning_points[-1][0] != n - 1:
        # For the final point, we decide its type based on the current trend.
        final_type = "trough" if trend else "peak"
        turning_points.append((n - 1, final_type))

    return turning_points


def expand_region(
    prices: List[float],
    extreme_index: int,
    extreme_type: str,
    region_tolerance_pct: float,
) -> Tuple[int, int]:
    """
    Given an extreme (turning point) at extreme_index with a given type ('peak' or 'trough'),
    expand outward to mark the region where the price remains close to the extreme value.

    For a 'peak', we include contiguous indices where the price is within (1 - region_tolerance_pct)
    of the peak value. For a 'trough', we include indices where the price is within (1 + region_tolerance_pct)
    of the trough value.

    Parameters:
        prices (List[float]): The price series.
        extreme_index (int): The index of the turning point.
        extreme_type (str): Either 'peak' or 'trough'.
        region_tolerance_pct (float): Tolerance as a fraction (e.g., 0.01 for 1%).

    Returns:
        Tuple[int, int]: A tuple (start_index, end_index) defining the region boundaries.
    """
    extreme_value = prices[extreme_index]
    start = extreme_index
    end = extreme_index
    n = len(prices)

    if extreme_type == "peak":
        # Expand left as long as the price is within region_tolerance_pct of the peak.
        while start > 0 and prices[start - 1] >= extreme_value * (
            1 - region_tolerance_pct
        ):
            start -= 1
        # Expand right similarly.
        while end < n - 1 and prices[end + 1] >= extreme_value * (
            1 - region_tolerance_pct
        ):
            end += 1
    elif extreme_type == "trough":
        while start > 0 and prices[start - 1] <= extreme_value * (
            1 + region_tolerance_pct
        ):
            start -= 1
        while end < n - 1 and prices[end + 1] <= extreme_value * (
            1 + region_tolerance_pct
        ):
            end += 1
    else:
        raise ValueError("extreme_type must be either 'peak' or 'trough'.")

    return start, end


def mark_extreme_regions(
    prices: List[float],
    threshold_abs: float = 0.0,
    threshold_pct: float = 0.03,
    use_pct: bool = True,
    dynamic_threshold: bool = False,
    vol_window: int = 10,
    vol_multiplier: float = 1.0,
    confirmation_period: int = 1,
    region_tolerance_pct: float = 0.01,
) -> List[Tuple[int, int, str]]:
    """
    First detect turning points using an advanced zigzag algorithm, then expand each turning point
    into a region where the price remains near the extreme value.

    Returns a list of tuples (region_start_index, region_end_index, type), where type is 'peak' or 'trough'.
    """
    # Detect turning points.
    turning_points = mark_extremes_advanced(
        prices,
        threshold_abs,
        threshold_pct,
        use_pct,
        dynamic_threshold,
        vol_window,
        vol_multiplier,
        confirmation_period,
    )
    regions = []
    for idx, t_type in turning_points:
        start, end = expand_region(prices, idx, t_type, region_tolerance_pct)
        regions.append((start, end, t_type))
    return regions


def detect_anomalies_agg_returns(
    df, L=10, rolling_window=50, threshold=3, method="sum"
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


def check_entry_signal(
    trade_type: str,
    current_rsi: float,
    current_price: float,
    current_ma_short: float,
    current_ma_long: float,
    rsi_entry: [list, tuple],
    exit_val: str,
    with_short: bool,
) -> bool:
    conditions = {
        ("buy", "rsi"): current_rsi < rsi_entry[0],
        ("buy", "ma"): current_price < current_ma_long,
        ("sell", "rsi"): (current_rsi > rsi_entry[1]) if with_short else False,
        ("sell", "ma"): (current_price > current_ma_short) if with_short else False,
    }
    return conditions.get((trade_type, exit_val), False)


class Trade(BaseModel):
    trade_type: Literal["buy", "sell"] = Field(..., description="Type of the trade")
    volume: float = Field(..., gt=0, description="Volume of the trade")
    entry_price: float = Field(
        ..., gt=0, description="Price at which the trade was entered"
    )
    entry_time: datetime = Field(
        ..., description="Timestamp when the trade was entered"
    )

    exit_time: Optional[datetime] = Field(
        None, description="Timestamp when the trade was closed"
    )
    exit_price: Optional[float] = Field(
        None, gt=0, description="Price at which the trade was closed"
    )
    profit: Optional[float] = Field(None, description="Profit or loss from the trade")

    @validator("entry_time", "exit_time", pre=True, always=True)
    def convert_np_datetime(cls, v):
        if isinstance(v, np.datetime64):
            # Convert numpy.datetime64 to Python datetime
            return pd.Timestamp(v).to_pydatetime()
        return v

    def close_trade(
        self, exit_time: datetime, exit_price: float, commission_rate: float
    ) -> None:
        """
        Closes the trade by setting the exit time, exit price, and calculating the profit.
        """
        self.exit_time = exit_time
        self.exit_price = exit_price

        if self.trade_type == "buy":
            raw_profit = (exit_price - self.entry_price) * (
                self.volume / self.entry_price
            )
        else:  # For "sell" trades
            raw_profit = (self.entry_price - exit_price) * (
                self.volume / self.entry_price
            )

        # Deduct commission from the profit
        self.profit = raw_profit - (self.volume * commission_rate)


# noinspection PyTypeChecker
def simulate_trading_std(
    model_name,
    price_data: pd.Series,
    scaled_prices: pd.Series,
    predicted_prices: pd.Series,
    ma_data_short: pd.Series,
    ma_data_long: pd.Series,
    rsi_data: pd.Series,
    scores,  # type unspecified
    rolling_anomalies,
    num_std: float,
    max_entries: int,
    exit_val,
    distr_len: int = 99,
    num_std_exit: float = 1,
    rsi_entry=(40, 60),
    rsi_exit=(60, 40),
    print_trades: bool = False,
    commission_rate: float = 0.0002,
    with_short: bool = True,
):
    """
    Simulates a trading strategy and returns a Pandas Series of daily profits and a list of closed Trade objects.
    """
    # Convert data to numpy arrays for fast access.
    price_array = price_data.to_numpy(dtype=np.float64)
    ma_array_short = ma_data_short.to_numpy(dtype=np.float64)
    ma_array_long = ma_data_long.to_numpy(dtype=np.float64)
    rsi_array = rsi_data.to_numpy(dtype=np.float64)
    scaled_prices_array = scaled_prices.to_numpy(dtype=np.float64)
    predicted_prices_array = predicted_prices.to_numpy(dtype=np.float64)
    anomalies_array = np.asarray(rolling_anomalies, dtype=bool)
    index_array = price_data.index.to_numpy()

    n = len(price_array)
    if len(rsi_array) != n or len(anomalies_array) != n:
        raise ValueError(
            "price_data, rsi_data, and rolling_anomalies must have the same length."
        )

    rolling_std = price_data.rolling(distr_len, min_periods=1).std().values

    capital = 1.0
    profit_array = np.full(n, np.nan, dtype=np.float64)
    open_trades: List[Trade] = []
    closed_trades: List[Trade] = []
    trade_entries = set()

    base_price = None
    base_trade_type = None

    def get_entry_signal() -> Optional[str]:
        if not is_anomaly:
            return None

        buy_signal = check_entry_signal(
            trade_type="buy",
            current_rsi=current_rsi,
            current_price=current_price,
            current_ma_short=current_ma_short,
            current_ma_long=current_ma_long,
            rsi_entry=rsi_entry,
            exit_val=exit_val,
            with_short=with_short,
        )

        sell_signal = (
            check_entry_signal(
                trade_type="sell",
                current_rsi=current_rsi,
                current_price=current_price,
                current_ma_short=current_ma_short,
                current_ma_long=current_ma_long,
                rsi_entry=rsi_entry,
                exit_val=exit_val,
                with_short=with_short,
            )
            if with_short
            else False
        )

        return "buy" if buy_signal else "sell" if sell_signal else None

    for i in range(n):
        timestamp = index_array[i]
        current_price = price_array[i]
        current_ma_short = ma_array_short[i]
        current_ma_long = ma_array_long[i]
        current_rsi = rsi_array[i]
        local_std = rolling_std[i]
        is_anomaly = anomalies_array[i]
        daily_profit = 0.0

        # ----- Handle Trade Exits -----
        if open_trades:
            # Check for a trade flip (opening an opposite position)
            opposite_trade_type = "sell" if base_trade_type == "buy" else "buy"

            flip_signal = get_entry_signal() == opposite_trade_type

            if flip_signal:
                for trade in open_trades:
                    trade.close_trade(
                        exit_time=timestamp,
                        exit_price=current_price,
                        commission_rate=commission_rate,
                    )
                    capital += trade.profit
                    daily_profit += trade.profit
                    if print_trades:
                        print(
                            f"Trade flipped: {trade.trade_type} entered at {trade.entry_time} "
                            f"(Entry Price: {trade.entry_price:.2f}), exited at {timestamp} "
                            f"(Exit Price: {current_price:.2f}, {current_ma_long=:1f}, {current_ma_short=:1f}), Volume: {trade.volume:.4f}, "
                            f"P&L: {trade.profit:.4f}"
                        )
                    closed_trades.append(trade)
                open_trades = []
                base_price, base_trade_type = None, None

            # Check individual trade exit conditions
            trades_to_close = []
            for trade in open_trades:
                exit_signal = False

                if exit_val == "rsi":
                    if trade.trade_type == "buy" and current_rsi >= rsi_exit[0]:
                        exit_signal = True
                    elif trade.trade_type == "sell" and current_rsi <= rsi_exit[1]:
                        exit_signal = True
                elif exit_val == "ma":
                    if trade.trade_type == "buy" and current_price >= current_ma_long:
                        exit_signal = True
                    elif (
                        trade.trade_type == "sell" and current_price <= current_ma_short
                    ):
                        exit_signal = True

                # Standard deviation-based exit condition
                if (
                    trade.trade_type == "buy"
                    and current_price >= trade.entry_price + local_std * num_std_exit
                ):
                    exit_signal = True
                if (
                    trade.trade_type == "sell"
                    and current_price <= trade.entry_price - local_std * num_std_exit
                ):
                    exit_signal = True

                if exit_signal:
                    trades_to_close.append(trade)

            for trade in trades_to_close:
                trade.close_trade(
                    exit_time=timestamp,
                    exit_price=current_price,
                    commission_rate=commission_rate,
                )
                capital += trade.profit
                daily_profit += trade.profit
                if print_trades:
                    print(
                        f"Trade executed: {trade.trade_type} entered at {trade.entry_time} "
                        f"(Entry Price: {trade.entry_price:.2f}), exited at {timestamp} "
                        f"(Exit Price: {current_price:.2f}), Volume: {trade.volume:.4f}, "
                        f"P&L: {trade.profit:.4f}"
                    )
                closed_trades.append(trade)
            # Keep open trades that haven't met exit conditions
            open_trades = [
                trade for trade in open_trades if trade not in trades_to_close
            ]

        # ----- Handle Trade Entries -----
        entry_signal = get_entry_signal()
        if entry_signal and len(open_trades) < max_entries:
            if not open_trades:
                if timestamp not in trade_entries:
                    trade_entries.add(timestamp)
                    trade_volume = capital / max_entries
                    new_trade = Trade(
                        trade_type=entry_signal,
                        volume=trade_volume,
                        entry_price=current_price,
                        entry_time=timestamp,
                    )
                    open_trades.append(new_trade)
                    base_price = current_price
                    base_trade_type = entry_signal
            else:
                required_distance = num_std * local_std
                price_condition = (
                    current_price >= base_price + required_distance
                    if base_trade_type == "sell"
                    else current_price <= base_price - required_distance
                )

                if price_condition and timestamp not in trade_entries:
                    trade_entries.add(timestamp)
                    trade_volume = capital / max_entries
                    new_trade = Trade(
                        trade_type=base_trade_type,
                        volume=trade_volume,
                        entry_price=current_price,
                        entry_time=timestamp,
                    )
                    open_trades.append(new_trade)
                    base_price = current_price

        profit_array[i] = daily_profit

    profits = pd.Series(profit_array, index=price_data.index).fillna(0)
    return profits, closed_trades


def simulate_trading_percentile(
    model_name,
    price_data,
    ma_data,
    scores,
    rolling_percentiles,
    rolling_anomalies,
    percentile,
    max_entries,
):
    initial_capital = 1.0
    capital = initial_capital
    capital_history = pd.Series(index=price_data.index, data=np.nan)
    capital_history.iloc[0] = capital

    open_trades = []
    trade_entries = set()  # Timestamps where trades are opened

    index_array = list(price_data.index)

    for i in range(len(index_array)):
        timestamp = index_array[i]

        current_price = price_data.iloc[i]
        current_ma = ma_data.iloc[i]

        trades_to_close = []
        for trade in open_trades:
            trade_type = trade["type"]
            entry_price = trade["entry_price"]
            entry_time = trade["entry_time"]
            anomaly_percentage = trade["anomaly_percentage"]
            if trade_type == "buy" and current_price >= current_ma:
                exit_price = current_price
                profit = (exit_price - entry_price) * (trade["volume"] / entry_price)
                capital += profit
                trades_to_close.append(trade)
                capital_history.iloc[i] = capital

                print(
                    f"Trade executed: {trade_type} at {entry_time} "
                    f"(Entry Price: {entry_price:.2f}), exited at {timestamp} "
                    f"(Exit Price: {exit_price:.2f}), Volume: {trade['volume']:.4f}, "
                    f"Anomaly: {anomaly_percentage:.2f}%, P&L: {profit:.4f}"
                )

            elif trade_type == "sell" and current_price <= current_ma:
                exit_price = current_price
                profit = (entry_price - exit_price) * (trade["volume"] / entry_price)
                capital += profit
                trades_to_close.append(trade)
                capital_history.iloc[i] = capital

                print(
                    f"Trade executed: {trade_type} at {entry_time} "
                    f"(Entry Price: {entry_price:.2f}), exited at {timestamp} "
                    f"(Exit Price: {exit_price:.2f}), Volume: {trade['volume']:.4f}, "
                    f"Anomaly: {anomaly_percentage:.2f}%, P&L: {profit:.4f}"
                )

        for closed_trade in trades_to_close:
            open_trades.remove(closed_trade)

        if rolling_anomalies[i] and len(open_trades) < max_entries:
            k = len(open_trades)
            required_threshold = percentile + k * step
            if required_threshold >= 100.0:
                required_threshold = 100.1

            anomaly_percentile = rolling_percentiles[i]
            if anomaly_percentile >= required_threshold:
                if timestamp not in trade_entries:
                    trade_entries.add(timestamp)
                    if current_price > current_ma:
                        trade_type = "sell"
                    elif current_price < current_ma:
                        trade_type = "buy"
                    else:
                        trade_type = None

                    if trade_type is not None:
                        trade_volume = capital / max_entries
                        anomaly_percentage = 100.0 - anomaly_percentile
                        open_trades.append(
                            {
                                "type": trade_type,
                                "volume": trade_volume,
                                "entry_price": current_price,
                                "entry_time": timestamp,
                                "anomaly_percentage": anomaly_percentage,
                            }
                        )

        capital_history.iloc[i] = capital

    capital_history.ffill(inplace=True)
    capital_history.fillna(capital, inplace=True)

    return capital_history


def calculate_rsi(df, window=14):
    rsi = ta.momentum.RSIIndicator(close=df["close"], window=window)
    rsi_values = rsi.rsi()
    rsi_values = np.maximum(rsi_values, 0)  # Ensure non-negative
    return rsi_values


def analyze_anomalies(
    anomaly_timestamps, original_data_sorted, window_minutes=13 * 24 * 60
):
    results = []

    for ts in anomaly_timestamps:
        if ts < original_data_sorted.index[window_minutes - 1]:
            continue

        future_time = ts + pd.Timedelta(hours=1)
        if future_time > original_data_sorted.index[-1]:
            continue

        try:
            open_price = original_data_sorted.loc[ts, "close"]
        except KeyError:
            continue

        prev_time = ts - pd.Timedelta(minutes=1)
        if prev_time not in original_data_sorted.index:
            continue
        rolling_avg = original_data_sorted.loc[prev_time, "rolling_avg"]
        rolling_low = original_data_sorted.loc[prev_time, "rolling_low"]
        rolling_high = original_data_sorted.loc[prev_time, "rolling_high"]

        if np.isnan(rolling_avg) or np.isnan(rolling_low) or np.isnan(rolling_high):
            continue

        above_avg = open_price > rolling_avg

        above_low = open_price > rolling_low
        below_high = open_price < rolling_high

        try:
            next_hour_data = original_data_sorted.loc[
                ts + pd.Timedelta(minutes=1) : ts + pd.Timedelta(hours=1)
            ]
        except KeyError:
            continue

        if next_hour_data.empty:
            continue

        min_open_next_hour = next_hour_data["close"].min()
        max_open_next_hour = next_hour_data["close"].max()

        try:
            close_price_next_hour = original_data_sorted.loc[
                ts + pd.Timedelta(hours=1), "close"
            ]
        except KeyError:
            continue

        if above_avg:
            price_after = min_open_next_hour
            price_change = min_open_next_hour - open_price
            direction = "Min"
        else:
            price_after = max_open_next_hour
            price_change = max_open_next_hour - open_price
            direction = "Max"

        close_change = close_price_next_hour - open_price

        price_change_pct = (price_change / open_price) * 100
        close_change_pct = (close_change / open_price) * 100

        if above_avg:
            if price_after < open_price:
                sign = 1
            elif price_after > open_price:
                sign = -1
            else:
                sign = 0
        else:
            if price_after > open_price:
                sign = 1
            elif price_after < open_price:
                sign = -1
            else:
                sign = 0

        price_change_pct_signed = price_change_pct * sign
        close_change_pct_signed = close_change_pct * sign

        relative_to_avg = "Above Avg" if above_avg else "Below Avg"
        relative_to_low_high = []
        if above_low:
            relative_to_low_high.append("Above Low")
        if below_high:
            relative_to_low_high.append("Below High")
        if not relative_to_low_high:
            relative_to_low_high.append("At Low/High")

        results.append(
            {
                "timestamp": ts,
                "open_price": open_price,
                "rolling_avg": rolling_avg,
                "rolling_low": rolling_low,
                "rolling_high": rolling_high,
                "relative_to_avg": relative_to_avg,
                "relative_to_low_high": ", ".join(relative_to_low_high),
                "price_after": price_after,
                "price_change_pct_signed": price_change_pct_signed,
                "direction": direction,
                "close_next_hour": close_price_next_hour,
                "close_change_pct_signed": close_change_pct_signed,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


def compute_statistics(results_df, model_name):
    if results_df.empty:
        print("No anomalies to analyze.")
        return

    print(f"\n=== Statistics for {model_name.capitalize()} ===")

    relative_to_avg = results_df["relative_to_avg"]
    print(f"\nRelative to {window_minutes}-minutes Average:")
    print(relative_to_avg.value_counts())

    relative_to_low_high = results_df["relative_to_low_high"]
    print(f"\nRelative to {window_minutes}-minutes Low/High:")
    print(relative_to_low_high.value_counts())

    print("\nNext Hour Min/Max Open Price Change (%):")
    print(results_df["price_change_pct_signed"].describe())

    print("\nNext Hour Close Price Change (%):")
    print(results_df["close_change_pct_signed"].describe())

    plt.figure(figsize=(10, 5))
    relative_to_avg.value_counts().plot(kind="bar")
    plt.title(
        f"{model_name.capitalize()} - Relative to {window_minutes}-minutes Average"
    )
    plt.xlabel("Position")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(10, 5))
    relative_to_low_high.value_counts().plot(kind="bar")
    plt.title(
        f"{model_name.capitalize()} - Relative to {window_minutes}-minutes Low/High"
    )
    plt.xlabel("Position")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(10, 5))
    results_df["price_change_pct_signed"].hist(bins=30)
    plt.title(
        f"{model_name.capitalize()} - Next Hour Min/Max Open Price Change (%) Distribution"
    )
    plt.xlabel("Price Change (%)")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(10, 5))
    results_df["close_change_pct_signed"].hist(bins=30)
    plt.title(
        f"{model_name.capitalize()} - Next Hour Close Price Change (%) Distribution"
    )
    plt.xlabel("Price Change (%)")
    plt.ylabel("Frequency")
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

file_path = (
    "/Users/alexanderdemachev/PycharmProjects/strategy/data/futures/1min/BTCUSDT.csv"
)
preds_path = "/Users/alexanderdemachev/PycharmProjects/strategy/aq/portfolio_optimization/market_regimes/trading_models/anomaly/results/"

file_path = "BTCUSDT.csv"
preds_path = "results/"


sequence_length = 100
start_date = pd.Timestamp("2020-01-01 00:00:00")
end_date = pd.Timestamp("2024-12-20 23:59:00")
# window_days = 55
window_minutes = 500  # for rolling mean
exit_val = "ma"
percentile = 99.5
max_entries = 5
step = (100 - percentile) / max_entries
mode = "std"  #'std'/'percentile'
num_std = 1
num_std_anomaly = 3

window_size_minutes = 1000  # window for final anomalies distribution


data = pd.read_csv(file_path)

data["datetime"] = pd.to_datetime(data["datetime"])
data.set_index("datetime", inplace=True)

original_data_sorted = data[["open", "close"]].copy().sort_index()

original_data_sorted["rolling_avg"] = (
    original_data_sorted["close"].rolling(window=window_minutes).mean()
)
original_data_sorted["rolling_low"] = (
    original_data_sorted["close"].rolling(window=window_minutes).min()
)
original_data_sorted["rolling_high"] = (
    original_data_sorted["close"].rolling(window=window_minutes).max()
)
original_data_sorted["rsi"] = calculate_rsi(original_data_sorted, window=window_minutes)

# model_names = ['autoencoder', 'cnnautoencoder', 'lstm', 'cnn', 'stackvaeg', 'omnianomaly', 'egads', 'kan'] ######
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

window_size = 10000

data["rolling_min"] = data["close"].rolling(window=window_size, min_periods=1).min()
data["rolling_max"] = data["close"].rolling(window=window_size, min_periods=1).max()

data["scaled_price"] = (data["close"] - data["rolling_min"]) / (
    data["rolling_max"] - data["rolling_min"]
)
data["scaled_price"].fillna(method="bfill", inplace=True)
data = data.drop(columns=["rolling_min", "rolling_max"])

combined_test_dates = []
combined_scores = {}
combined_predictions = {}
combined_anomalies = {}
original_prices_buffer = []
scaled_prices = []

# for model_name in model_names:
#     combined_scores[model_name] = []
#     combined_anomalies[model_name] = []
#     combined_predictions[model_name] = []

# current_train_start = start_date
# current_train_end = current_train_start + train_initial_length
# current_test_start = current_train_end
# current_test_end = current_test_start + test_length

# window_idx = 0

# while current_test_end <= end_date or current_test_start < end_date:
#     if current_test_end > end_date:
#         current_test_end = end_date
#     print(
#         f"\n=== Processing Window: Train {current_train_start} to {current_train_end}, "
#         f"Test {current_test_start} to {current_test_end} ==="
#     )

#     train_data = data.loc[current_train_start:current_train_end]
#     test_data = data.loc[current_test_start:current_test_end]

#     validation_start = current_train_end - pd.Timedelta(days=30)
#     validation_data = train_data.loc[validation_start:current_train_end]
#     train_data = train_data.loc[:validation_start]

#     if test_data.empty:
#         break

#     train_scaled = train_data["scaled_price"].to_numpy().reshape(-1, 1)
#     validation_scaled = validation_data["scaled_price"].to_numpy().reshape(-1, 1)
#     test_scaled = test_data["scaled_price"].to_numpy().reshape(-1, 1)

#     original_prices_buffer.extend(test_data["close"].values[sequence_length:])
#     scaled_prices.extend(test_scaled[sequence_length:, 0])

#     train_torch = torch.tensor(train_scaled, dtype=torch.float32)
#     validation_torch = torch.tensor(validation_scaled, dtype=torch.float32)
#     test_torch = torch.tensor(test_scaled, dtype=torch.float32)

#     train_sequences = create_sequences(train_torch, sequence_length)
#     validation_sequences = create_sequences(validation_torch, sequence_length)
#     test_sequences = create_sequences(test_torch, sequence_length)
#     test_dates = test_data.index[sequence_length:]
#     combined_test_dates.extend(test_dates)

#     for model_name in model_names:
#         print(f"\n=== Processing Model: {model_name} ===")

#         # params = model_params[model_name]
#         # detector = AnomalyDetector(
#         #     model_name=model_name,
#         #     model_params=params,
#         #     sequence_length=sequence_length,
#         #     device=device,
#         # )

#         # if model_name not in ['egads', 'dtw']:
#         #     detector.train(
#         #         train_sequences=train_sequences,
#         #         num_epochs=10,
#         #         batch_size=32,
#         #         learning_rate=1e-3,
#         #         validation_sequences=validation_sequences,
#         #         early_stopping_patience=5,
#         #         reduce_lr_patience=3,
#         #     )

#         # torch.save(
#         #         detector.model.state_dict(),
#         #         f"{model_name}_window_{window_idx}.pth"
#         #     )

#         # if model_name == 'egads':
#         #     detector.train(
#         #         train_sequences=None,
#         #         train_raw=train_torch
#         #     )
#         # elif model_name == 'dtw':
#         #     detector.train(
#         #         train_sequences=train_sequences,
#         #         num_epochs=0,
#         #         batch_size=0,
#         #         learning_rate=0,
#         #         train_raw=None
#         #     )

#         # if model_name not in ['egads', 'dtw']:
#         #     scores, preds = detector.predict(test_sequences=test_sequences)
#         # elif model_name == 'egads':
#         #     scores, _ = detector.predict(test_raw=test_torch)
#         #     preds = np.zeros_like(scores)
#         #     scores = scores[sequence_length:]
#         #     preds = preds[sequence_length:]
#         # elif model_name == 'dtw':
#         #     scores, _ = detector.predict(test_sequences=test_sequences)
#         #     preds = np.zeros_like(scores)

#         # combined_scores[model_name].extend(scores)
#         # combined_predictions[model_name].extend(preds)

#     window_idx += 1

#     current_train_end += train_expand_step
#     current_test_start += train_expand_step
#     current_test_end = current_test_start + test_length


# del combined_scores["autoencoder"]

# combined_scores["autoencoder"] = np.load(
#     "autoencoder_combined_scores_3m_20202021full.npy"
# )


def _test_outputs():
    for model_name in model_names:
        print(f"\n=== Final Outputs for Model: {model_name} ===")

        combined_scores[model_name] = np.array(combined_scores[model_name])
        scores = np.array(combined_scores[model_name])

        rolling_percentiles = np.zeros_like(scores, dtype=float)
        rolling_anomalies = np.zeros_like(scores, dtype=bool)
        # rolling_stds = np.zeros_like(scores, dtype=float)

        for i in range(window_size_minutes, len(scores)):
            start_idx = i - window_size_minutes
            end_idx = i
            window_slice = scores[start_idx:end_idx]

            current_score = scores[i]

            rank = percentileofscore(window_slice, current_score, kind="rank")
            # rolling_percentiles[i] = rank

            if rank >= percentile:
                rolling_anomalies[i] = True

            # deviation = deviation_from_mean(window_slice[-99:], current_score)
            # rolling_stds[i] = deviation

        anomaly_dates = np.array(combined_test_dates)[rolling_anomalies]
        print(
            f"Found {rolling_anomalies.sum()} anomalies using rolling percentile = {percentile}"
        )

        plt.figure(figsize=(15, 5))
        unscaled_prices = np.array(original_prices_buffer)
        plt.plot(combined_test_dates, unscaled_prices, label="Open Price", color="blue")

        rolling_avg_values = (
            original_data_sorted["rolling_avg"].reindex(combined_test_dates).values
        )
        plt.plot(
            combined_test_dates,
            rolling_avg_values,
            label="Rolling Avg",
            color="orange",
            linewidth=2,
        )

        anomaly_prices = unscaled_prices[rolling_anomalies]
        plt.scatter(anomaly_dates, anomaly_prices, color="red", label="Anomalies")
        plt.title(
            f"{model_name.capitalize()} Anomaly Detection (Rolling) for Entire Test Period"
        )
        plt.xlabel("Date")
        plt.ylabel("Open Price")
        plt.legend()
        plt.show()

        # analysis_results = analyze_anomalies(anomaly_dates, original_data_sorted, window_minutes=window_minutes)
        # compute_statistics(analysis_results, model_name)

        ma_window = window_minutes
        original_data_sorted["MA"] = (
            original_data_sorted["close"]
            .rolling(window=ma_window, min_periods=1)
            .mean()
        )

        combined_test_data = original_data_sorted.loc[combined_test_dates, "close"]
        combined_ma_data = original_data_sorted.loc[combined_test_dates, "MA"]

        if mode == "percentile":
            capital_history = simulate_trading_percentile(
                model_name=model_name,
                price_data=original_data_sorted.loc[combined_test_dates, "close"],
                ma_data=original_data_sorted.loc[combined_test_dates, "MA"],
                scores=scores,
                rolling_percentiles=rolling_percentiles,
                rolling_anomalies=rolling_anomalies,
                percentile=percentile,
                max_entries=max_entries,
            )
        elif mode == "std":
            capital_history, closed_trades = simulate_trading_std(
                model_name=model_name,
                price_data=original_data_sorted.loc[combined_test_dates, "close"],
                scaled_prices=scaled_prices,
                predicted_prices=combined_predictions[model_name],
                ma_data=original_data_sorted.loc[combined_test_dates, "MA"],
                rsi_data=original_data_sorted.loc[combined_test_dates, "rsi"],
                scores=scores,
                rolling_anomalies=rolling_anomalies,
                num_std=num_std,
                max_entries=max_entries,
                exit_val=exit_val,
                distr_len=99,
                num_std_exit=1,
            )

        plt.figure(figsize=(15, 5))
        plt.plot(capital_history.index, capital_history.values - 1, label="Capital")
        plt.title(
            f"P&L Curve for {model_name.capitalize()} (Rolling Window Simulation)"
        )
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.legend()
        plt.show()

        final_capital = capital_history.iloc[-1]
        print(f"Final Capital for {model_name}: {final_capital:.4f}")


def calc_pl(data_dict, params):
    results_dict = {}

    num_std = params.get("num_std", 1)
    num_std_exit = params.get("num_std_exit", 1)
    percentile = params.get("percentile", 99.5)
    exit_val = params.get("exit_val", "rsi")
    plot_pl = params.get("plot_pl", False)
    distr_len = params.get("distr_len", 99)
    ma_window_short = params.get("ma_window_short", 100)
    ma_window_long = params.get("ma_window_long", 200)
    RSI_window_minutes = params.get("RSI_window_minutes", 500)
    window_size_minutes = params["window_size_minutes"]
    multiplier = params["multiplier"]
    rsi_entry = (
        int(params["rsi_entry"].split(",")[0]),
        int(params["rsi_entry"].split(",")[1]),
    )
    rsi_exit = (
        int(params["rsi_exit"].split(",")[0]),
        int(params["rsi_exit"].split(",")[1]),
    )
    max_entries = params.get("max_entries", 5)
    print_trades = params.get("print_trades", False)
    comission_rate = params.get("comission_rate", 0.0002)
    with_short = params.get("with_short", True)
    ignore_anomalies = params["ignore_anomalies"]

    for ticker, df in data_dict.items():
        if print_trades:
            print(f"\n=== Processing Ticker: {ticker} ===")
        price_data = df["close"]
        scaled_prices = df["scaled_price"]
        predicted_prices = df["predicted_price"]
        rsi_data = calculate_rsi(df, window=RSI_window_minutes)
        ma_data_short = (
            df["close"].rolling(window=ma_window_short, min_periods=1).mean()
        )
        ma_data_long = df["close"].rolling(window=ma_window_long, min_periods=1).mean()
        scores = df["scores"].values  # array

        # rolling_anomalies = np.zeros_like(scores, dtype=bool)

        # for i in range(window_size_minutes, len(scores)):
        #     start_idx = i - window_size_minutes
        #     end_idx = i
        #     window_slice = scores[start_idx:end_idx]
        #     current_score = scores[i]

        #     rank = percentileofscore(window_slice, current_score, kind="rank")
        #     if rank >= percentile:
        #         rolling_anomalies[i] = True

        if not ignore_anomalies:
            rolling_anomalies = detect_multi_scale_anomalies(
                scores=scores,
                window_sizes=[
                    window_size_minutes // 2,
                    window_size_minutes,
                    window_size_minutes * 2,
                    window_size_minutes * 3,
                ],
                multiplier=multiplier,
            )
        else:
            rolling_anomalies = np.ones(len(scores), dtype=bool)

        if print_trades:
            print(rolling_anomalies.sum())
        profits, closed_trades = simulate_trading_std(
            model_name="autoencoder",
            price_data=price_data,
            scaled_prices=scaled_prices,
            predicted_prices=predicted_prices,
            ma_data_short=ma_data_short,
            ma_data_long=ma_data_long,
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
            print_trades=print_trades,
            commission_rate=comission_rate,
            with_short=with_short,
        )

        if plot_pl:
            plt.figure(figsize=(12, 5))
            plt.plot(profits.index, profits.cumsum(), label="Profits history")
            plt.title(f"P&L Curve for {ticker} (exit_val={exit_val})")
            plt.xlabel("Date")
            plt.ylabel("Returns")
            plt.legend()
            plt.show()

        daily_rets = profits.resample("D").sum()

        results_dict[ticker] = daily_rets

    combined_returns = pd.concat(results_dict.values(), axis=1).sum(axis=1)

    return combined_returns


def calculate_sharpe_ratio(profits, risk_free_rate=0.0, annualization_factor=365):
    # Convert profits to daily returns
    daily_returns = profits.resample("D").sum()

    # Calculate mean and standard deviation of daily returns
    excess_returns = daily_returns - risk_free_rate  # Subtract risk-free rate if needed
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()

    # Avoid division by zero in case of zero volatility
    if std_return == 0:
        return np.nan

    # Compute Sharpe Ratio (annualized)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(annualization_factor)

    return sharpe_ratio


def detect_multi_scale_anomalies(
    scores: np.ndarray, window_sizes: list = [500, 1000, 5000], multiplier: float = 3.0
):
    anomalies = np.zeros_like(scores, dtype=bool)
    for window_size in window_sizes:
        rolling_mean = (
            pd.Series(scores)
            .rolling(window=window_size, min_periods=1)
            .mean()
            .to_numpy()
        )
        rolling_std = (
            pd.Series(scores)
            .rolling(window=window_size, min_periods=1)
            .std()
            .to_numpy()
        )
        threshold = rolling_mean + multiplier * rolling_std
        anomalies |= scores > threshold
    return anomalies


def main():
    arrs = np.load("results/data_2024.npz", allow_pickle=True)

    combined_test_dates = arrs["combined_test_dates"]
    scaled_prices = arrs["scaled_prices"]
    original_prices_buffer = arrs["original_prices_buffer"]

    ignore_anomalies = False

    dates = pd.to_datetime(combined_test_dates)
    df = pd.DataFrame(index=dates)
    df.index.name = "datetime"

    if not ignore_anomalies:
        for filename in os.listdir(preds_path):
            if filename.endswith("_combined_scores_3m_20202024full.npy"):
                model_name = filename.replace(
                    "_combined_scores_3m_20202024full.npy", ""
                )
                combined_scores[model_name] = np.load(
                    preds_path + filename, allow_pickle=True
                )
                print(f"Loaded combined_scores for {model_name} from {filename}")

        df["scores"] = np.minimum(
            combined_scores["kan"], combined_scores["autoencoder"]
        )
    else:
        df["scores"] = np.zeros_like(combined_test_dates)

    for filename in os.listdir(preds_path):
        if filename.endswith("_combined_preds_3m_20202024.npy"):
            model_name = filename.replace("_combined_preds_3m_20202024.npy", "")
            combined_predictions[model_name] = np.load(
                preds_path + filename, allow_pickle=True
            )
            print(f"Loaded combined_predictions for {model_name} from {filename}")

    model_name = "kan"

    df["close"] = original_prices_buffer
    df["scaled_price"] = scaled_prices

    df["predicted_price"] = combined_predictions[model_name]

    data_dict = {"BTCUSDT": df}

    best_params = {
        "num_std": 3,
        "num_std_exit": 3,
        "percentile": 5,
        "exit_val": "rsi",
        "max_entries": 5,
        "distr_len": 144,
        "ma_window_short": 34,
        "ma_window_long": 200,
        "window_size_minutes": 100,
        "RSI_window_minutes": 55,
        "rsi_entry": "40,60",
        "rsi_exit": "50,50",
        "plot_pl": True,
        "print_trades": True,
        "comission_rate": 0.0004,
        "with_short": False,
        "multiplier": 2.0,
        "ignore_anomalies": ignore_anomalies,
    }

    # start_date = "2021-06-01"
    # end_date = "2021-12-31"

    # data_dict["BTCUSDT"] = data_dict["BTCUSDT"].loc[:end_date]

    # results = calc_pl(data_dict, best_params)

    params = {
        "num_std": [1, 2, 3],
        "num_std_exit": [1, 2, 3],
        "percentile": [75, 90, 95, 99],
        "exit_val": ["rsi", "ma"],  # or "ma"
        "max_entries": [5, 10, 20],
        "plot_pl": False,
        "distr_len": [34, 144],
        "RSI_window_minutes": [55, 89],
        "Ma_window_short": [13, 34, 89],
        "Ma_window_long": [55, 100, 200, 500],
        "window_size_minutes": [100, 1000, 5000],
        "rsi_entry": ["30,70", "40,60"],
        "rsi_exit": ["30,70", "40,60", "50,50"],
        "print_trades": False,
        "comission_rate": 0.0004,
        "with_short": [True, False],
        "multiplier": [1.8, 2.0, 2.2],
        "ignore_anomalies": [True, False],
    }

    save_path = "/Users/alexanderdemachev/PycharmProjects/strategy/aq/portfolio_optimization/market_regimes/trading_models/anomaly/results/"
    save_path = "results/"
    file_prefix = f"anomaly_"

    optimizer = ParameterOptimizer(
        calc_pl, save_path=save_path, save_file_prefix=file_prefix, n_jobs=10
    )

    # optimizer.split_data(data_dict, "2024-01-01")
    # optimizer.optimize(
    #     data_dict=data_dict,
    #     params=params,
    #     n_runs=96,
    #     best_trials_pct=0.1,
    #     n_splits=3,
    #     n_test_splits=1,
    # )
    # optimizer.read_saved_params()
    # best_params = optimizer.cluster_and_aggregate(1)
    # data_dict = optimizer.load_data_from_parquet("train")
    # optimizer.plot_returns(data_dict, best_params)
    # data_dict = optimizer.load_data_from_parquet("test")
    optimizer.plot_returns(data_dict, best_params)


if __name__ == "__main__":
    main()
