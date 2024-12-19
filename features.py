import os
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from hurst import compute_Hc
import ta
from scipy import stats
from scipy.stats import linregress


def load_data(ticker, data_path):
    try:
        df = pd.read_csv(
            os.path.join(data_path, f"{ticker}.csv"),
            index_col=0,
            parse_dates=[0],
            low_memory=False,
        )
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None


### Volatility Features


def calculate_supertrend(df, period=10, multiplier=3):
    # SuperTrend from ta library
    try:
        st = ta.trend.Supertrend(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            period=period,
            multiplier=multiplier,
        )
        supertrend = st.supertrend()
        supertrend_direction = st.supertrend_direction().astype(float)
        supertrend_direction = supertrend_direction.replace({True: 1.0, False: -1.0})
        return supertrend, supertrend_direction
    except Exception:
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)


def calculate_ichimoku(df, conversion_window=9, base_window=26, lagging_window=52):
    # Ichimoku Indicator
    try:
        ichi = ta.trend.IchimokuIndicator(
            high=df["high"],
            low=df["low"],
            window1=conversion_window,
            window2=base_window,
            window3=lagging_window,
        )
        # Ichimoku lines
        conversion_line = ichi.ichimoku_conversion_line()
        base_line = ichi.ichimoku_base_line()
        chikou_span = df["close"].shift(-base_window)

        leading_span_a = ((conversion_line + base_line) / 2).shift(base_window)
        highest_high = df["high"].rolling(lagging_window).max()
        lowest_low = df["low"].rolling(lagging_window).min()
        leading_span_b = ((highest_high + lowest_low) / 2).shift(base_window)

        return conversion_line, base_line, chikou_span, leading_span_a, leading_span_b
    except Exception:
        return (
            pd.Series(np.nan, index=df.index),
            pd.Series(np.nan, index=df.index),
            pd.Series(np.nan, index=df.index),
            pd.Series(np.nan, index=df.index),
            pd.Series(np.nan, index=df.index),
        )


def calculate_zigzag(df, pct_threshold=0.05):
    prices = df["close"]
    zigzag = pd.Series(np.nan, index=prices.index)
    direction = 0  # 1 up, -1 down
    last_extreme = prices.iloc[0]

    for i in range(1, len(prices)):
        price = prices.iloc[i]
        change = (price - last_extreme) / (last_extreme + 1e-10)
        if direction >= 0 and change > pct_threshold:
            direction = 1
            last_extreme = price
            zigzag.iloc[i] = price
        elif direction <= 0 and change < -pct_threshold:
            direction = -1
            last_extreme = price
            zigzag.iloc[i] = price

    return zigzag


def calculate_linear_regression_slope(series):
    if series.dropna().shape[0] < 2:
        return np.nan
    x = np.arange(len(series))
    y = series.values
    # Handle constant series or NaNs
    if np.all(np.isnan(y)):
        return np.nan
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(y) < 2:
        return np.nan
    slope, _, _, _, _ = linregress(x, y)
    return slope


def calculate_donchian_channel_width(df, window=20):
    try:
        dc = ta.volatility.DonchianChannel(
            high=df["high"], low=df["low"], close=df["close"], window=window
        )
        upper = dc.donchian_channel_hband()
        lower = dc.donchian_channel_lband()
        width = upper - lower
        return width
    except Exception:
        return pd.Series(np.nan, index=df.index)


def process_additional_trend_features(df_daily, features_df, window):
    # Supertrend
    st, st_dir = calculate_supertrend(df_daily, period=window, multiplier=3)
    features_df[f"Supertrend_{window}"] = st
    features_df[f"Supertrend_Direction_{window}"] = st_dir

    # Ichimoku
    conv, base, chikou, span_a, span_b = calculate_ichimoku(
        df_daily,
        conversion_window=min(9, window),
        base_window=min(26, window),
        lagging_window=min(52, 2 * window),
    )
    features_df[f"Ichimoku_Conversion_{window}"] = conv
    features_df[f"Ichimoku_Base_{window}"] = base
    features_df[f"Ichimoku_Chikou_{window}"] = chikou
    features_df[f"Ichimoku_SpanA_{window}"] = span_a
    features_df[f"Ichimoku_SpanB_{window}"] = span_b

    # ZigZag
    zigzag_line = calculate_zigzag(df_daily)
    features_df[f"ZigZag_{window}"] = zigzag_line

    # Linear Regression Slope of Close prices
    features_df[f"Linear_Regression_Slope_Close_{window}"] = (
        df_daily["close"]
        .rolling(window)
        .apply(calculate_linear_regression_slope, raw=False)
    )

    # Donchian Channel Width
    dc_width = calculate_donchian_channel_width(df_daily, window=window)
    features_df[f"Donchian_Channel_Width_{window}"] = dc_width


def process_additional_volatility_features(features_df, window):
    # Apply skewness and kurtosis on Historical Volatility from existing code
    hist_vol_col = f"Historical_Volatility_{window}"
    if hist_vol_col in features_df.columns:
        vol = features_df[hist_vol_col]
        features_df[f"Volatility_Skewness_{window}"] = calculate_volatility_skewness(
            vol, window=window
        )
        features_df[f"Volatility_Kurtosis_{window}"] = calculate_volatility_kurtosis(
            vol, window=window
        )


def process_additional_returns_features(returns, features_df, window):
    # Slope of cumulative returns
    if "Cumulative_Returns" in features_df.columns:
        features_df[
            f"Slope_of_CumReturns_{window}"
        ] = calculate_slope_of_cumulative_returns(
            features_df["Cumulative_Returns"], window=window
        )
    else:
        cum_ret = (1 + returns).cumprod() - 1
        features_df["Cumulative_Returns"] = cum_ret
        features_df[
            f"Slope_of_CumReturns_{window}"
        ] = calculate_slope_of_cumulative_returns(cum_ret, window=window)

    # Variation of Returns (variance)
    features_df[f"Variation_of_Returns_{window}"] = calculate_variation_of_returns(
        returns, window=window
    )


# Volatility Features
# Apply skewness and kurtosis on Historical Volatility
def calculate_volatility_skewness(volatility, window=30):
    return volatility.rolling(window).apply(
        lambda x: x.skew() if len(x.dropna()) > 1 else np.nan
    )


def calculate_volatility_kurtosis(volatility, window=30):
    return volatility.rolling(window).apply(
        lambda x: x.kurt() if len(x.dropna()) > 1 else np.nan
    )


# Returns Features
def calculate_slope_of_cumulative_returns(cum_returns, window=30):
    return cum_returns.rolling(window).apply(
        calculate_linear_regression_slope, raw=False
    )


def calculate_variation_of_returns(returns, window=30):
    # variation as variance
    return returns.rolling(window).var()


def calculate_log_returns(df):
    return np.log(df["close"] / df["close"].shift(1)).dropna()


def calculate_atr(df, window=14):
    atr = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=window
    )
    atr_values = atr.average_true_range()
    atr_values = np.maximum(atr_values, 0)  # Ensure non-negative
    return atr_values


def calculate_bollinger_bands(df, window=20, window_dev=2):
    bollinger = ta.volatility.BollingerBands(
        close=df["close"], window=window, window_dev=window_dev
    )
    bb_df = pd.DataFrame(
        {
            "bollinger_upper": bollinger.bollinger_hband(),
            "bollinger_middle": bollinger.bollinger_mavg(),
            "bollinger_lower": bollinger.bollinger_lband(),
        },
        index=df.index,
    )
    return bb_df


def calculate_ewma_volatility(returns, span=30, trading_days=252):
    ewma_variance = returns.ewm(span=span, adjust=False).mean()
    ewma_volatility = np.sqrt(np.maximum(ewma_variance, 0)) * np.sqrt(trading_days)
    return ewma_volatility


def calculate_historical_volatility(returns, window=30, trading_days=252):
    hv = returns.rolling(window=window).std() * np.sqrt(trading_days)
    hv = np.maximum(hv, 0)
    return hv


def calculate_realized_volatility(df, window=30, trading_days=252):
    intraday_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    daily_rv = intraday_returns.groupby(intraday_returns.index.date).apply(
        lambda x: np.sqrt(np.sum(x**2))
    )
    daily_rv.index = pd.to_datetime(daily_rv.index)
    realized_volatility = daily_rv.rolling(window=window).mean() * np.sqrt(trading_days)
    realized_volatility = np.maximum(realized_volatility, 0)
    return realized_volatility


def calculate_parkinson_volatility(df, window=30, trading_days=252):
    log_hl = np.log(df["high"] / df["low"])
    parkinson_variance = (1 / (4 * math.log(2))) * (log_hl**2)
    rolling_parkinson_var = parkinson_variance.rolling(window).sum() / window
    parkinson_volatility = np.sqrt(np.maximum(rolling_parkinson_var, 0)) * np.sqrt(
        trading_days
    )
    return parkinson_volatility


def calculate_garman_klass_volatility(df, window=30, trading_days=252):
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    gk_var = 0.5 * (log_hl**2) - (2 * math.log(2) - 1) * (log_co**2)
    rolling_gk_var = gk_var.rolling(window).sum() / window
    gk_volatility = np.sqrt(np.maximum(rolling_gk_var, 0)) * np.sqrt(trading_days)
    return gk_volatility


def calculate_max_drawdown_duration(cum_returns):
    max_duration = pd.Series(index=cum_returns.index, dtype="float64")
    current_duration = 0
    max_duration_val = 0
    peak = cum_returns.iloc[0]

    for i in range(1, len(cum_returns)):
        if cum_returns.iloc[i] < peak:
            current_duration += 1
            max_duration_val = max(max_duration_val, current_duration)
        else:
            peak = cum_returns.iloc[i]
            current_duration = 0
        max_duration.iloc[i] = max_duration_val
    return max_duration


def calculate_volatility_of_volatility(volatility, window=30):
    vol_of_vol = volatility.rolling(window).std()
    return vol_of_vol


### Trend Features


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


def calculate_rsi(df, window=14):
    rsi = ta.momentum.RSIIndicator(close=df["close"], window=window)
    rsi_values = rsi.rsi()
    rsi_values = np.maximum(rsi_values, 0)  # Ensure non-negative
    return rsi_values


def calculate_adx(df, window=14):
    adx = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=window
    )
    adx_values = adx.adx()
    adx_values = np.maximum(adx_values, 0)  # Ensure non-negative
    return adx_values


def calculate_momentum(df, window=10):
    roc = ta.momentum.ROCIndicator(close=df["close"], window=window)
    momentum = roc.roc()
    momentum = np.maximum(momentum, 0)
    return momentum


def compute_hurst_exponent(prices):
    if len(prices) < 20:
        return np.nan
    try:
        H, c, data = compute_Hc(prices, kind="price", simplified=True)
        return H
    except Exception:
        return np.nan


def compute_ou_theta(prices):
    if len(prices) < 2:
        return np.nan
    try:
        log_prices = np.log(prices)
        delta_log_prices = np.diff(log_prices)
        log_prices_lagged = log_prices[:-1]
        X = sm.add_constant(log_prices_lagged)
        y = delta_log_prices
        model = sm.OLS(y, X).fit()
        if len(model.params) > 1:
            phi = model.params.iloc[1]
            theta = -phi
            return theta
        else:
            return np.nan
    except Exception:
        return np.nan


### Returns Features


def calculate_acf(returns, lags=10):
    acf_values = [returns.autocorr(lag=i) for i in range(1, lags + 1)]
    return acf_values


def calculate_mean_returns(returns, window=10):
    return returns.rolling(window=window).mean()


def calculate_sharpe_ratio(returns, window=30, risk_free_rate=0.0, trading_days=252):
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    sharpe = (
        (rolling_mean - (risk_free_rate / trading_days))
        / rolling_std
        * np.sqrt(trading_days)
    )
    sharpe = pd.Series(sharpe, index=returns.index)
    sharpe = sharpe.where(
        rolling_std != 0, np.nan
    )  # Handle division by zero using pandas
    return sharpe


def calculate_sortino_ratio(returns, window=30, risk_free_rate=0.0, trading_days=252):
    rolling_mean = returns.rolling(window).mean()
    rolling_std_down = returns.where(returns < 0).rolling(window).std()
    sortino = (
        (rolling_mean - (risk_free_rate / trading_days))
        / rolling_std_down
        * np.sqrt(trading_days)
    )
    sortino = sortino.where(
        rolling_std_down != 0, np.nan
    )  # Use Pandas' where for alignment
    return sortino


def calculate_calmar_ratio(returns, window=30, trading_days=252):
    rolling_return = (1 + returns).rolling(window).apply(
        lambda x: x.prod(), raw=True
    ) - 1
    rolling_drawdown = calculate_max_drawdown_duration(rolling_return)
    calmar = rolling_return / np.abs(rolling_drawdown)
    calmar = pd.Series(calmar, index=returns.index)
    calmar = calmar.where(rolling_drawdown != 0, np.nan)
    return calmar


def calculate_omega_ratio(returns, threshold=0.0, window=30):
    omega = returns.rolling(window).apply(
        lambda x: (x[x > threshold].sum()) / (np.abs(x[x < threshold].sum()) + 1e-10)
        if len(x[x < threshold]) > 0
        else np.nan,
        raw=False,
    )
    return omega


def calculate_jarque_bera(returns, window=30):
    jb_stat = returns.rolling(window).apply(
        lambda x: stats.jarque_bera(x)[0], raw=False
    )
    jb_pvalue = returns.rolling(window).apply(
        lambda x: stats.jarque_bera(x)[1], raw=False
    )
    return pd.DataFrame(
        {"Jarque-Bera_statistic": jb_stat, "Jarque-Bera_pvalue": jb_pvalue}
    )


def calculate_skewness_kurtosis(returns, window=30):
    skew = returns.rolling(window).skew()
    kurtosis = returns.rolling(window).kurt()
    skew = pd.Series(skew, index=returns.index)
    kurtosis = pd.Series(kurtosis, index=returns.index)
    skew = skew.where(skew >= -5, np.nan)  # Handle extreme skewness
    kurtosis = kurtosis.where(kurtosis >= -10, np.nan)  # Handle extreme kurtosis
    return pd.DataFrame({"skewness": skew, "kurtosis": kurtosis})


def calculate_cumulative_returns(returns):
    return (1 + returns).cumprod() - 1


def compute_hurst_returns(returns):
    if len(returns) < 20:
        return np.nan
    try:
        H, c, data = compute_Hc(returns, kind="price", simplified=True)
        return H
    except Exception:
        return np.nan


### Features by Category


def process_volatility_features(df_daily, volatility_df, window, trading_days=365):
    volatility_df[f"Historical_Volatility_{window}"] = calculate_historical_volatility(
        df_daily["Log_Returns"], window=window, trading_days=trading_days
    )

    volatility_df[f"Realized_Volatility_{window}"] = calculate_realized_volatility(
        df_daily, window=window, trading_days=trading_days
    )

    volatility_df[f"EWMA_Volatility_{window}"] = calculate_ewma_volatility(
        df_daily["Log_Returns"], span=window, trading_days=trading_days
    )

    volatility_df[f"Parkinson_Volatility_{window}"] = calculate_parkinson_volatility(
        df_daily, window=window, trading_days=trading_days
    )

    volatility_df[
        f"Garman_Klass_Volatility_{window}"
    ] = calculate_garman_klass_volatility(
        df_daily, window=window, trading_days=trading_days
    )

    volatility_df[
        f"Volatility_of_Volatility_{window}"
    ] = calculate_volatility_of_volatility(
        volatility_df[f"Historical_Volatility_{window}"], window=window
    )

    volatility_df[f"Max_Drawdown_Duration_{window}"] = (
        calculate_max_drawdown_duration(df_daily["Cumulative_Returns"])
        .rolling(window=window)
        .max()
    )


def process_trend_features(df_daily, features_df, window):
    macd_df = calculate_macd(df_daily)
    features_df[f"MACD_{window}"] = macd_df["macd"]
    features_df[f"MACD_Signal_{window}"] = macd_df["macd_signal"]
    features_df[f"MACD_Diff_{window}"] = macd_df["macd_diff"]

    features_df[f"RSI_{window}"] = calculate_rsi(df_daily, window=window)
    features_df[f"ADX_{window}"] = calculate_adx(df_daily, window=window)
    features_df[f"Momentum_{window}"] = calculate_momentum(df_daily, window=window)

    features_df[f"Hurst_{window}"] = (
        df_daily["close"].rolling(window).apply(compute_hurst_exponent, raw=False)
    )
    features_df[f"OU_Theta_{window}"] = (
        df_daily["close"].rolling(window).apply(compute_ou_theta, raw=False)
    )


def process_returns_features(returns, features_df, window):
    features_df[f"Mean_Returns_{window}"] = calculate_mean_returns(
        returns, window=window
    )
    features_df[f"Sharpe_Ratio_{window}"] = calculate_sharpe_ratio(
        returns, window=window, risk_free_rate=0.0
    )
    features_df[f"Sortino_Ratio_{window}"] = calculate_sortino_ratio(
        returns, window=window, risk_free_rate=0.0
    )
    features_df[f"Calmar_Ratio_{window}"] = calculate_calmar_ratio(
        returns, window=window
    )

    skew_kurt = calculate_skewness_kurtosis(returns, window=window)
    features_df[f"Skewness_{window}"] = skew_kurt["skewness"]
    features_df[f"Kurtosis_{window}"] = skew_kurt["kurtosis"]

    features_df[f"Hurst_Returns_{window}"] = returns.rolling(window).apply(
        compute_hurst_returns, raw=False
    )

    for lag in range(1, 11):
        features_df[f"ACF_Lag_{lag}_{window}"] = returns.rolling(window).apply(
            lambda x: x.autocorr(lag=lag), raw=False
        )

    jb = calculate_jarque_bera(returns, window=window)
    features_df["Jarque-Bera_statistic"] = jb["Jarque-Bera_statistic"]
    features_df["Jarque-Bera_pvalue"] = jb["Jarque-Bera_pvalue"]

    features_df[f"VaR_95_{window}"] = returns.rolling(window).quantile(0.05)
    features_df[f"CVaR_95_{window}"] = returns.rolling(window).apply(
        lambda x: x[x <= x.quantile(0.05)].mean()
        if len(x[x <= x.quantile(0.05)]) > 0
        else np.nan,
        raw=False,
    )

    features_df[f"Omega_Ratio_{window}"] = calculate_omega_ratio(
        returns, threshold=0.0, window=window
    )


def process_volatility_data(ticker, data_path, windows, trading_days=365):
    df = load_data(ticker, data_path)
    if df is None or df.empty:
        print(f"No data for {ticker}.")
        return

    df_daily = (
        df.resample("D")
        .agg(
            {
                "close": "last",
                "high": "max",
                "low": "min",
                "open": "first",
                "volume": "sum",
            }
        )
        .dropna()
    )

    if df_daily.empty:
        print(f"No daily data for {ticker}.")
        return

    df_daily["Log_Returns"] = calculate_log_returns(df_daily)
    df_daily["Cumulative_Returns"] = calculate_cumulative_returns(
        df_daily["Log_Returns"]
    )

    volatility_df = pd.DataFrame(index=df_daily.index)

    for window in windows:
        process_volatility_features(df_daily, volatility_df, window, trading_days)

    features_df = volatility_df.copy()

    for window in windows:
        process_trend_features(df_daily, features_df, window)
        process_additional_trend_features(df_daily, features_df, window)
        process_additional_volatility_features(features_df, window)

    features_df["Volatility_of_Volatility_30"] = calculate_volatility_of_volatility(
        features_df["Historical_Volatility_30"], window=30
    )

    features_df["Max_Drawdown_Duration_30"] = (
        calculate_max_drawdown_duration(df_daily["Cumulative_Returns"])
        .rolling(window=30)
        .max()
    )

    all_features = features_df.copy()

    # numeric_columns = all_features.select_dtypes(include=[np.number]).columns
    # all_features[numeric_columns] = all_features[numeric_columns].applymap(lambda x: x if x >=0 else np.nan)

    all_features.to_csv(f"all_features_{ticker}.csv")
    return all_features


def process_rets_data(rets_path, output_dir, windows, trading_days=365):
    try:
        rets_df = pd.read_csv(rets_path, index_col=0, parse_dates=[0], low_memory=False)
        rets_df = rets_df.sort_index()
    except Exception as e:
        print(f"{e}")
        return

    strategies = [col for col in rets_df.columns if col.lower() not in ["cash"]]

    if not strategies:
        print("No strategies found")
        return

    for strategy in strategies:
        returns = rets_df[strategy].dropna()

        strategy_features_df = pd.DataFrame(index=returns.index)

        for window in windows:
            process_returns_features(returns, strategy_features_df, window)
            process_additional_returns_features(returns, strategy_features_df, window)

        strategy_features_df["Cumulative_Returns"] = calculate_cumulative_returns(
            returns
        )
        strategy_features_df["Log_Returns"] = np.log(1 + returns).dropna()
        strategy_features_df[f"Mean_Returns_1"] = calculate_mean_returns(
            returns, window=1
        )

        strategy_features_df[
            "Volatility_of_Volatility_30"
        ] = calculate_volatility_of_volatility(
            returns.rolling(window=30).std(), window=30
        )
        strategy_features_df["Max_Drawdown_Duration_30"] = (
            calculate_max_drawdown_duration(strategy_features_df["Cumulative_Returns"])
            .rolling(window=30)
            .max()
        )

        strategy_features_path = os.path.join(
            output_dir, f"{strategy}_returns_features.csv"
        )
        strategy_features_df.to_csv(strategy_features_path)

    return


def main():
    data_path = ""
    rets_path = "rets.csv"
    output_rets_features_dir = "returns_features"
    tickers = ["BTCUSDT"]
    trading_days = 365
    windows = [10, 30, 60, 90]

    os.makedirs(output_rets_features_dir, exist_ok=True)

    for ticker in tickers:
        ticker_file = os.path.join(data_path, f"{ticker}.csv")
        if os.path.isfile(ticker_file):
            process_volatility_data(ticker, data_path, windows, trading_days)
        else:
            print(f"{ticker}.csv not found")

    if os.path.isfile(rets_path):
        process_rets_data(rets_path, output_rets_features_dir, windows, trading_days)
    else:
        print(f"{rets_path} not found")


if __name__ == "__main__":
    main()
