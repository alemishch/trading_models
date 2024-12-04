import os
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from hurst import compute_Hc
import ta
from scipy import stats

def load_data(ticker, data_path):
    try:
        df = pd.read_csv(
            os.path.join(data_path, f"{ticker}.csv"),
            index_col=0,
            parse_dates=[0],
            low_memory=False  # Prevent DtypeWarning
        )
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None

### volatility 

def calculate_log_returns(df):
    return np.log(df['close'] / df['close'].shift(1)).dropna()

def calculate_atr(df, window=14):
    atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window)
    atr_values = atr.average_true_range()
    atr_values = np.maximum(atr_values, 0)  
    return atr_values

def calculate_bollinger_bands(df, window=20, window_dev=2):
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=window, window_dev=window_dev)
    bb_df = pd.DataFrame({
        'bollinger_upper': bollinger.bollinger_hband(),
        'bollinger_middle': bollinger.bollinger_mavg(),
        'bollinger_lower': bollinger.bollinger_lband()
    }, index=df.index)
    return bb_df

def calculate_ewma_volatility(returns, span=30, trading_days=252):
    ewma_variance = returns.ewm(span=span, adjust=False).mean()
    ewma_volatility = np.sqrt(np.maximum(ewma_variance, 0)) * np.sqrt(trading_days)  # Ensure non-negative
    return ewma_volatility

def calculate_historical_volatility(returns, window=30, trading_days=252):
    hv = returns.rolling(window=window).std() * np.sqrt(trading_days)
    hv = np.maximum(hv, 0)  # Ensure non-negative
    return hv

def calculate_realized_volatility(df, window=30, trading_days=252):
    intraday_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    daily_rv = intraday_returns.groupby(intraday_returns.index.date).apply(lambda x: np.sqrt(np.sum(x**2)))
    daily_rv.index = pd.to_datetime(daily_rv.index)
    realized_volatility = daily_rv.rolling(window=window).mean() * np.sqrt(trading_days)
    realized_volatility = np.maximum(realized_volatility, 0)  # Ensure non-negative
    return realized_volatility

def calculate_parkinson_volatility(df, window=30, trading_days=252):
    log_hl = np.log(df['high'] / df['low'])
    parkinson_variance = (1 / (4 * math.log(2))) * (log_hl ** 2)
    rolling_parkinson_var = parkinson_variance.rolling(window).sum() / window
    parkinson_volatility = np.sqrt(np.maximum(rolling_parkinson_var, 0)) * np.sqrt(trading_days)
    return parkinson_volatility

def calculate_garman_klass_volatility(df, window=30, trading_days=252):
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    gk_var = 0.5 * (log_hl ** 2) - (2 * math.log(2) - 1) * (log_co ** 2)
    rolling_gk_var = gk_var.rolling(window).sum() / window
    gk_volatility = np.sqrt(np.maximum(rolling_gk_var, 0)) * np.sqrt(trading_days)
    return gk_volatility

def calculate_max_drawdown_duration(cum_returns):
    max_duration = pd.Series(index=cum_returns.index, dtype='float64')
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

### trend Features

def calculate_macd(df, window_slow=26, window_fast=12, window_sign=9):
    macd = ta.trend.MACD(close=df['close'], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
    macd_df = pd.DataFrame({
        'macd': macd.macd(),
        'macd_signal': macd.macd_signal(),
        'macd_diff': macd.macd_diff()
    }, index=df.index)
    return macd_df

def calculate_rsi(df, window=14):
    rsi = ta.momentum.RSIIndicator(close=df['close'], window=window)
    rsi_values = rsi.rsi()
    rsi_values = np.maximum(rsi_values, 0)  # Ensure non-negative
    return rsi_values

def calculate_adx(df, window=14):
    adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=window)
    adx_values = adx.adx()
    adx_values = np.maximum(adx_values, 0)  # Ensure non-negative
    return adx_values

def calculate_momentum(df, window=10):
    roc = ta.momentum.ROCIndicator(close=df['close'], window=window)
    momentum = roc.roc()
    momentum = np.maximum(momentum, 0) 
    return momentum

def compute_hurst_exponent(prices):
    if len(prices) < 20:
        return np.nan
    try:
        H, c, data = compute_Hc(prices, kind='price', simplified=True)
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

### returns

def calculate_acf(returns, lags=10):
    acf_values = [returns.autocorr(lag=i) for i in range(1, lags + 1)]
    return acf_values

def calculate_mean_returns(returns, window=10):
    return returns.rolling(window=window).mean()

def calculate_sharpe_ratio(returns, window=30, risk_free_rate=0.0, trading_days=252):
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    sharpe = (rolling_mean - (risk_free_rate / trading_days)) / rolling_std * np.sqrt(trading_days)
    sharpe = np.where(rolling_std != 0, sharpe, np.nan)  # Handle division by zero
    sharpe = pd.Series(sharpe, index=returns.index)
    return sharpe

def calculate_sortino_ratio(returns, window=30, risk_free_rate=0.0, trading_days=252):
    rolling_mean = returns.rolling(window).mean()
    rolling_std_down = returns.where(returns < 0).rolling(window).std()
    sortino = (rolling_mean - (risk_free_rate / trading_days)) / rolling_std_down * np.sqrt(trading_days)
    sortino = sortino.where(rolling_std_down != 0, np.nan)  
    return sortino

def calculate_calmar_ratio(returns, window=30, trading_days=252):
    rolling_return = (1 + returns).rolling(window).apply(lambda x: x.prod(), raw=True) - 1
    rolling_drawdown = calculate_max_drawdown_duration(rolling_return)
    calmar = rolling_return / np.abs(rolling_drawdown)
    calmar = np.where(rolling_drawdown != 0, calmar, np.nan)
    calmar = pd.Series(calmar, index=returns.index)
    return calmar

def calculate_omega_ratio(returns, threshold=0.0, window=30):
    excess_returns = returns - threshold
    omega = (excess_returns[excess_returns > 0].sum() + threshold * (excess_returns > 0).sum()) / \
            (np.abs(excess_returns[excess_returns < 0].sum()) + threshold * (excess_returns < 0).sum())
    return omega

def calculate_jarque_bera(returns):
    jb_statistic, jb_pvalue = stats.jarque_bera(returns)
    return jb_statistic, jb_pvalue

def calculate_skewness_kurtosis(returns, window=30):
    skew = returns.rolling(window).skew()
    kurtosis = returns.rolling(window).kurt()
    skew = np.where(skew >= -5, skew, np.nan)  # Handle extreme skewness
    kurtosis = np.where(kurtosis >= -10, kurtosis, np.nan)  # Handle extreme kurtosis
    return pd.DataFrame({'skewness': skew, 'kurtosis': kurtosis}, index=returns.index)

def calculate_cumulative_returns(returns):
    return (1 + returns).cumprod() - 1

def compute_hurst_returns(returns):
    if len(returns) < 20:
        return np.nan
    try:
        H, c, data = compute_Hc(returns, kind='price', simplified=True)
        return H
    except Exception:
        return np.nan

### features by category

def process_volatility_features(df_daily, volatility_df, window, trading_days=252):
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
    
    volatility_df[f"Garman_Klass_Volatility_{window}"] = calculate_garman_klass_volatility(
        df_daily, window=window, trading_days=trading_days
    )
    
    volatility_df[f"Volatility_of_Volatility_{window}"] = calculate_volatility_of_volatility(
        volatility_df[f"Historical_Volatility_{window}"], window=window
    )
    
    volatility_df[f"Max_Drawdown_Duration_{window}"] = calculate_max_drawdown_duration(
        df_daily["Cumulative_Returns"]
    ).rolling(window=window).max()

def process_trend_features(df_daily, features_df, window):
    macd_df = calculate_macd(df_daily)
    features_df[f"MACD_{window}"] = macd_df['macd']
    features_df[f"MACD_Signal_{window}"] = macd_df['macd_signal']
    features_df[f"MACD_Diff_{window}"] = macd_df['macd_diff']
    
    features_df[f"RSI_{window}"] = calculate_rsi(df_daily, window=window)
    features_df[f"ADX_{window}"] = calculate_adx(df_daily, window=window)
    features_df[f"Momentum_{window}"] = calculate_momentum(df_daily, window=window)
    
    features_df[f"Hurst_{window}"] = df_daily['close'].rolling(window).apply(compute_hurst_exponent, raw=False)
    features_df[f"OU_Theta_{window}"] = df_daily['close'].rolling(window).apply(compute_ou_theta, raw=False)

def process_returns_features(returns, features_df, window):
    features_df[f"Mean_Returns_{window}"] = calculate_mean_returns(returns, window=window)
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
    features_df[f"Skewness_{window}"] = skew_kurt['skewness']
    features_df[f"Kurtosis_{window}"] = skew_kurt['kurtosis']
    
    features_df[f"Hurst_Returns_{window}"] = returns.rolling(window).apply(compute_hurst_returns, raw=False)
    
    for lag in range(1, 11):
        features_df[f"ACF_Lag_{lag}_{window}"] = returns.rolling(window).apply(lambda x: x.autocorr(lag=lag), raw=False)
    
    jb_stat, jb_p = calculate_jarque_bera(returns)
    features_df["Jarque-Bera_statistic"] = jb_stat
    features_df["Jarque-Bera_pvalue"] = jb_p
    
    features_df["VaR_95"] = returns.rolling(window).quantile(0.05)
    features_df["CVaR_95"] = returns.rolling(window).apply(lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else np.nan, raw=False)
    
    features_df["Omega_Ratio"] = calculate_omega_ratio(returns, threshold=0.0, window=window)


def process_volatility_data(ticker, data_path, windows, trading_days=252):
    df = load_data(ticker, data_path)
    if df is None or df.empty:
        print(f"No data for {ticker}.")
        return

    df_daily = df.resample("D").agg({
        "close": "last",
        "high": "max",
        "low": "min",
        "open": "first",
        "volume": "sum"
    }).dropna()

    if df_daily.empty:
        print(f"No daily data for {ticker}.")
        return

    df_daily["Log_Returns"] = calculate_log_returns(df_daily)
    df_daily["Cumulative_Returns"] = calculate_cumulative_returns(df_daily["Log_Returns"])
    
    volatility_df = pd.DataFrame(index=df_daily.index)
    
    for window in windows:
        process_volatility_features(df_daily, volatility_df, window, trading_days)
    
    features_df = volatility_df.copy()
    
    for window in windows:
        process_trend_features(df_daily, features_df, window)
    
    features_df["Volatility_of_Volatility_30"] = calculate_volatility_of_volatility(
        features_df["Historical_Volatility_30"], window=30
    )
    
    features_df["Max_Drawdown_Duration_30"] = calculate_max_drawdown_duration(
        df_daily["Cumulative_Returns"]  
    ).rolling(window=30).max()
    
    all_features = features_df.copy()
    
    #numeric_columns = all_features.select_dtypes(include=[np.number]).columns
    #all_features[numeric_columns] = all_features[numeric_columns].applymap(lambda x: x if x >=0 else np.nan)
    
    all_features.to_csv(f"all_features_{ticker}.csv")
    return all_features

def process_rets_data(rets_path, output_path, windows, trading_days=252):
    try:
        rets_df = pd.read_csv(rets_path, index_col=0, parse_dates=[0], low_memory=False)
        rets_df = rets_df.sort_index()
    except Exception as e:
        print(f"Error loading rets.csv: {e}")
        return
    
    strategies = [col for col in rets_df.columns if col.lower() not in ['cash']]
    
    if not strategies:
        print("No strategies found in rets.csv.")
        return
    
    features_df = pd.DataFrame(index=rets_df.index)
    
    for strategy in strategies:
        returns = rets_df[strategy].dropna()
        for window in windows:
            process_returns_features(returns, features_df, window)
    
    # cumulative and log returns based on average of all strategies
    average_returns = rets_df[strategies].mean(axis=1).dropna()
    features_df["Cumulative_Returns"] = calculate_cumulative_returns(average_returns)
    features_df["Log_Returns"] = np.log(1 + average_returns).dropna()
    
    # additional returns metrics
    features_df["Volatility_of_Volatility_30"] = calculate_volatility_of_volatility(
        rets_df[strategies].mean(axis=1).rolling(window=30).std(), window=30
    )
    features_df["Max_Drawdown_Duration_30"] = calculate_max_drawdown_duration(
        features_df["Cumulative_Returns"]
    ).rolling(window=30).max()
    
    #numeric_columns = features_df.select_dtypes(include=[np.number]).columns
    #features_df[numeric_columns] = features_df[numeric_columns].applymap(lambda x: x if x >=0 else np.nan)
    
    features_df.to_csv(output_path)
    return features_df


def main():
    data_path = ""  ######
    rets_path = "rets.csv" 
    output_rets_features = "rets_features.csv" 
    tickers = ["BTCUSDT"]  
    trading_days = 252
    windows = [10, 30]
    
    for ticker in tickers:
        ticker_file = os.path.join(data_path, f"{ticker}.csv")
        if os.path.isfile(ticker_file):
            process_volatility_data(ticker, data_path, windows, trading_days)
        else:
            print(f"{ticker}.csv not found")
    
    if os.path.isfile(rets_path):
        process_rets_data(rets_path, output_rets_features, windows, trading_days)
    else:
        print(f"{rets_path} not found")

if __name__ == "__main__":
    main()
