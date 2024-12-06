import os
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def select_columns(df, metrics, windows):
    selected_columns = [
        col for col in df.columns
        if any(metric in col for metric in metrics) and any(f"_{window}" in col for window in windows)
    ]
    return df[selected_columns]

windows = [30]
metrics = [
    'Realized_Volatility', 'Garman_Klass_Volatility', 'OU_Theta', 'Hurst', 'Momentum', 'RSI', 'ADX'
]

market_features = pd.read_csv('all_features_BTCUSDT.csv', index_col=0, parse_dates=[0], low_memory=False)
market_features = select_columns(market_features, metrics, windows)
market_features = market_features.resample('D').ffill()

strategies_data = {}
returns_features_path = 'returns_features'
for file in os.listdir(returns_features_path):
    if file.endswith('_returns_features.csv'):
        strategy_name = file.replace('_returns_features.csv', '')
        df = pd.read_csv(os.path.join(returns_features_path, file), index_col=0, parse_dates=[0], low_memory=False)
        df_selected = df[['Mean_Returns_30', 'ACF_Lag_1_30', 'Sharpe_Ratio_60']]
        df_selected = df_selected.resample('D').ffill()
        strategies_data[strategy_name] = df_selected


merged_data = {}
for strategy, df in strategies_data.items():
    merged_df = pd.merge(market_features, df, left_index=True, right_index=True, how='inner')
    if strategy in ['G59_V1', 'G59_V2', 'G90_V1', 'G24']:
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
strategy_names=train_data.keys()

def scale_fit_transform(sequences):
    scaler = StandardScaler()
    scaler.fit(sequences)
    scaled_sequences = scaler.transform(sequences)
    return scaled_sequences, scaler

def scale_transform(sequences, scaler):
    scaled_sequences = scaler.transform(sequences)
    return scaled_sequences


def assign_regime_labels(hidden_states_dict, regime_mapping_dict, strategy_names):
    regime_labels_dict = {}
    
    for strategy in strategy_names:
        hidden_states = hidden_states_dict[strategy]
        state_to_regime = regime_mapping_dict[strategy]
        regime_labels = [state_to_regime.get(state, 'Unknown') for state in hidden_states]
        regime_labels_dict[strategy] = regime_labels
    
    return regime_labels_dict

def get_regime_colors(regime_labels):
    REGIME_COLOR_MAPPING = {
        'Low': 'red',
        'Normal': 'blue',
        'High': 'green'
    }
    return [REGIME_COLOR_MAPPING.get(label, 'gray') for label in regime_labels]

def visualize_regimes(data, regime_labels, strategy, dataset_type='Test', method_name='HMM'):
    df = data[strategy].copy()
    df['Regime_Label'] = regime_labels[strategy]
    df['datetime'] = df.index  # Ensure datetime is available for plotting
    
    plt.figure(figsize=(15,7))
    
    plt.plot(df['datetime'], df['Sharpe_Ratio_60'], label='Sharpe Ratio', color='black', linewidth=1)
    
    colors = get_regime_colors(df['Regime_Label'])
    
    plt.scatter(df['datetime'], df['Sharpe_Ratio_60'], 
                c=colors, label='Regime', alpha=0.6, marker='o')
    
    handles = []
    labels = []
    unique_regimes = sorted(df['Regime_Label'].unique(), key=lambda x: {'Low':0, 'Normal':1, 'High':2}.get(x, 99))
    for regime in unique_regimes:
        color = get_regime_colors([regime])[0]
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=regime,
                                  markerfacecolor=color, markersize=10))
        labels.append(regime)
    
    plt.legend(handles, labels)
    plt.title(f'Sharpe Ratio Over Time with {method_name} Market Regimes for {strategy} ({dataset_type} Set)')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.tight_layout()
    plt.savefig(f'{strategy}.png')
    plt.show()

def fit_hmm(train_data, n_states=3, covariance_type='diag', n_iter=5000):
    hmm_models = {}
    
    for strategy in train_data.keys():
        X = train_data[strategy]['Sharpe_Ratio_60'].values.reshape(-1, 1)  
        
        model = hmm.GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter, random_state=42)
        
        model.fit(X)
        
        hmm_models[strategy] = model
            
    return hmm_models

def map_hmm_states_to_regimes(train_data, hmm_models, n_states=3, regime_labels=['Low', 'Normal', 'High']):
    regime_mapping_dict = {}

    for strategy in train_data.keys():
        X = train_data[strategy]['Sharpe_Ratio_60'].values.reshape(-1, 1)
        model = hmm_models[strategy]
        hidden_states = model.predict(X)

        state_means = []
        for state in range(n_states):
            state_mean = X[hidden_states == state].mean()
            state_means.append((state, state_mean))

        sorted_states = sorted(state_means, key=lambda x: x[1])

        state_to_regime = {}
        for i, (state, _) in enumerate(sorted_states):
            if i < len(regime_labels):
                state_to_regime[state] = regime_labels[i]

        regime_mapping_dict[strategy] = state_to_regime

    return regime_mapping_dict

def assign_regime_labels_train(train_data, hmm_models, regime_mapping_dict, strategy_names):
    train_regime_labels = {}
    
    for strategy in strategy_names:
        X_train = train_data[strategy]['Sharpe_Ratio_60'].values.reshape(-1, 1)
        model = hmm_models[strategy]
        hidden_states = model.predict(X_train)
        
        state_to_regime = regime_mapping_dict[strategy]
        regime_labels = [state_to_regime.get(state, 'Unknown') for state in hidden_states]
        train_regime_labels[strategy] = regime_labels
    
    return train_regime_labels

def predict_hmm_regimes(hmm_models, test_data):
    test_hidden_states = {}
    
    for strategy in test_data.keys():
        X_test = test_data[strategy]['Sharpe_Ratio_60'].values.reshape(-1, 1)  # (n_samples, n_features)
        
        model = hmm_models[strategy]
        
        hidden_states = model.predict(X_test)
        
        test_hidden_states[strategy] = hidden_states
            
    return test_hidden_states

def visualize_hmm_regimes(test_data, test_regime_labels, regime_mapping_dict, strategy, method_name='HMM'):
    state_to_regime = regime_mapping_dict[strategy]
    regime_labels = [state_to_regime.get(state, 'Unknown') for state in test_regime_labels[strategy]]
    
    test_regime_labels_qualitative = regime_labels
    
    test_regime_labels[strategy] = test_regime_labels_qualitative
    
    visualize_regimes(test_data, test_regime_labels, strategy, method_name=method_name)


strategy_names = train_data.keys()


hmm_models = fit_hmm(
    train_data, 
    n_states=3,  
    covariance_type='diag', 
    n_iter= 1000
)


regime_mapping_dict = map_hmm_states_to_regimes(
    train_data, 
    hmm_models, 
    n_states=3, 
    regime_labels=['Low', 'Normal', 'High']
)


test_hidden_states = predict_hmm_regimes(
    hmm_models, 
    test_data
)

test_regime_labels = assign_regime_labels(
    test_hidden_states, 
    regime_mapping_dict, 
    strategy_names=strategy_names
)


for strategy in strategy_names:
    visualize_regimes(
        test_data, 
        test_regime_labels, 
        strategy, 
        dataset_type='Test',
        method_name='HMM'
    )