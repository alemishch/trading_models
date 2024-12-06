import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from dtaidistance import dtw
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

REGIME_COLOR_MAPPING = {
    'Low': 'red',
    'Normal': 'blue',
    'High': 'green'
}

def get_regime_colors(regime_labels):
    return [REGIME_COLOR_MAPPING.get(label, 'gray') for label in regime_labels]

def visualize_regimes(test_data, test_regime_labels, strategy, method_name='Clustering'):
    df_test = test_data[strategy].copy()
    df_test['Regime_Label'] = test_regime_labels[strategy]
    df_test['datetime'] = df_test.index  # Ensure datetime is available for plotting
    
    plt.figure(figsize=(15,7))
    
    plt.plot(df_test['datetime'], df_test['Sharpe_Ratio_60'], label='Sharpe Ratio', color='black', linewidth=1)
    
    colors = get_regime_colors(df_test['Regime_Label'])
    
    plt.scatter(df_test['datetime'], df_test['Sharpe_Ratio_60'], 
                c=colors, label='Regime', alpha=0.6, marker='o')
    
    handles = []
    labels = []
    for regime, color in REGIME_COLOR_MAPPING.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=regime,
                                  markerfacecolor=color, markersize=10))
        labels.append(regime)

    if 'Noise' in df_test['Regime_Label'].unique():
        handles.append(plt.Line2D([0], [0], marker='x', color='w', label='Noise',
                                  markerfacecolor='gray', markersize=10))
        labels.append('Noise')
    
    plt.legend(handles, labels)
    plt.title(f'Sharpe Ratio Over Time with {method_name} Market Regimes for {strategy} (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.tight_layout()
    plt.show()

def pad_truncate_sequences(sequences, max_length):
    """
    makes sequences of same length
    """
    fixed_length_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            # Pad with zeros
            padded_seq = np.pad(seq, (0, max_length - len(seq)), 'constant')
        else:
            # Truncate to max_length
            padded_seq = seq[:max_length]
        fixed_length_sequences.append(padded_seq)
    return np.array(fixed_length_sequences)

def scale_sequences_fit_transform(sequences):
    #for train
    scaler = StandardScaler()
    scaler.fit(sequences)
    scaled_sequences = scaler.transform(sequences)
    return scaled_sequences, scaler

def scale_sequences_transform(sequences, scaler):
    #for test
    scaled_sequences = scaler.transform(sequences)
    return scaled_sequences

def fit_vlstar(train_data, n_clusters=3):
    kmedoids_models = {}
    scalers = {}
    train_labels = {}
    X_train_scaled_dict = {}
    
    for strategy in train_data.keys():
        X_train = train_data[strategy]['Sharpe_Ratio_60'].values.tolist()
        
        X_train = np.array(X_train)
        
        max_length_train = X_train.shape[1] if X_train.ndim > 1 else 1
        
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        
        # pad and scale 
        X_train_padded = pad_truncate_sequences(X_train, max_length_train)
        X_train_scaled, scaler = scale_sequences_fit_transform(X_train_padded)
        scalers[strategy] = scaler
        X_train_scaled_dict[strategy] = X_train_scaled
        
        # distance matrix for train
        n_train = X_train_scaled.shape[0]
        distance_matrix = np.zeros((n_train, n_train))
        for i in range(n_train):
            for j in range(i + 1, n_train):
                distance = dtw.distance(X_train_scaled[i], X_train_scaled[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance 
        
        # k-meoids with dist matrix
        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', init='k-medoids++', random_state=42)
        kmedoids.fit(distance_matrix)
        
        train_labels[strategy] = kmedoids.labels_
        
        kmedoids_models[strategy] = kmedoids
        
    return kmedoids_models, scalers, train_labels, X_train_scaled_dict


def assign_test_labels(kmedoids_models, scalers, test_data, X_train_scaled_dict, n_clusters=3):
    #label test data based on train
    test_labels = {}
    
    for strategy in test_data.keys():
        X_test = test_data[strategy]['Sharpe_Ratio_60'].values.tolist()
        X_test = np.array(X_test)
        
        max_length_train = X_train_scaled_dict[strategy].shape[1] if X_train_scaled_dict[strategy].ndim > 1 else 1
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        X_test_padded = pad_truncate_sequences(X_test, max_length_train)
        
        scaler = scalers[strategy]
        X_test_scaled = scale_sequences_transform(X_test_padded, scaler)
        
        kmedoids = kmedoids_models[strategy]
        
        medoid_indices = kmedoids.medoid_indices_
        medoids = X_train_scaled_dict[strategy][medoid_indices]
        
        n_test = X_test_scaled.shape[0]
        distances_test = np.zeros((n_test, n_clusters))
        for i in range(n_test):
            for j in range(n_clusters):
                distances_test[i, j] = dtw.distance(X_test_scaled[i], medoids[j])
        
        test_labels[strategy] = distances_test.argmin(axis=1)
    
    return test_labels

def map_clusters_to_regimes(train_data, train_labels, test_labels, regime_labels=['Low', 'Normal', 'High']):
    #label names based on train
    train_regime_labels = {}
    test_regime_labels = {}
    
    for strategy in train_data.keys():
        df_train = train_data[strategy].copy()
        labels = train_labels[strategy]
        df_train['Cluster'] = labels
        
        cluster_sharpe = df_train.groupby('Cluster')['Sharpe_Ratio_60'].mean()
        
        sorted_clusters = cluster_sharpe.sort_values().index.tolist()
        
        regime_mapping = {}
        num_regimes = len(regime_labels)
        if len(sorted_clusters) >= num_regimes:
            for cluster, label in zip(sorted_clusters[:num_regimes], regime_labels):
                regime_mapping[cluster] = label
            for cluster in sorted_clusters[num_regimes:]:
                regime_mapping[cluster] = f'Label_{cluster}'
        else:
            for cluster, label in zip(sorted_clusters, regime_labels):
                regime_mapping[cluster] = label
        
        df_train['Regime_Label'] = df_train['Cluster'].map(regime_mapping)
        train_regime_labels[strategy] = df_train['Regime_Label']
        
        labels_test = test_labels[strategy]
        regime_label_test = []
        for lbl in labels_test:
            regime = regime_mapping.get(lbl, 'Noise')  # 'Noise' if label not found
            regime_label_test.append(regime)
        
        test_regime_labels[strategy] = regime_label_test
        
    return train_regime_labels, test_regime_labels




kmedoids_models_vlstar, scalers_vlstar, vlstar_train_labels, X_train_scaled_dict = fit_vlstar(
    train_data, 
    n_clusters=3
)

vlstar_test_labels = assign_test_labels(
    kmedoids_models_vlstar, 
    scalers_vlstar, 
    test_data, 
    X_train_scaled_dict, 
    n_clusters=3
)


vlstar_train_regime_labels, vlstar_test_regime_labels = map_clusters_to_regimes(
    train_data, 
    vlstar_train_labels, 
    vlstar_test_labels, 
    regime_labels=['Low', 'Normal', 'High']
)


for strategy in strategy_names:
    visualize_regimes(test_data, vlstar_test_regime_labels, strategy, method_name='VLSTAR')
