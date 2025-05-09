import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

def rnn_prediction(input, output, target, key_cols, sequence_length):
    # Load and sort data
    df = pd.read_csv(input)
    df = df.sort_values(['Player', 'year'])

    # Get index of only players with enough seasons because LSTM requires a sequence of seasons
    player_season_counts = df.groupby('Player')['year'].nunique()
    eligible_players = player_season_counts[player_season_counts > sequence_length].index

    # Generate sequential training samples
    X, y, metadata = [], [], []

    for player in eligible_players:
        player_data = df[df['Player'] == player]
        feature_data = player_data[key_cols].copy().reset_index(drop=True)
        target_series = player_data[target].reset_index(drop=True)
        year_series = player_data['year'].reset_index(drop=True)

        for i in range(len(feature_data) - sequence_length):
            seq = feature_data.iloc[i:i + sequence_length].values
            if seq.shape != (sequence_length, len(key_cols)):
                continue
            X.append(seq)
            y.append(target_series[i + sequence_length])
            metadata.append((player, year_series[i + sequence_length]))

    X = np.array(X)
    y = np.array(y)

    # Normalize the features and the target 
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_flat = X.reshape(-1, X.shape[2])
    X_scaled = scaler_X.fit_transform(X_flat).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Split into training and test sets
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    test_metadata = metadata[split_idx:]

    # Build and train the LSTM model
    model = Sequential([
        LSTM(50, input_shape=(sequence_length, X.shape[2])),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Predict and evaluate
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)

    # Save predictions
    results_df = pd.DataFrame({
        'Player': [p for p, _ in test_metadata],
        'Year': [y for _, y in test_metadata],
        'Actual_' + target: y_test_orig,
        'Predicted_' + target: y_pred
    })
    results_df.to_csv(output, index=False)

    return model, results_df, rmse, r2

def rf_prediction(input, output, target, key_columns):
    df = pd.read_csv(input).copy()
    
    if target not in df.columns:
        raise ValueError(f"{target} column not found in data")
    
    df = df.sort_values(['Player', 'year']).reset_index(drop=True)

    # Lag features
    df['WAR_lag1'] = df.groupby('Player')[target].shift(1)
    df['WAR_lag2'] = df.groupby('Player')[target].shift(2)
    df['WAR_lag3'] = df.groupby('Player')[target].shift(3)

    used_cols = key_columns + ['WAR_lag1', 'WAR_lag2', 'WAR_lag3']
    df = df[['Player', 'year', target] + used_cols].dropna().reset_index(drop=True)

    X = df[used_cols]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Important: save full index to reconstruct predictions
    df['index_in_df'] = df.index

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, df['index_in_df'], test_size=0.2
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    df_pred = df.loc[idx_test].copy()
    df_pred['Predicted_WAR'] = y_pred
    df_pred.to_csv(output, index=False)

    return model, df_pred, rmse, r2

def rf_predict_multiple_stats_2025(input_csv, targets, key_columns,
                                   eval_output_csv='Eval_Multi.csv',
                                   predict_2025_csv='Predictions_2025.csv'):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from functools import reduce

    df = pd.read_csv(input_csv).sort_values(['Player', 'year']).reset_index(drop=True)
    
    all_cols = list(set(targets + key_columns))
    for col in all_cols:
        for i in [1, 2, 3]:
            df[f'{col}_lag{i}'] = df.groupby('Player')[col].shift(i)

    lag_cols = [f'{col}_lag{i}' for col in all_cols for i in [1, 2, 3]]
    df_model = df.dropna(subset=lag_cols).copy()

    all_eval_rows = []
    all_pred_dfs = []

    for target in targets:
        print(f"\n📊 Predicting {target}...")
        df_train = df_model[df_model['year'] < 2025]
        features = [f'{col}_lag{i}' for col in all_cols for i in [1, 2, 3]]
        X = df_train[features]
        y = df_train[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_scaled, y, df_train.index, test_size=0.2
        )

        model = RandomForestRegressor(n_estimators=200, max_depth=5)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"{target} RMSE: {rmse:.3f}, R²: {r2:.3f}")

        eval_df = df_train.loc[idx_test, ['Player', 'year', target]].copy()
        eval_df[f'Predicted_{target}'] = y_pred
        all_eval_rows.append(eval_df)

        # Predict 2025
        latest = df[df['year'] < 2025].groupby('Player').tail(3).copy()
        counts = latest['Player'].value_counts()
        valid_players = counts[counts == 3].index
        latest = latest[latest['Player'].isin(valid_players)]

        pred_rows, players = [], []
        for player in valid_players:
            pl_data = latest[latest['Player'] == player].sort_values('year')
            row = {}
            try:
                for i, lag in enumerate([3, 2, 1]):
                    for col in all_cols:
                        row[f'{col}_lag{lag}'] = pl_data.iloc[i][col]
                pred_rows.append(row)
                players.append(player)
            except Exception:
                continue

        if not pred_rows:
            continue

        X_pred = pd.DataFrame(pred_rows)
        X_pred_scaled = scaler.transform(X_pred[features])
        y_pred_2025 = model.predict(X_pred_scaled)

        df_pred = pd.DataFrame({
            'Player': players,
            f'Predicted_{target}_2025': y_pred_2025
        })
        all_pred_dfs.append(df_pred)

    # Merge all prediction DataFrames by Player (outer join)
    if all_pred_dfs:
        merged_pred_df = reduce(lambda left, right: pd.merge(left, right, on='Player', how='outer'), all_pred_dfs)
    else:
        merged_pred_df = pd.DataFrame(columns=['Player'] + [f'Predicted_{t}_2025' for t in targets])

    merged_pred_df.to_csv(predict_2025_csv, index=False)

    # Combine evaluation results
    full_eval_df = pd.concat(all_eval_rows, axis=0)
    full_eval_df.to_csv(eval_output_csv, index=False)

    return merged_pred_df, full_eval_df


input = "Complete_Data.csv"
output1 = "Predictions_rnn.csv"
output2 = "Predictions_rf.csv"
key_columns = ['xba', 'barrel_batted_rate', 'player_age', 'batting_avg', 'k_percent', 'bb_percent', 'on_base_plus_slg', 'Rbat+']
target = "WAR"
target_stats = ['WAR', 'home_run', 'OPS+']
sequence_length = 3
#model, df, rmse, r2 = rnn_prediction(input, output1, target, key_columns, sequence_length)
#print(f"RMSE: {rmse:.3f}")
#print(f"R²: {r2:.3f}")
model, predictions, rmse, r2 = rf_prediction(input, output2, target, key_columns)
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")
pred_2025_df, eval_df = rf_predict_multiple_stats_2025(
    input_csv=input,
    targets=target_stats,
    key_columns=key_columns,
    eval_output_csv='Eval_MultiStats.csv',
    predict_2025_csv='Predictions_2025_Multi.csv'
)
print(pred_2025_df.head())
