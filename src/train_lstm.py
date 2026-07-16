import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_sequences(data, target, lookback, horizon):
    """
    Creates sequences for the LSTM
    If lookback is 12 (60 mins) the model looks at 12 rows of data
    If horizon is 6 (30 mins) the target is 6 rows into the future
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i : (i + lookback)])
        y.append(target[i + lookback + horizon])
    return np.array(X), np.array(y)

def run_lstm_model(csv_path):
    print("Loading dataset")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)


    # Feature Engineering 
    print("Preparing features for Deep Learning")
    # The LSTM will look at ALL these variables simultaneously
    features = ['request_count', 'cpu_usage', 'db_query_time_ms', 'p50_latency_ms', 'p99_latency_ms']
    
    # Neural Networks need data scaled between 0 and 1
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_X = scaler_X.fit_transform(df[features])
    # We specifically want to predict the p99 latency
    scaled_y = scaler_y.fit_transform(df[['p99_latency_ms']])

    # Look back 60 minutes (12 steps of 5 mins)
    LOOKBACK = 12 
    # Predict 30 minutes into the future (6 steps of 5 mins)
    HORIZON = 6   

    X, y = create_sequences(scaled_X, scaled_y, LOOKBACK, HORIZON)


    #Train/Test Split
    #the last 2 days (576 intervals) for testing train on the rest
    test_size = 576 
    
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    #timestamps for the test set to plot them later
    test_dates = df.index[-test_size:]

    print(f"Training data shape: {X_train.shape} (Samples, Timesteps, Features)")

    

    print("Building and training the LSTM Model")
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(LOOKBACK, len(features))),
        Dropout(0.2), # Prevents overfitting
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    
 
    history = model.fit(
        X_train, y_train, 
        epochs=15, 
        batch_size=32, 
        validation_split=0.1, 
        verbose=1
    )

    

    print("\nForecasting 30minutes ahead for the Test Set")
    predictions = model.predict(X_test)

    # Un scale the data back to milliseconds
    predictions_ms = scaler_y.inverse_transform(predictions)
    actual_ms = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(actual_ms, predictions_ms)
    
    print(f"LSTM 30-Min Ahead MAE: {mae:.2f} ms")



    plt.figure(figsize=(15, 6))
    plt.plot(test_dates, actual_ms, label='Actual p99 Latency', color='purple', alpha=0.5, linewidth=2)
    plt.plot(test_dates, predictions_ms, label='LSTM 30-Min Ahead Forecast', color='blue', linestyle='--', linewidth=2)
    
    plt.title(f'AI Prediction vs Actual Latency 30 Minutes Later (MAE: {mae:.2f} ms)', fontsize=14)
    plt.ylabel('Milliseconds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    run_lstm_model('DataSets/latency_data.csv')