"""
Final Latency Forecasting Model — Regression (30-min ahead)
=============================================================
Predicts p99_latency_ms 30 minutes ahead using an LSTM trained on
engineered features from latency_data_v2.csv.

Design choices (see accompanying write-up for rationale):
  - Regression, not classification: gives a continuous curve + flexible
    alert thresholds, and classification metrics are derived from it anyway.
  - Log1p target transform: latency is right-skewed.
  - Chronological train/val/test split, scaler fit on train ONLY.
  - No p50/p95/p99 latency lags used as features by default (LATENCY_AWARE=False)
    to force the model to learn from true leading indicators (CPU/memory/DB/
    error/traffic) rather than latency autocorrelation. Flip the flag to True
    to see how much lift latency history adds.
  - Validates lead-time directly against failure_windows_v2.csv: for each
    injected incident, checks whether the model's predicted curve crossed
    the alert threshold before the incident's recorded start.
"""

import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import numpy as np
import random
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              confusion_matrix, classification_report)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

 
# CONFIG
 
DATA_PATH = 'DataSets/latency_data_v2.csv'
FAILURES_PATH = 'DataSets/failure_windows_v2.csv'

LOOKBACK = 12          # 1 hour of history (12 x 5-min steps)
DELAY = 6               # forecast 30 minutes ahead
ALERT_THRESHOLD = 300   # ms — used only for reporting classification metrics
LATENCY_AWARE = False   # True = include latency lag features (easier, less "early warning")

TRAIN_FRAC = 0.65
VAL_FRAC = 0.15         # test = remaining 0.20

 
# 1. LOAD + FEATURE ENGINEERING
 
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
failures = pd.read_csv(FAILURES_PATH)

hour_frac = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
df['hour_sin'] = np.sin(2 * np.pi * hour_frac / 24)
df['hour_cos'] = np.cos(2 * np.pi * hour_frac / 24)
dow = df['timestamp'].dt.dayofweek
df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
df['is_weekend'] = (dow >= 5).astype(int)

# Regression target — log1p smooths the right-skew
df['target_log_latency'] = np.log1p(df['p99_latency_ms'])

CAUSAL_FEATURES = [
    'request_count', 'active_users', 'cpu_usage', 'memory_usage', 'db_query_time_ms',
    'error_rate',
    'cpu_change_5m', 'memory_change_5m', 'db_change_5m',
    'cpu_2h_mean', 'cpu_2h_std', 'memory_2h_mean', 'memory_2h_std',
    'cpu_percentile', 'memory_percentile',
    'cpu_memory_stress', 'cpu_load_stress',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
]

FEATURES = CAUSAL_FEATURES + (['p99_latency_ms'] if LATENCY_AWARE else [])

print(f"Using {len(FEATURES)} features, LATENCY_AWARE={LATENCY_AWARE}")

 
# 2. CHRONOLOGICAL SPLIT (before any scaling — avoids leakage)
 
n = len(df)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

train_df = df.iloc[:train_end].reset_index(drop=True)
val_df = df.iloc[train_end:val_end].reset_index(drop=True)
test_df = df.iloc[val_end:].reset_index(drop=True)

print(f"Train: {len(train_df)} rows (days 0-{train_df['timestamp'].iloc[-1].day + (train_df['timestamp'].iloc[-1].month-4)*30:.0f})")
print(f"Val:   {len(val_df)} rows")
print(f"Test:  {len(test_df)} rows  ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

 
# 3. SCALE (fit on train only)
 
x_scaler = MinMaxScaler().fit(train_df[FEATURES])
y_scaler = MinMaxScaler().fit(train_df[['target_log_latency']])


def transform(sub_df):
    X = x_scaler.transform(sub_df[FEATURES])
    y = y_scaler.transform(sub_df[['target_log_latency']]).flatten()
    return X, y


X_train_raw, y_train_raw = transform(train_df)
X_val_raw, y_val_raw = transform(val_df)
X_test_raw, y_test_raw = transform(test_df)

 
# 4. SEQUENCE CREATION
 
def make_sequences(feature_arr, target_arr, lookback=LOOKBACK, delay=DELAY):
    X, y = [], []
    for i in range(len(feature_arr) - lookback - delay + 1):
        X.append(feature_arr[i: i + lookback])
        y.append(target_arr[i + lookback + delay - 1])
    return np.array(X), np.array(y)


X_train, y_train = make_sequences(X_train_raw, y_train_raw)
X_val, y_val = make_sequences(X_val_raw, y_val_raw)
X_test, y_test = make_sequences(X_test_raw, y_test_raw)

print(f"\nSequence shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

# Real (unscaled) target values aligned to test sequences, for plotting/reporting
test_actual_latency = test_df['p99_latency_ms'].values[LOOKBACK + DELAY - 1:]
test_timestamps = test_df['timestamp'].values[LOOKBACK + DELAY - 1:]

 
# 5. MODEL
 
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(96, return_sequences=True),
    Dropout(0.25),
    LSTM(48, return_sequences=False),
    Dropout(0.2),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='linear'),
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
]

history = model.fit(
    X_train, y_train,
    epochs=150,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    batch_size=64,
    verbose=1,
)

 
# 6. PREDICT + INVERT SCALING/LOG
 
pred_scaled = model.predict(X_test, verbose=0)
pred_log = y_scaler.inverse_transform(pred_scaled).flatten()
pred_latency = np.expm1(pred_log)

mae = mean_absolute_error(test_actual_latency, pred_latency)
rmse = np.sqrt(mean_squared_error(test_actual_latency, pred_latency))
mape = np.mean(np.abs((test_actual_latency - pred_latency) / test_actual_latency)) * 100

print("\n" + "=" * 60)
print("REGRESSION PERFORMANCE (held-out test set)")
print("=" * 60)
print(f"MAE:  {mae:.2f} ms")
print(f"RMSE: {rmse:.2f} ms")
print(f"MAPE: {mape:.2f} %")

# Naive persistence baseline: "predict the current latency, 30 min ago"
naive_pred = test_df['p99_latency_ms'].values[LOOKBACK - 1: -DELAY if DELAY else None][:len(test_actual_latency)]
if len(naive_pred) == len(test_actual_latency):
    naive_mae = mean_absolute_error(test_actual_latency, naive_pred)
    print(f"\nNaive persistence baseline MAE: {naive_mae:.2f} ms")
    print(f"Model improvement over naive:   {(1 - mae / naive_mae) * 100:.1f}%")

 
# 7. DERIVED CLASSIFICATION METRICS (threshold the regression output)
 
actual_spike = (test_actual_latency > ALERT_THRESHOLD).astype(int)
predicted_spike = (pred_latency > ALERT_THRESHOLD).astype(int)

print(f"\nClassification report (derived @ threshold={ALERT_THRESHOLD}ms):")
print(classification_report(actual_spike, predicted_spike, target_names=['Normal', 'Spike'], zero_division=0))

 
# 8. LEAD-TIME VALIDATION AGAINST KNOWN FAILURES
 
print("=" * 60)
print("LEAD-TIME CHECK (does the model warn before each incident?)")
print("=" * 60)

test_ts_series = pd.Series(pd.to_datetime(test_timestamps))
pred_series = pd.Series(pred_latency, index=test_ts_series)

lead_time_results = []
for _, fw in failures.iterrows():
    incident_start = pd.Timestamp('2026-04-01') + pd.Timedelta(days=fw['start_day'])
    incident_end = incident_start + pd.Timedelta(hours=fw['duration_hours'])

    # Only check incidents that fall inside the test window
    if incident_start < test_ts_series.min() or incident_start > test_ts_series.max():
        continue

    # Look at predictions in the 45 min before the incident officially starts
    lookback_window_start = incident_start - pd.Timedelta(minutes=45)
    pre_incident_preds = pred_series[(pred_series.index >= lookback_window_start) &
                                      (pred_series.index < incident_start)]

    warned = (pre_incident_preds > ALERT_THRESHOLD).any()
    first_warn_time = pre_incident_preds[pre_incident_preds > ALERT_THRESHOLD].index.min() if warned else None
    lead_minutes = (incident_start - first_warn_time).total_seconds() / 60 if warned else None

    lead_time_results.append({
        'incident': fw['name'],
        'incident_start': incident_start,
        'warned_early': warned,
        'lead_time_minutes': lead_minutes,
    })
    status = f"WARNED {lead_minutes:.0f} min early" if warned else "NO EARLY WARNING"
    print(f"  {fw['name']:25s} @ {incident_start}: {status}")

if lead_time_results:
    detected = sum(r['warned_early'] for r in lead_time_results)
    print(f"\nDetected {detected}/{len(lead_time_results)} in-test-window incidents with early warning.")
else:
    print("\nNo failure windows fall inside the test split — widen TEST split or check failure_windows_v2.csv.")

 
# 9. PLOTS
 
plt.figure(figsize=(15, 5))
plt.plot(test_timestamps, test_actual_latency, label='Actual p99 latency', alpha=0.8, linewidth=1)
plt.plot(test_timestamps, pred_latency, label='Predicted p99 latency (30-min ahead)', alpha=0.8, linewidth=1)
plt.axhline(ALERT_THRESHOLD, color='red', linestyle='--', label=f'Alert threshold ({ALERT_THRESHOLD}ms)', alpha=0.6)
plt.title(f'Predicted vs Actual p99 Latency — Test Set (MAE={mae:.1f}ms, RMSE={rmse:.1f}ms)', fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Latency (ms)')
plt.legend()
plt.tight_layout()
plt.savefig('Outputs/final_forecast_vs_actual.png', dpi=120)
plt.close()

plt.figure(figsize=(6, 5))
cm = confusion_matrix(actual_spike, predicted_spike)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Spike'],
            yticklabels=['Actual Normal', 'Actual Spike'])
plt.title(f'Alert Confusion Matrix (threshold={ALERT_THRESHOLD}ms)', fontweight='bold')
plt.tight_layout()
plt.savefig('Outputs/final_confusion_matrix.png', dpi=120)
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training History', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Huber Loss (scaled log-latency)')
plt.legend()
plt.tight_layout()
plt.savefig('Outputs/final_training_history.png', dpi=120)
plt.close()

print("\nSaved: final_forecast_vs_actual.png, final_confusion_matrix.png, final_training_history.png")

 
# 10. SAVE MODEL + SCALERS FOR REUSE
 
model.save('Outputs/latency_forecast_model.keras')
import joblib
joblib.dump(x_scaler, 'Outputs/x_scaler.pkl')
joblib.dump(y_scaler, 'Outputs/y_scaler.pkl')
print("Saved model + scalers for reuse (latency_forecast_model.keras, x_scaler.pkl, y_scaler.pkl)")