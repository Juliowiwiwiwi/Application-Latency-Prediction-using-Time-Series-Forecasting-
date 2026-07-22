"""
FINAL Latency Forecasting Model — v3
=====================================
Trained on latency_data_v3.csv (150 days, 24 failure instances).

Two changes vs the previous version, specifically targeting the spike
UNDERPREDICTION problem seen in the chart (actual ~1250ms, predicted
~1050ms during the biggest spike):

  1. Loss changed from 'huber' -> 'mse'.
     Huber deliberately softens the gradient for large residuals (that's
     the whole point of Huber loss — it's designed to be robust to
     outliers). But here the "outliers" ARE the thing we care about
     most (the spikes), so Huber was actively working against us.

  2. Sample weighting during training.
     Without weighting, 95% of your training data is "normal" traffic
     and the model is never pushed hard to nail the rare extreme values.
     We now weight each training sample by how extreme its target latency
     is, so spike examples count several times more toward the loss than
     an average data point.

Everything else (chronological split, scaler fit on train only, no
latency-lag features, lead-time validation against failure windows)
stays the same — that part of the pipeline was already correct.
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
import joblib

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

DATA_PATH = 'DataSets/latency_data_v3.csv'
FAILURES_PATH = 'DataSets/failure_windows_v3.csv'

LOOKBACK = 12
DELAY = 6
ALERT_THRESHOLD = 300
LATENCY_AWARE = False

TRAIN_FRAC = 0.65
VAL_FRAC = 0.15

# How aggressively to upweight high-latency training samples.
# weight = 1 + SPIKE_WEIGHT_ALPHA * (scaled_target)^2
# scaled_target is 0-1 after MinMax scaling of the log-latency target, so
# a normal-traffic point gets weight ~1, and the most extreme point in
# train gets weight ~1+SPIKE_WEIGHT_ALPHA.
SPIKE_WEIGHT_ALPHA = 6.0


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


# 2. CHRONOLOGICAL SPLIT

n = len(df)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

train_df = df.iloc[:train_end].reset_index(drop=True)
val_df = df.iloc[train_end:val_end].reset_index(drop=True)
test_df = df.iloc[val_end:].reset_index(drop=True)

print(f"Train: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
print(f"Val:   {len(val_df)} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
print(f"Test:  {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")


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

test_actual_latency = test_df['p99_latency_ms'].values[LOOKBACK + DELAY - 1:]
test_timestamps = test_df['timestamp'].values[LOOKBACK + DELAY - 1:]


# 5. SAMPLE WEIGHTS — upweight spike examples so the model is pushed
#    to reproduce their full magnitude instead of averaging them down.

sample_weights_train = 1.0 + SPIKE_WEIGHT_ALPHA * (y_train ** 2)
print(f"\nSample weight range: min={sample_weights_train.min():.2f}, "
      f"max={sample_weights_train.max():.2f}, mean={sample_weights_train.mean():.2f}")


# 6. MODEL

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

# MSE instead of Huber: Huber caps the gradient on large residuals, which
# is exactly the behavior that was causing spikes to be underpredicted.
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
]

history = model.fit(
    X_train, y_train,
    sample_weight=sample_weights_train,
    epochs=150,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    batch_size=64,
    verbose=2,   # one line per epoch instead of a progress bar per batch
)


# 7. PREDICT + INVERT SCALING/LOG

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

naive_pred = test_df['p99_latency_ms'].values[LOOKBACK - 1: -DELAY if DELAY else None][:len(test_actual_latency)]
if len(naive_pred) == len(test_actual_latency):
    naive_mae = mean_absolute_error(test_actual_latency, naive_pred)
    print(f"\nNaive persistence baseline MAE: {naive_mae:.2f} ms")
    print(f"Model improvement over naive:   {(1 - mae / naive_mae) * 100:.1f}%")

# Specifically check peak-capture: how well does the model do on just the
# TOP 5% highest-latency moments in the test set? This is the number that
# tells you whether the underprediction problem is actually fixed.
top5_mask = test_actual_latency >= np.percentile(test_actual_latency, 95)
top5_actual = test_actual_latency[top5_mask]
top5_pred = pred_latency[top5_mask]
top5_mae = mean_absolute_error(top5_actual, top5_pred)
top5_underpred_pct = np.mean((top5_actual - top5_pred) / top5_actual) * 100
print(f"\nTop-5% latency moments (the spikes): MAE={top5_mae:.1f}ms, "
      f"avg underprediction={top5_underpred_pct:+.1f}% "
      f"(negative = model overpredicts, positive = still underpredicting)")


# 8. CLASSIFICATION METRICS (derived)

actual_spike = (test_actual_latency > ALERT_THRESHOLD).astype(int)
predicted_spike = (pred_latency > ALERT_THRESHOLD).astype(int)

print(f"\nClassification report (derived @ threshold={ALERT_THRESHOLD}ms):")
print(classification_report(actual_spike, predicted_spike, target_names=['Normal', 'Spike'], zero_division=0))


# 9. LEAD-TIME VALIDATION

print("=" * 60)
print("LEAD-TIME CHECK (does the model warn before each incident?)")
print("=" * 60)

test_ts_series = pd.Series(pd.to_datetime(test_timestamps))
pred_series = pd.Series(pred_latency, index=test_ts_series)

lead_time_results = []
for _, fw in failures.iterrows():
    incident_start = pd.Timestamp('2026-04-01') + pd.Timedelta(days=fw['start_day'])

    if incident_start < test_ts_series.min() or incident_start > test_ts_series.max():
        continue

    lookback_window_start = incident_start - pd.Timedelta(minutes=45)
    pre_incident_preds = pred_series[(pred_series.index >= lookback_window_start) &
                                      (pred_series.index < incident_start)]

    warned = (pre_incident_preds > ALERT_THRESHOLD).any()
    first_warn_time = pre_incident_preds[pre_incident_preds > ALERT_THRESHOLD].index.min() if warned else None
    lead_minutes = (incident_start - first_warn_time).total_seconds() / 60 if warned else None

    lead_time_results.append({'incident': fw['name'], 'warned_early': warned, 'lead_time_minutes': lead_minutes})
    status = f"WARNED {lead_minutes:.0f} min early" if warned else "NO EARLY WARNING"
    print(f"  {fw['name']:25s} @ {incident_start}: {status}")

if lead_time_results:
    detected = sum(r['warned_early'] for r in lead_time_results)
    print(f"\nDetected {detected}/{len(lead_time_results)} in-test-window incidents with early warning.")


# 10. PLOTS

plt.figure(figsize=(15, 5))
plt.plot(test_timestamps, test_actual_latency, label='Actual p99 latency', alpha=0.8, linewidth=1)
plt.plot(test_timestamps, pred_latency, label='Predicted p99 latency (30-min ahead)', alpha=0.8, linewidth=1)
plt.axhline(ALERT_THRESHOLD, color='red', linestyle='--', label=f'Alert threshold ({ALERT_THRESHOLD}ms)', alpha=0.6)
plt.title(f'Predicted vs Actual p99 Latency — Test Set (MAE={mae:.1f}ms, RMSE={rmse:.1f}ms)', fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Latency (ms)')
plt.legend()
plt.tight_layout()
plt.savefig('final_forecast_vs_actual_v3.png', dpi=120)
plt.close()

plt.figure(figsize=(6, 5))
cm = confusion_matrix(actual_spike, predicted_spike)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Spike'],
            yticklabels=['Actual Normal', 'Actual Spike'])
plt.title(f'Alert Confusion Matrix (threshold={ALERT_THRESHOLD}ms)', fontweight='bold')
plt.tight_layout()
plt.savefig('final_confusion_matrix_v3.png', dpi=120)
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training History', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MSE (scaled log-latency)')
plt.legend()
plt.tight_layout()
plt.savefig('final_training_history_v3.png', dpi=120)
plt.close()

print("\nSaved: final_forecast_vs_actual_v3.png, final_confusion_matrix_v3.png, final_training_history_v3.png")


# 11. SAVE MODEL + SCALERS

model.save('latency_forecast_model_v3.keras')
joblib.dump(x_scaler, 'x_scaler_v3.pkl')
joblib.dump(y_scaler, 'y_scaler_v3.pkl')
print("Saved model + scalers for reuse.")