"""
FINAL Latency Forecasting Model — v4
=====================================
Trained on latency_data_v4.csv (150 days, 24 failures, WITH precursor
signal — see GenerateDataV4.py for why that matters).

Two things changed vs v3:

1. LOOKBACK increased from 12 steps (1h) to 18 steps (1.5h). The
   precursor window is 30 minutes; giving the model 90 minutes of
   history instead of 60 comfortably guarantees the precursor is fully
   visible inside the lookback window for targets right at/after
   incident start, instead of being clipped at the edge.

2. The lead-time check is REWRITTEN. The v3 version checked whether
   *predictions* were elevated in the 45 minutes BEFORE an incident
   started — but *actual* latency is still normal in that window too
   (the precursor drift is in CPU/memory, not yet in latency), so a
   correct model SHOULD predict "normal" there. That's not a failure,
   it was just measuring the wrong thing.

   The new check instead asks the question that actually matters for
   an alerting system: "using only data available before time X, does
   the model's forecast cross the alert threshold before the ACTUAL
   latency does?" That's the standard definition of alerting lead time
   (how many minutes of warning did ops actually get).

Everything else (MSE loss + sample weighting for spike magnitude,
chronological split, scaler fit on train only) is unchanged from v3 —
that part was already confirmed working (top-5% underprediction moved
from positive/bad to slightly negative/good).
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

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_PATH = 'DataSets/latency_data_v4.csv'
FAILURES_PATH = 'DataSets/failure_windows_v4.csv'

LOOKBACK = 18          # 1.5 hours — long enough to fully see the 30-min precursor
DELAY = 6              # forecast 30 minutes ahead
ALERT_THRESHOLD = 300
LATENCY_AWARE = False

TRAIN_FRAC = 0.65
VAL_FRAC = 0.15

SPIKE_WEIGHT_ALPHA = 6.0

# ----------------------------------------------------------------------
# 1. LOAD + FEATURE ENGINEERING
# ----------------------------------------------------------------------
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
print(f"Using {len(FEATURES)} features, LATENCY_AWARE={LATENCY_AWARE}, LOOKBACK={LOOKBACK} steps")

# ----------------------------------------------------------------------
# 2. CHRONOLOGICAL SPLIT
# ----------------------------------------------------------------------
n = len(df)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

train_df = df.iloc[:train_end].reset_index(drop=True)
val_df = df.iloc[train_end:val_end].reset_index(drop=True)
test_df = df.iloc[val_end:].reset_index(drop=True)

print(f"Train: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
print(f"Val:   {len(val_df)} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
print(f"Test:  {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

# ----------------------------------------------------------------------
# 3. SCALE (fit on train only)
# ----------------------------------------------------------------------
x_scaler = MinMaxScaler().fit(train_df[FEATURES])
y_scaler = MinMaxScaler().fit(train_df[['target_log_latency']])


def transform(sub_df):
    X = x_scaler.transform(sub_df[FEATURES])
    y = y_scaler.transform(sub_df[['target_log_latency']]).flatten()
    return X, y


X_train_raw, y_train_raw = transform(train_df)
X_val_raw, y_val_raw = transform(val_df)
X_test_raw, y_test_raw = transform(test_df)

# ----------------------------------------------------------------------
# 4. SEQUENCE CREATION
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 5. SAMPLE WEIGHTS (unchanged from v3 — already confirmed working)
# ----------------------------------------------------------------------
sample_weights_train = 1.0 + SPIKE_WEIGHT_ALPHA * (y_train ** 2)
print(f"\nSample weight range: min={sample_weights_train.min():.2f}, "
      f"max={sample_weights_train.max():.2f}, mean={sample_weights_train.mean():.2f}")

# ----------------------------------------------------------------------
# 6. MODEL (unchanged architecture — LSTM is not the bottleneck here)
# ----------------------------------------------------------------------
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
    verbose=2,
)

# ----------------------------------------------------------------------
# 7. PREDICT + INVERT SCALING/LOG
# ----------------------------------------------------------------------
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

top5_mask = test_actual_latency >= np.percentile(test_actual_latency, 95)
top5_actual = test_actual_latency[top5_mask]
top5_pred = pred_latency[top5_mask]
top5_mae = mean_absolute_error(top5_actual, top5_pred)
top5_underpred_pct = np.mean((top5_actual - top5_pred) / top5_actual) * 100
print(f"\nTop-5% latency moments (the spikes): MAE={top5_mae:.1f}ms, "
      f"avg underprediction={top5_underpred_pct:+.1f}%")

# ----------------------------------------------------------------------
# 8. CLASSIFICATION METRICS
# ----------------------------------------------------------------------
actual_spike = (test_actual_latency > ALERT_THRESHOLD).astype(int)
predicted_spike = (pred_latency > ALERT_THRESHOLD).astype(int)

print(f"\nClassification report (derived @ threshold={ALERT_THRESHOLD}ms):")
print(classification_report(actual_spike, predicted_spike, target_names=['Normal', 'Spike'], zero_division=0))

# ----------------------------------------------------------------------
# 9. LEAD-TIME VALIDATION — CORRECTED METHODOLOGY
#
#    For each failure: find when ACTUAL latency first crosses the alert
#    threshold. Then find when the model's PREDICTIONS first cross the
#    threshold (searching from a bit before the incident onward). If the
#    prediction crossing happens BEFORE the actual crossing, that gap in
#    minutes is genuine lead time — the model warned before the real
#    spike was even visible in the raw metric.
# ----------------------------------------------------------------------
print("=" * 60)
print("LEAD-TIME CHECK (corrected: forecast-crossing vs actual-crossing)")
print("=" * 60)

test_ts_series = pd.Series(pd.to_datetime(test_timestamps))
actual_series = pd.Series(test_actual_latency, index=test_ts_series)
pred_series = pd.Series(pred_latency, index=test_ts_series)

lead_time_results = []
for _, fw in failures.iterrows():
    incident_start = pd.Timestamp('2026-04-01') + pd.Timedelta(days=float(fw['start_day']))
    incident_end = incident_start + pd.Timedelta(hours=float(fw['duration_hours']))

    if incident_start < test_ts_series.min() or incident_start > test_ts_series.max():
        continue

    # Search window: from 60 min before official start, through the end
    # of the incident (so we capture both the crossing point of the real
    # spike and any earlier crossing point in the forecast).
    window_start = incident_start - pd.Timedelta(minutes=60)
    window_end = min(incident_end, test_ts_series.max())

    actual_window = actual_series[(actual_series.index >= window_start) & (actual_series.index <= window_end)]
    pred_window = pred_series[(pred_series.index >= window_start) & (pred_series.index <= window_end)]

    actual_crossings = actual_window[actual_window > ALERT_THRESHOLD]
    pred_crossings = pred_window[pred_window > ALERT_THRESHOLD]

    if len(actual_crossings) == 0:
        print(f"  {fw['name']:25s}: actual latency never crossed threshold in this window — skipping")
        continue

    actual_hit_time = actual_crossings.index.min()

    if len(pred_crossings) == 0:
        print(f"  {fw['name']:25s} @ {incident_start}: NO WARNING (model never predicted a spike)")
        lead_time_results.append({'incident': fw['name'], 'warned_early': False, 'lead_time_minutes': None})
        continue

    pred_hit_time = pred_crossings.index.min()
    lead_minutes = (actual_hit_time - pred_hit_time).total_seconds() / 60

    if lead_minutes > 0:
        print(f"  {fw['name']:25s} @ {incident_start}: WARNED {lead_minutes:.0f} min BEFORE actual latency crossed threshold")
        lead_time_results.append({'incident': fw['name'], 'warned_early': True, 'lead_time_minutes': lead_minutes})
    else:
        print(f"  {fw['name']:25s} @ {incident_start}: caught it {abs(lead_minutes):.0f} min AFTER actual crossing (reactive, not predictive)")
        lead_time_results.append({'incident': fw['name'], 'warned_early': False, 'lead_time_minutes': lead_minutes})

if lead_time_results:
    detected = sum(r['warned_early'] for r in lead_time_results)
    avg_lead = np.mean([r['lead_time_minutes'] for r in lead_time_results if r['warned_early']]) if detected else 0
    print(f"\nGenuine early warning on {detected}/{len(lead_time_results)} in-test-window incidents"
          + (f", avg lead time {avg_lead:.0f} min" if detected else ""))

# ----------------------------------------------------------------------
# 10. PLOTS
# ----------------------------------------------------------------------
plt.figure(figsize=(15, 5))
plt.plot(test_timestamps, test_actual_latency, label='Actual p99 latency', alpha=0.8, linewidth=1)
plt.plot(test_timestamps, pred_latency, label='Predicted p99 latency (30-min ahead)', alpha=0.8, linewidth=1)
plt.axhline(ALERT_THRESHOLD, color='red', linestyle='--', label=f'Alert threshold ({ALERT_THRESHOLD}ms)', alpha=0.6)
plt.title(f'Predicted vs Actual p99 Latency — Test Set (MAE={mae:.1f}ms, RMSE={rmse:.1f}ms)', fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Latency (ms)')
plt.legend()
plt.tight_layout()
plt.savefig('final_forecast_vs_actual_v4.png', dpi=120)
plt.close()

plt.figure(figsize=(6, 5))
cm = confusion_matrix(actual_spike, predicted_spike)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Spike'],
            yticklabels=['Actual Normal', 'Actual Spike'])
plt.title(f'Alert Confusion Matrix (threshold={ALERT_THRESHOLD}ms)', fontweight='bold')
plt.tight_layout()
plt.savefig('final_confusion_matrix_v4.png', dpi=120)
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training History', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MSE (scaled log-latency)')
plt.legend()
plt.tight_layout()
plt.savefig('final_training_history_v4.png', dpi=120)
plt.close()

print("\nSaved: final_forecast_vs_actual_v4.png, final_confusion_matrix_v4.png, final_training_history_v4.png")

# ----------------------------------------------------------------------
# 11. SAVE MODEL + SCALERS
# ----------------------------------------------------------------------
model.save('latency_forecast_model_v4.keras')
joblib.dump(x_scaler, 'x_scaler_v4.pkl')
joblib.dump(y_scaler, 'y_scaler_v4.pkl')
print("Saved model + scalers for reuse.")