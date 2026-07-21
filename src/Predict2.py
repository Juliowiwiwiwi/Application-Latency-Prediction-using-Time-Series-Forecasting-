import sys
import pathlib

# ==========================================
# 0. LINUX TO WINDOWS POSIXPATH FIX
# ==========================================
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# ==========================================
# 1. PATH RESOLUTION & MODEL LOADING
# ==========================================
SRC_DIR = Path(__file__).resolve().parent

MODEL_DIR = SRC_DIR / "autogluon_model"
DATA_FILE = SRC_DIR / "DataSets" / "latency_data_v2.csv"

print(f"Loading pre-trained AutoGluon model from: {MODEL_DIR}")
predictor = TimeSeriesPredictor.load(str(MODEL_DIR))

# ==========================================
# 2. LOAD TELEMETRY DATASET
# ==========================================
df = pd.read_csv(DATA_FILE)

if "item_id" not in df.columns:
    df["item_id"] = "server_1"

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ==========================================
# 3. HISTORICAL SPIKE WINDOW CONFIGURATION
# ==========================================
# Cutoff set right before the major SLA breach escalation on April 9th
CUTOFF_TIME = pd.to_datetime("2026-04-09 04:10:00")
PREDICT_STEPS = 12  # 1-hour prediction horizon
SLA_THRESHOLD = 320.0

# Extract 200 historical timesteps up to CUTOFF_TIME as model input
historical_df = df[df["timestamp"] <= CUTOFF_TIME].tail(200)

ts_context = TimeSeriesDataFrame.from_data_frame(
    historical_df,
    id_column="item_id",
    timestamp_column="timestamp"
)

# ==========================================
# 4. GENERATE FORECAST FOR SPIKE WINDOW
# ==========================================
print(f"Generating backtest forecast starting from {CUTOFF_TIME}...")
forecasts = predictor.predict(ts_context)

first_item = forecasts.item_ids[0]
forecast_df = forecasts.loc[first_item].reset_index()

# Get ground truth actuals (historical context + forecast horizon)
window_end_time = forecast_df["timestamp"].max()
actuals_df = df[
    (df["timestamp"] >= historical_df["timestamp"].iloc[-40]) &
    (df["timestamp"] <= window_end_time)
]

# ==========================================
# 5. ACCURACY & STATE-TRANSITION ALERTING
# ==========================================
eval_df = pd.merge(forecast_df, df, on="timestamp", how="inner")
mae = np.mean(np.abs(eval_df["p95_latency_ms"] - eval_df["mean"]))

predicted_means = forecast_df["mean"].values
forecast_times = forecast_df["timestamp"]

# Identify initial SLA breach transition
is_above = predicted_means > SLA_THRESHOLD
was_below = np.roll(is_above, 1)
was_below[0] = False
alert_triggers = is_above & (~was_below)

print(f"\n=== Spike Window Evaluation Summary ===")
print(f"Mean Absolute Error (MAE): {mae:.2f} ms")
if alert_triggers.any():
    print(f"[ALERT TRIGGERED] SLA Breach predicted! Peak forecast: {predicted_means.max():.2f} ms")
else:
    print("[NO ALERT] Model predicted latency stays below SLA threshold.")

# ==========================================
# 6. VISUALIZATION OVERLAY
# ==========================================
plt.figure(figsize=(12, 6))

# Plot actual telemetry ground truth
plt.plot(
    actuals_df["timestamp"],
    actuals_df["p95_latency_ms"],
    color="#9c27b0",
    linewidth=2.2,
    label="Actual p95 Latency (Ground Truth)",
)

# Plot model prediction line
plt.plot(
    forecast_times,
    predicted_means,
    color="#0288d1",
    linestyle="--",
    linewidth=2.5,
    label="AI Forecast (Predicted)",
)

# 10% - 90% Prediction Interval
plt.fill_between(
    forecast_times,
    forecast_df["0.1"],
    forecast_df["0.9"],
    color="#0288d1",
    alpha=0.2,
    label="10%-90% Prediction Interval",
)

# SLA Threshold Line
plt.axhline(
    y=SLA_THRESHOLD,
    color="red",
    linestyle="-",
    linewidth=1.5,
    label=f"SLA Threshold ({SLA_THRESHOLD:.0f}ms)",
)

# Triggered Alert Marker
if alert_triggers.any():
    plt.scatter(
        forecast_times[alert_triggers],
        predicted_means[alert_triggers],
        color="red",
        s=120,
        zorder=5,
        label="Predicted Initial SLA Breach Alert",
    )

plt.title(
    f"SLA Breach Spike Backtest (April 9, 2026) | MAE: {mae:.2f} ms",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Time")
plt.ylabel("p95 Latency (ms)")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()