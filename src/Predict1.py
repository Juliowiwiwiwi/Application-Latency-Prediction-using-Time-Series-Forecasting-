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

ts_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)

# ==========================================
# 3. BACKTEST: HOLD OUT LAST 12 TIMESTEPS
# ==========================================
PREDICT_STEPS = 12  # 1-hour forecast horizon (12 steps of 5-min intervals)

# Historical context: Cut off the final 12 timesteps so the model hasn't seen them
context_data = ts_data.slice_by_timestep(-200 - PREDICT_STEPS, -PREDICT_STEPS)

print(f"Generating backtest forecast for the held-out {PREDICT_STEPS} timesteps...")
forecasts = predictor.predict(context_data)

# Extract actuals for comparison (last 60 timesteps to show context + test window)
first_item = forecasts.item_ids[0]
recent_actuals = ts_data.slice_by_timestep(-60, None).loc[first_item].reset_index()
forecast_df = forecasts.loc[first_item].reset_index()

# ==========================================
# 4. CALCULATE ACCURACY METRICS
# ==========================================
# Align forecast with actual ground truth on timestamp
eval_df = pd.merge(forecast_df, recent_actuals, on="timestamp", how="inner")

mae = np.mean(np.abs(eval_df["p95_latency_ms"] - eval_df["mean"]))
print(f"\n=== Evaluation Metrics ===")
print(f"Mean Absolute Error (MAE): {mae:.2f} ms")

# ==========================================
# 5. OVERLAY PLOT (GROUND TRUTH VS FORECAST)
# ==========================================
plt.figure(figsize=(12, 6))

# Plot actual historical latency (including held-out period)
plt.plot(
    recent_actuals["timestamp"],
    recent_actuals["p95_latency_ms"],
    color="#9c27b0",
    linewidth=2,
    label="Actual p95 Latency (Ground Truth)",
)

# Overlay predicted forecast line directly on top of the held-out test window
plt.plot(
    forecast_df["timestamp"],
    forecast_df["mean"],
    color="#0288d1",
    linestyle="--",
    linewidth=2.5,
    label="AI Forecast (Predicted)",
)

# Prediction interval (10% - 90%)
plt.fill_between(
    forecast_df["timestamp"],
    forecast_df["0.1"],
    forecast_df["0.9"],
    color="#0288d1",
    alpha=0.2,
    label="10%-90% Prediction Interval",
)

plt.title(
    f"Model Accuracy Evaluation (Out-of-Sample Backtest) | MAE: {mae:.2f} ms",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Time")
plt.ylabel("p95 Latency (ms)")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()