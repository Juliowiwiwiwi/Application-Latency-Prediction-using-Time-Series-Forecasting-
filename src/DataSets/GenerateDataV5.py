"""
Production-Grade Latency Dataset Generator v4
==============================================
Same 150 days / 24 failures as v3, with ONE critical structural fix:

v3 PROBLEM (confirmed empirically): every failure's CPU/memory/DB ramp
started EXACTLY at its recorded start_day with zero buildup beforehand.
45 minutes before "Cascading Failure #4", CPU sat at a completely normal
31%. Then it jumped straight to elevated levels the instant the clock hit
start_day. That means there was LITERALLY NOTHING for any model to learn
as an early-warning signal — the input 30-45 minutes before an incident
was statistically identical to any random normal period. No architecture
change (LSTM, XGBoost, Transformer) can fix "there is no signal here".

v4 FIX: every failure now has a PRECURSOR_MINUTES (default 30) window
BEFORE its recorded start where CPU/memory/DB begin a gradual, genuine
climb — mimicking real incidents (resource exhaustion, thermal buildup,
connection pool saturation) where infrastructure metrics degrade before
user-facing latency does. The big, sharp latency-specific spike still
kicks in exactly at start_day (unchanged) — only the underlying resource
metrics now show real anticipatory drift.

This is what actually makes 30-minute-ahead early warning POSSIBLE.
Whether the model realizes that possibility is now a fair test of the
model/training setup, not a test of an unsolvable problem.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import percentileofscore

np.random.seed(42)

# ======================================================================
# CONFIGURATION
# ======================================================================
START_DATE = datetime(2026, 4, 1)
DAYS = 150
INTERVAL_MINUTES = 5
TOTAL_POINTS = (DAYS * 24 * 60) // INTERVAL_MINUTES
MAX_TRAFFIC = 800
PRECURSOR_MINUTES = 75  # <-- the key new parameter

print(f"Generating {DAYS} days of data ({TOTAL_POINTS} samples)")
print(f"Precursor window: {PRECURSOR_MINUTES} minutes before every failure\n")

# ======================================================================
# 1. TIMESTAMPS & BASE TRAFFIC
# ======================================================================
timestamps = [START_DATE + timedelta(minutes=i * INTERVAL_MINUTES) for i in range(TOTAL_POINTS)]


def generate_ar1_noise(length, rho=0.8, sigma=1.0):
    noise = np.zeros(length)
    noise[0] = np.random.normal(0, sigma)
    for i in range(1, length):
        noise[i] = rho * noise[i - 1] + np.random.normal(0, sigma)
    return noise


traffic_noise = generate_ar1_noise(TOTAL_POINTS, rho=0.85, sigma=8)
request_count = []

for i, ts in enumerate(timestamps):
    hour = ts.hour
    dow = ts.weekday()

    if 8 <= hour < 12:
        base = np.random.uniform(350, 400)
    elif 12 <= hour < 18:
        base = np.random.uniform(450, 550)
    elif 18 <= hour < 22:
        base = np.random.uniform(300, 350)
    else:
        base = np.random.uniform(100, 150)

    if dow >= 5:
        base *= 0.7

    request_count.append(base + traffic_noise[i])

request_count = np.clip(np.array(request_count), 10, None)

# ======================================================================
# 2. BASELINE METRICS
# ======================================================================
active_users = request_count * 0.45 + generate_ar1_noise(TOTAL_POINTS, rho=0.9, sigma=2)
active_users = np.clip(active_users, 5, None)

cpu_usage = (
    25
    + (request_count / MAX_TRAFFIC) * 45
    + generate_ar1_noise(TOTAL_POINTS, rho=0.7, sigma=1.5)
)
cpu_usage = np.clip(cpu_usage, 5, 75)

memory_usage = (
    35
    + (request_count / MAX_TRAFFIC) * 35
    + generate_ar1_noise(TOTAL_POINTS, rho=0.8, sigma=1)
)
memory_usage = np.clip(memory_usage, 15, 70)

db_query_time = (
    8
    + (cpu_usage / 100) * 12
    + (memory_usage / 100) * 8
    + generate_ar1_noise(TOTAL_POINTS, rho=0.8, sigma=0.5)
)
db_query_time = np.clip(db_query_time, 5, 40)

# ======================================================================
# 3. INJECT FAILURE PATTERNS (with precursor buildup)
# ======================================================================
failure_windows = []
precursor_steps = int(PRECURSOR_MINUTES / INTERVAL_MINUTES)


def inject_failure(name, start_day, duration_hours, cpu_mult=1.0, mem_delta=0,
                    db_mult=1.0, impact_on_latency=1.0):
    start_idx = int(start_day * 24 * 60 / INTERVAL_MINUTES)
    end_idx = int((start_day + duration_hours / 24) * 24 * 60 / INTERVAL_MINUTES)

    # Ramp now begins PRECURSOR_STEPS earlier than the recorded start_idx.
    # Using ONE continuous sin(pi*x)**0.5 curve stretched over this longer
    # span means: at the recorded start_idx, a meaningful chunk of the
    # ramp has already happened (real precursor signal) — with no
    # discontinuity, because it's a single smooth curve, not two pasted
    # pieces.
    ramp_start_idx = max(0, start_idx - precursor_steps)
    combined_duration = end_idx - ramp_start_idx

    for i in range(ramp_start_idx, min(end_idx, TOTAL_POINTS)):
        progress = (i - ramp_start_idx) / combined_duration
        ramp = np.sin(np.pi * progress) ** 0.5

        cpu_usage[i] = np.clip(cpu_usage[i] * (1 + (cpu_mult - 1) * ramp), 5, 99)
        memory_usage[i] = np.clip(memory_usage[i] + mem_delta * ramp, 15, 95)
        db_query_time[i] *= (1 + (db_mult - 1) * ramp)

    # NOTE: start_idx/end_idx recorded here are UNCHANGED from v3 — this is
    # still "when the big user-facing latency spike officially happens."
    # The precursor is invisible in this bookkeeping; it only lives in the
    # raw cpu/memory/db columns, which is exactly where it should live.
    failure_windows.append({
        'name': name, 'start_day': start_day, 'duration_hours': duration_hours,
        'start_idx': start_idx, 'end_idx': end_idx, 'latency_impact': impact_on_latency,
        'precursor_minutes': PRECURSOR_MINUTES
    })
    print(f"  {name:25s} @ day {start_day:3.0f} ({duration_hours:4.1f}h)  "
          f"CPU×{cpu_mult:.2f} Mem+{mem_delta:+3.0f}% DB×{db_mult:.2f} Latency×{impact_on_latency:.2f}")


def inject_traffic_flood(name, start_day, duration_hours, req_mult, cpu_mult, mem_delta, impact_on_latency):
    start_idx = int(start_day * 24 * 60 / INTERVAL_MINUTES)
    end_idx = int((start_day + duration_hours / 24) * 24 * 60 / INTERVAL_MINUTES)

    ramp_start_idx = max(0, start_idx - precursor_steps)
    combined_duration = end_idx - ramp_start_idx

    for i in range(ramp_start_idx, min(end_idx, TOTAL_POINTS)):
        progress = (i - ramp_start_idx) / combined_duration
        ramp = np.sin(np.pi * progress) ** 0.5
        request_count[i] *= (1 + (req_mult - 1) * ramp)
        cpu_usage[i] = np.clip(cpu_usage[i] * (1 + (cpu_mult - 1) * ramp), 5, 99)
        memory_usage[i] = np.clip(memory_usage[i] + mem_delta * ramp, 15, 95)

    failure_windows.append({
        'name': name, 'start_day': start_day, 'duration_hours': duration_hours,
        'start_idx': start_idx, 'end_idx': end_idx, 'latency_impact': impact_on_latency,
        'precursor_minutes': PRECURSOR_MINUTES
    })
    print(f"  {name:25s} @ day {start_day:3.0f} ({duration_hours:4.1f}h)  Req×{req_mult:.2f} Latency×{impact_on_latency:.2f}")


print("Injecting failure patterns (4 instances per type, each with precursor buildup):")

inject_failure("Memory Leak #1", 5, 18, cpu_mult=1.20, mem_delta=28, db_mult=1.30, impact_on_latency=0.80)
inject_failure("Memory Leak #2", 41, 15, cpu_mult=1.15, mem_delta=25, db_mult=1.25, impact_on_latency=0.75)
inject_failure("Memory Leak #3", 77, 20, cpu_mult=1.25, mem_delta=32, db_mult=1.35, impact_on_latency=0.90)
inject_failure("Memory Leak #4", 113, 16, cpu_mult=1.18, mem_delta=27, db_mult=1.28, impact_on_latency=0.78)

inject_failure("CPU Spike #1", 12, 6, cpu_mult=2.20, mem_delta=10, db_mult=1.80, impact_on_latency=1.80)
inject_failure("CPU Spike #2", 48, 5, cpu_mult=2.00, mem_delta=8, db_mult=1.60, impact_on_latency=1.60)
inject_failure("CPU Spike #3", 84, 7, cpu_mult=2.40, mem_delta=12, db_mult=1.90, impact_on_latency=2.00)
inject_failure("CPU Spike #4", 120, 6, cpu_mult=2.10, mem_delta=9, db_mult=1.70, impact_on_latency=1.70)

inject_failure("Database Lock #1", 19, 12, cpu_mult=1.40, mem_delta=15, db_mult=3.00, impact_on_latency=1.50)
inject_failure("Database Lock #2", 55, 10, cpu_mult=1.30, mem_delta=12, db_mult=2.80, impact_on_latency=1.40)
inject_failure("Database Lock #3", 91, 14, cpu_mult=1.45, mem_delta=17, db_mult=3.20, impact_on_latency=1.65)
inject_failure("Database Lock #4", 127, 11, cpu_mult=1.35, mem_delta=13, db_mult=2.90, impact_on_latency=1.45)

inject_traffic_flood("Request Flood #1", 26, 8, req_mult=1.80, cpu_mult=1.50, mem_delta=10, impact_on_latency=0.60)
inject_traffic_flood("Request Flood #2", 62, 7, req_mult=1.70, cpu_mult=1.45, mem_delta=8, impact_on_latency=0.55)
inject_traffic_flood("Request Flood #3", 98, 9, req_mult=1.95, cpu_mult=1.60, mem_delta=12, impact_on_latency=0.70)
inject_traffic_flood("Request Flood #4", 134, 8, req_mult=1.75, cpu_mult=1.48, mem_delta=9, impact_on_latency=0.58)

inject_failure("Cascading Failure #1", 33, 9, cpu_mult=1.80, mem_delta=35, db_mult=2.50, impact_on_latency=2.00)
inject_failure("Cascading Failure #2", 69, 8, cpu_mult=1.75, mem_delta=32, db_mult=2.40, impact_on_latency=1.90)
inject_failure("Cascading Failure #3", 105, 10, cpu_mult=1.95, mem_delta=40, db_mult=2.70, impact_on_latency=2.30)
inject_failure("Cascading Failure #4", 141, 9, cpu_mult=1.85, mem_delta=37, db_mult=2.55, impact_on_latency=2.10)

inject_failure("Thermal Throttle #1", 44, 11, cpu_mult=1.30, mem_delta=5, db_mult=1.40, impact_on_latency=0.85)
inject_failure("Thermal Throttle #2", 80, 10, cpu_mult=1.25, mem_delta=4, db_mult=1.35, impact_on_latency=0.80)
inject_failure("Thermal Throttle #3", 116, 12, cpu_mult=1.35, mem_delta=6, db_mult=1.45, impact_on_latency=0.92)
inject_failure("Thermal Throttle #4", 148, 10, cpu_mult=1.28, mem_delta=5, db_mult=1.38, impact_on_latency=0.82)

request_count = np.clip(request_count, 10, None)
cpu_usage = np.clip(cpu_usage, 5, 99.9)
memory_usage = np.clip(memory_usage, 10, 99.9)
db_query_time = np.clip(db_query_time, 5, 80)

# ======================================================================
# 4. CALCULATE LATENCY
#    The big failure-specific latency bump still starts EXACTLY at
#    start_idx (unchanged) — only the underlying resource metrics
#    (cpu/memory/db, above) now show precursor drift beforehand.
# ======================================================================
p50_latency = []
latency_noise = generate_ar1_noise(TOTAL_POINTS, rho=0.6, sigma=2)

for i, ts in enumerate(timestamps):
    hour_fraction = (ts.hour + ts.minute / 60) / 24
    daily_cycle = 8 * np.sin(2 * np.pi * hour_fraction)

    latency = (
        35
        + (request_count[i] / MAX_TRAFFIC) * 40
        + daily_cycle
        + (cpu_usage[i] ** 1.1) * 0.18
        + db_query_time[i] * 1.5
        + latency_noise[i]
    )

    for fw in failure_windows:
        if fw['start_idx'] <= i < fw['end_idx']:
            progress = (i - fw['start_idx']) / (fw['end_idx'] - fw['start_idx'])
            impact = np.sin(np.pi * progress) ** 0.5 * fw['latency_impact']
            latency += 150 * impact

    p50_latency.append(latency)

p50_latency = np.array(p50_latency)
p50_latency = np.clip(p50_latency, 20, 900)

p95_latency = p50_latency * (1.4 + (cpu_usage / 100) * 1.0 + generate_ar1_noise(TOTAL_POINTS, rho=0.5, sigma=0.1))
p99_latency = p50_latency * (1.8 + (cpu_usage / 100) * 2.0 + generate_ar1_noise(TOTAL_POINTS, rho=0.5, sigma=0.2))

# ======================================================================
# 5. ERROR RATE
# ======================================================================
cpu_risk = np.maximum(0, (cpu_usage - 75) / 25)
db_risk = np.maximum(0, (db_query_time - 30) / 30)
latency_risk = np.maximum(0, (p99_latency - 300) / 400)

error_noise = np.abs(generate_ar1_noise(TOTAL_POINTS, rho=0.7, sigma=0.001))
error_rate = (
    0.0008
    + (cpu_risk * 0.035)
    + (db_risk * 0.04)
    + (latency_risk * 0.05)
    + error_noise
)
error_rate = np.clip(error_rate, 0.0005, 0.15)

# ======================================================================
# 6. FEATURE ENGINEERING
# ======================================================================
cpu_change_5m = np.gradient(cpu_usage, edge_order=2)
memory_change_5m = np.gradient(memory_usage, edge_order=2)
db_change_5m = np.gradient(db_query_time, edge_order=2)

cpu_rolling_mean = pd.Series(cpu_usage).rolling(window=24).mean().values
cpu_rolling_std = pd.Series(cpu_usage).rolling(window=24).std().values
memory_rolling_mean = pd.Series(memory_usage).rolling(window=24).mean().values
memory_rolling_std = pd.Series(memory_usage).rolling(window=24).std().values

cpu_rolling_mean = np.nan_to_num(cpu_rolling_mean, nan=np.mean(cpu_usage))
cpu_rolling_std = np.nan_to_num(cpu_rolling_std, nan=np.std(cpu_usage))
memory_rolling_mean = np.nan_to_num(memory_rolling_mean, nan=np.mean(memory_usage))
memory_rolling_std = np.nan_to_num(memory_rolling_std, nan=np.std(memory_usage))

cpu_percentile = np.array([percentileofscore(cpu_usage, x) for x in cpu_usage])
memory_percentile = np.array([percentileofscore(memory_usage, x) for x in memory_usage])

cpu_memory_interaction = (cpu_usage / 100) * (memory_usage / 100)
cpu_load_interaction = (cpu_usage / 100) * (request_count / MAX_TRAFFIC)

# ======================================================================
# 7. DATAFRAME
# ======================================================================
df = pd.DataFrame({
    'timestamp': timestamps,
    'request_count': np.round(request_count).astype(int),
    'active_users': np.round(active_users).astype(int),
    'cpu_usage': np.round(cpu_usage, 2),
    'memory_usage': np.round(memory_usage, 2),
    'db_query_time_ms': np.round(db_query_time, 2),
    'p50_latency_ms': np.round(p50_latency, 2),
    'p95_latency_ms': np.round(p95_latency, 2),
    'p99_latency_ms': np.round(p99_latency, 2),
    'error_rate': np.round(error_rate, 5),
    'cpu_change_5m': np.round(cpu_change_5m, 3),
    'memory_change_5m': np.round(memory_change_5m, 3),
    'db_change_5m': np.round(db_change_5m, 3),
    'cpu_2h_mean': np.round(cpu_rolling_mean, 2),
    'cpu_2h_std': np.round(cpu_rolling_std, 2),
    'memory_2h_mean': np.round(memory_rolling_mean, 2),
    'memory_2h_std': np.round(memory_rolling_std, 2),
    'cpu_percentile': np.round(cpu_percentile, 1),
    'memory_percentile': np.round(memory_percentile, 1),
    'cpu_memory_stress': np.round(cpu_memory_interaction, 4),
    'cpu_load_stress': np.round(cpu_load_interaction, 4),
})

df.to_csv('latency_data_v5.csv', index=False)

failure_df = pd.DataFrame(failure_windows)
failure_df.to_csv('failure_windows_v5.csv', index=False)

print("\n" + "=" * 70)
print(f"✓ Generated latency_data_v5.csv ({len(df)} samples)")
print(f"✓ Generated failure_windows_v5.csv ({len(failure_windows)} incidents)")
print("=" * 70)
print(f"p99_latency_ms: mean={df['p99_latency_ms'].mean():.1f}ms, "
      f"max={df['p99_latency_ms'].max():.1f}ms, min={df['p99_latency_ms'].min():.1f}ms")

n = len(df)
train_end = int(n * 0.65)
val_end = int(n * 0.80)
train_cut = df['timestamp'].iloc[train_end - 1]
val_cut = df['timestamp'].iloc[val_end - 1]

train_n = val_n = test_n = 0
for fw in failure_windows:
    start = START_DATE + timedelta(days=fw['start_day'])
    if start <= train_cut:
        train_n += 1
    elif start <= val_cut:
        val_n += 1
    else:
        test_n += 1
print(f"\nSplit coverage -> TRAIN: {train_n}  VAL: {val_n}  TEST: {test_n}  (total {len(failure_windows)})")

# Prove the precursor now exists (same check as before, should look different now)
sample_fw = failure_df[failure_df['name'] == 'Cascading Failure #4'].iloc[0]
start = START_DATE + timedelta(days=float(sample_fw['start_day']))
before = df[(df['timestamp'] >= start - timedelta(minutes=45)) & (df['timestamp'] < start)]
print(f"\nSanity check — 45 min before 'Cascading Failure #4' (should now show a climb, not flat):")
print(f"  CPU:    {before['cpu_usage'].min():.1f} -> {before['cpu_usage'].max():.1f}")
print(f"  Memory: {before['memory_usage'].min():.1f} -> {before['memory_usage'].max():.1f}")