import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

def generate_ar1_noise(length, rho=0.8, sigma=1.0):
    noise = np.zeros(length)
    noise[0] = np.random.normal(0, sigma)
    for i in range(1, length):
        noise[i] = rho * noise[i-1] + np.random.normal(0, sigma)
    return noise



START_DATE = datetime(2026, 4, 1)
DAYS = 30
INTERVAL = 5  # minutes
TOTAL_POINTS = (DAYS * 24 * 60) // INTERVAL
MAX_EXPECTED_TRAFFIC = 800  # FIX 4: Fixed scalar for stable normalization

timestamps = [START_DATE + timedelta(minutes=i * INTERVAL) for i in range(TOTAL_POINTS)]



request_count = []
traffic_noise = generate_ar1_noise(TOTAL_POINTS, rho=0.85, sigma=8)

for i, ts in enumerate(timestamps):
    hour = ts.hour
    if 8 <= hour < 12:
        base = np.random.uniform(350, 400)
    elif 12 <= hour < 18:
        base = np.random.uniform(450, 550)
    elif 18 <= hour < 22:
        base = np.random.uniform(300, 350)
    else:
        base = np.random.uniform(100, 150)

    if ts.weekday() >= 5:
        base *= 0.7
        
    request_count.append(base + traffic_noise[i])

request_count = np.clip(np.array(request_count), 10, None)



#INJECT CAUSE: Traffic Spike (Day 12)
start_traffic = 12 * 24 * 12
end_traffic = start_traffic + 12 * 12
for i in range(start_traffic, end_traffic):
    multiplier = 1 + 0.8 * np.sin(np.pi * (i - start_traffic) / (end_traffic - start_traffic))
    request_count[i] *= multiplier



active_users = request_count * 0.45 + generate_ar1_noise(TOTAL_POINTS, rho=0.9, sigma=2)
active_users = np.clip(active_users, 5, None)


memory_usage = (
    30 
    + (request_count / MAX_EXPECTED_TRAFFIC) * 45 
    + generate_ar1_noise(TOTAL_POINTS, rho=0.8, sigma=1)
)


cpu_usage = (
    20 
    + (request_count / MAX_EXPECTED_TRAFFIC) * 55 
    + generate_ar1_noise(TOTAL_POINTS, rho=0.7, sigma=1.5)
)


#INJECT CAUSE: Memory Leak (Day 6 to Day 8)

start_mem = 6 * 24 * 12
end_mem = 8 * 24 * 12
for i in range(start_mem, end_mem):
    progress = (i - start_mem) / (end_mem - start_mem)
    memory_usage[i] += progress * 35  


    cpu_usage[i] += progress * 18 

cpu_usage = np.clip(cpu_usage, 5, 99.9)
memory_usage = np.clip(memory_usage, 10, 99.9)



db_query_time = (
    8
    + (cpu_usage / 100) * 15
    + (memory_usage / 100) * 20
    + generate_ar1_noise(TOTAL_POINTS, rho=0.8, sigma=1)
)


#INJECT CAUSE: Database Slowdown (Day 18)
start_db = 18 * 24 * 12
end_db = start_db + 24 * 12
# Simulate a locked table or bad query plan
db_query_time[start_db:end_db] *= 2.5 


#Calculate Latency
p50_latency = []
latency_noise = generate_ar1_noise(TOTAL_POINTS, rho=0.6, sigma=3)

for i, ts in enumerate(timestamps):
    hour_fraction = (ts.hour + ts.minute / 60) / 24
    daily_cycle = 12 * np.sin(2 * np.pi * hour_fraction)

    
    latency = (
        30
        + (request_count[i] / MAX_EXPECTED_TRAFFIC) * 50
        + daily_cycle
        + (cpu_usage[i] ** 1.2) * 0.15   
        + db_query_time[i] * 1.2
        + latency_noise[i]
    )
    p50_latency.append(latency)

p50_latency = np.array(p50_latency)


p95_latency = p50_latency * (1.5 + (cpu_usage / 100) * 1.2 + generate_ar1_noise(TOTAL_POINTS, rho=0.5, sigma=0.1))
p99_latency = p50_latency * (2.0 + (cpu_usage / 100) * 2.5 + generate_ar1_noise(TOTAL_POINTS, rho=0.5, sigma=0.2))


#INJECT LATENCY: Deployment Bug (Day 25)
start_bug = 25 * 24 * 12
end_bug = start_bug + 8 * 12

p50_latency[start_bug:end_bug] += np.random.normal(30, 5, end_bug - start_bug)
p95_latency[start_bug:end_bug] += np.random.normal(100, 15, end_bug - start_bug)
p99_latency[start_bug:end_bug] += np.random.normal(250, 30, end_bug - start_bug)



cpu_risk = np.maximum(0, (cpu_usage - 80) / 20)           # Risk builds above 80% CPU
db_risk = np.maximum(0, (db_query_time - 35) / 30)        # Risk builds if DB > 35ms
latency_risk = np.maximum(0, (p99_latency - 300) / 500)   # Risk builds if P99 > 300ms

error_noise = np.abs(generate_ar1_noise(TOTAL_POINTS, rho=0.7, sigma=0.002))

error_rate = (
    0.001 
    + (cpu_risk * 0.04) 
    + (db_risk * 0.05) 
    + (latency_risk * 0.06) 
    + error_noise
)
error_rate = np.clip(error_rate, 0.001, 0.20)


df = pd.DataFrame({
    "timestamp": timestamps,
    "request_count": np.round(request_count).astype(int),
    "active_users": np.round(active_users).astype(int),
    "cpu_usage": np.round(cpu_usage, 2),
    "memory_usage": np.round(memory_usage, 2),
    "db_query_time_ms": np.round(db_query_time, 2),
    "p50_latency_ms": np.round(p50_latency, 2),
    "p95_latency_ms": np.round(p95_latency, 2),
    "p99_latency_ms": np.round(p99_latency, 2),
    "error_rate": np.round(error_rate, 4),
})


df.to_csv("latency_data_production_grade.csv", index=False)

print("=" * 65)
print("Production-Grade Causal Dataset Generated Successfully")
print("=" * 65)
print(f"Rows            : {len(df):,}")
print(f"Duration        : {DAYS} days")
print(f"Interval        : {INTERVAL} minutes")
print(f"Output File     : latency_data_production_grade.csv")
print()
print(df.head())