
# Gave just a straight line. Not of much use. Need to try a better model

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

def run_baseline_model(csv_path):
    print("Loading dataset")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    #Train/Test Split
    print("Splitting data into training and testing sets...")
    train_data = df.loc['2026-04-01':'2026-04-20', 'p99_latency_ms']
    test_data = df.loc['2026-04-21':'2026-04-22', 'p99_latency_ms']

    print("Training Baseline ARIMA")
    # order=(2,1,2) im guessing and taking parametes here, will implement grid search later to find the best parameters
    model = ARIMA(train_data, order=(2,1,2)) 
    model_fit = model.fit()

    print("Forecasting the next 48 hours blindly")
    forecast = model_fit.forecast(steps=len(test_data))

    #Calculate Evaluation Metric
    mae = mean_absolute_error(test_data, forecast)
    print(f"Baseline Mean Absolute Error (MAE): {mae:.2f} ms")

    #Plot the Results
    plt.figure(figsize=(15, 6))
    plt.plot(test_data.index, test_data, label='Actual p99 Latency', color='purple', alpha=0.7)
    plt.plot(test_data.index, forecast, label='ARIMA Forecast', color='red', linestyle='--', linewidth=2)
    plt.title(f'Baseline Model: ARIMA vs Actual Latency (MAE: {mae:.2f} ms)', fontsize=14)
    plt.ylabel('Milliseconds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    print("Opening forecast plot")
    plt.show()

if __name__ == "__main__":
    run_baseline_model('DataSets/latency_data_production_grade.csv')