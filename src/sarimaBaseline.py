import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import warnings


def run_sarima_model(csv_path):
    print("Loading dataset")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    #Train/Test Split
    # Test on the last 2 days 
    test_data = df.loc['2026-04-21':'2026-04-22', 'p99_latency_ms']
    
    # Trained on 5 days before the test set to save time
    train_data = df.loc['2026-04-16':'2026-04-20', 'p99_latency_ms']

    print(f"Training on {len(train_data)} points. Testing on {len(test_data)} points.")

    #Build and Train Seasonal ARIMA
    print("Training Seasonal ARIMA")
    
    # order=(1, 1, 1): im guessing and taking parametes here, will implement grid search later to find the best parameters
    # seasonal_order=(1, 0, 1, 288): im guessing and taking parametes here, will implement grid search later to find the best parameters The Seasonal part 288 = number of 5 min intervals in 24 hourss
    model = SARIMAX(
        train_data, 
        order=(1, 1, 1), 
        seasonal_order=(1, 0, 1, 288),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    model_fit = model.fit(disp=False)


    #Forecast and Evaluate
    print("3. Forecasting the next 48 hours blindly")
    forecast = model_fit.forecast(steps=len(test_data))

    mae = mean_absolute_error(test_data, forecast)
    
    print(f"SARIMA Mean Absolute Error (MAE): {mae:.2f} ms")


    #Plot the Results
    plt.figure(figsize=(15, 6))
    plt.plot(test_data.index, test_data, label='Actual p99 Latency', color='purple', alpha=0.7, linewidth=1.5)
    plt.plot(test_data.index, forecast, label='SARIMA Forecast', color='orange', linestyle='--', linewidth=2.5)
    
    plt.title(f'Seasonal Model: SARIMA vs Actual Latency (MAE: {mae:.2f} ms)', fontsize=14)
    plt.ylabel('Milliseconds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    run_sarima_model('DataSets/latency_data_production_grade.csv')