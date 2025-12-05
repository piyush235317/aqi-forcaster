import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import yaml
from src.preprocessing import load_and_process

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def generate_forecast(model, last_available_data, days=7):
    """
    Generates future forecast using recursive strategy for Random Forest.
    """
    future_dates = pd.date_range(start=last_available_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days)
    forecasts = []
    
    # Get the last row of data
    current_data = last_available_data.iloc[-1].copy()
    
    # We need to reconstruct the feature set for the prediction.
    # The models are trained on [AQI_lag_1, ..., AQI_lag_7]
    
    last_known_aqi = current_data['AQI']
    
    # History: [AQI_T, AQI_T-1, AQI_T-2, ...]
    # We assume the dataframe has these lags populated correctly already.
    history = [last_known_aqi] + [current_data[f'AQI_lag_{i}'] for i in range(1, 8)]
    
    for date in future_dates:
        # Construct features for this new day
        features = {}
        
        # 1. Update lag features
        # AQI_lag_1 is history[0] (yesterday)
        # AQI_lag_2 is history[1] (day before yesterday)
        for i in range(1, 8):
            features[f'AQI_lag_{i}'] = history[i-1]
            
        # Create DataFrame for prediction (single row)
        X_new = pd.DataFrame([features])
        
        # Ensure we only pass the columns the model expects (lags)
        feature_cols = [f'AQI_lag_{i}' for i in range(1, 8)]
        X_new = X_new[feature_cols]
        
        # Predict
        pred_aqi = model.predict(X_new)[0]
        forecasts.append({'Date': date, 'Predicted AQI': pred_aqi, 'AQI Category': get_aqi_category(pred_aqi)})
        
        # Update history for next iteration
        history.insert(0, pred_aqi)
        history = history[:8]
        
    return pd.DataFrame(forecasts)

def main():
    print("Step 5: Generating Forecast with Best Model...")
    
    # Load Processed Data
    # We can reuse the load_and_process to get the config and full df
    _, _, config, full_df = load_and_process()
    
    # Check if full_df is loaded
    if full_df is None:
        print("Error: Could not load data.")
        return

    print("Loading best model...")
    model_path = os.path.join(config['MODEL_SAVE_PATH'], 'best_model.pkl')
    try:
        rf_model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: '{model_path}' not found. Please run 02_tune_models.py first.")
        return

    print(f"Generating {config['FORECAST_DAYS']}-Day forecast...")
    forecast_df = generate_forecast(rf_model, full_df, days=config['FORECAST_DAYS'])
    
    print("\nForecast Results:")
    print(forecast_df.to_string(index=False))
    
    # Plotting
    print("\nPlotting forecast...")
    plt.figure(figsize=(12, 6))
    
    # Last 30 days of actual data
    last_30_days = full_df.iloc[-30:]
    plt.plot(last_30_days['Date'], last_30_days['AQI'], label='Actual History', color='black')
    
    # Forecast
    last_date = last_30_days['Date'].iloc[-1]
    last_aqi = last_30_days['AQI'].iloc[-1]
    
    plot_dates = [last_date] + list(forecast_df['Date'])
    plot_aqi = [last_aqi] + list(forecast_df['Predicted AQI'])
    
    plt.plot(plot_dates, plot_aqi, label='Forecast (Best Model)', color='green', linestyle='--', marker='o')
    
    plt.title(f"AQI Forecast: Next {config['FORECAST_DAYS']} Days ({config['CITY_NAME']})")
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(config['RESULTS_PATH'], 'best_model_forecast.png')
    plt.savefig(save_path)
    print(f"Plot saved as '{save_path}'")

if __name__ == "__main__":
    main()
