import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from data_processor import load_data, filter_by_city, handle_missing_values, create_lag_features
from models.random_forest import AQIRandomForest

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
    
    # We need to maintain the lag features. 
    # AQI_lag_1 is yesterday's AQI.
    # When predicting day T+1, AQI_lag_1 is AQI at T.
    # When predicting day T+2, AQI_lag_1 is prediction at T+1.
    
    # Initialize lags from the last known data
    # We need a way to shift lags. 
    # current_data has AQI_lag_1, AQI_lag_2, ...
    
    # Let's create a list of recent AQI values to easily update lags
    # We need the last 7 days of AQI to populate lags 1-7
    # Assuming the input dataframe has correct lags, we can reconstruct the history
    
    # Actually, simpler:
    # AQI_lag_1 is the most recent value.
    # AQI_lag_2 is the one before that.
    
    recent_aqi = []
    for i in range(1, 8):
        recent_aqi.append(current_data[f'AQI_lag_{i}'])
    
    # recent_aqi[0] is lag_1 (T-1), recent_aqi[1] is lag_2 (T-2)...
    # Wait, if current_data is at time T, then AQI_lag_1 is T-1.
    # But we want to predict T+1.
    # The features for T+1 should be:
    # AQI_lag_1 = AQI at T (which is current_data['AQI'])
    # AQI_lag_2 = AQI at T-1 (which is current_data['AQI_lag_1'])
    
    # So we need the actual AQI of the last row to start the recursion
    last_known_aqi = current_data['AQI']
    
    # History: [AQI_T, AQI_T-1, AQI_T-2, ...]
    history = [last_known_aqi] + [current_data[f'AQI_lag_{i}'] for i in range(1, 8)]
    
    for date in future_dates:
        # Construct features for this new day
        features = {}
        
        # 1. Weather features (Persistence: use last known values)
        # Identify non-lag, non-date, non-AQI columns
        for col in last_available_data.columns:
            if 'lag' not in col and col not in ['Date', 'AQI', 'City', 'AQI_Bucket']:
                features[col] = current_data[col] # Persistence
        
        # 2. Lag features
        # AQI_lag_1 is history[0] (yesterday)
        # AQI_lag_2 is history[1] (day before yesterday)
        for i in range(1, 8):
            features[f'AQI_lag_{i}'] = history[i-1]
            
        # Create DataFrame for prediction (single row)
        X_new = pd.DataFrame([features])
        
        # Ensure column order matches training
        # The model object (AQIRandomForest) has feature_cols, but that only includes lags.
        # Wait, our AQIRandomForest.train ONLY uses lag features.
        # "self.feature_cols = [f'AQI_lag_{i}' for i in range(1, 8)]"
        # So we ONLY need to update lags! Weather data is ignored by our specific RF implementation.
        # That simplifies things greatly.
        
        # Predict
        pred_aqi = model.predict(X_new)[0]
        forecasts.append({'Date': date, 'Predicted AQI': pred_aqi, 'AQI Category': get_aqi_category(pred_aqi)})
        
        # Update history for next iteration
        # Newest value becomes history[0]
        history.insert(0, pred_aqi)
        # Keep only needed history length
        history = history[:8]
        
    return pd.DataFrame(forecasts)

def main():
    print("Loading data...")
    df = load_data()
    df = filter_by_city(df)
    df = handle_missing_values(df)
    df = create_lag_features(df)
    
    print("Loading best model...")
    try:
        rf_model = joblib.load('saved_models/best_model.pkl')
    except FileNotFoundError:
        print("Error: 'saved_models/best_model.pkl' not found. Please run 02_tune_models.py first.")
        return

    print("Generating forecast...")
    forecast_df = generate_forecast(rf_model, df, days=7)
    
    print("\n7-Day Forecast:")
    print(forecast_df.to_string(index=False))
    
    # Plotting
    print("\nPlotting forecast...")
    plt.figure(figsize=(12, 6))
    
    # Last 30 days of actual data
    last_30_days = df.iloc[-30:]
    plt.plot(last_30_days['Date'], last_30_days['AQI'], label='Actual History', color='black')
    
    # Forecast
    # Connect last actual point to first forecast point for continuity
    last_date = last_30_days['Date'].iloc[-1]
    last_aqi = last_30_days['AQI'].iloc[-1]
    
    plot_dates = [last_date] + list(forecast_df['Date'])
    plot_aqi = [last_aqi] + list(forecast_df['Predicted AQI'])
    
    plt.plot(plot_dates, plot_aqi, label='Forecast', color='red', linestyle='--', marker='o')
    
    plt.title('AQI Forecast: Next 7 Days (Delhi)')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('aqi_future_forecast.png')
    print("Plot saved as 'aqi_future_forecast.png'")

if __name__ == "__main__":
    main()
