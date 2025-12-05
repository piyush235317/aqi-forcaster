import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import load_and_process
from src.models import AQIRandomForest, AQIProphet, AQILSTM

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
    
    current_data = last_available_data.iloc[-1].copy()
    last_known_aqi = current_data['AQI']
    
    # History: [AQI_T, AQI_T-1, AQI_T-2, ...]
    history = [last_known_aqi] + [current_data[f'AQI_lag_{i}'] for i in range(1, 8)]
    
    for date in future_dates:
        features = {}
        # Persistence for non-lag features
        for col in last_available_data.columns:
            if 'lag' not in col and col not in ['Date', 'AQI', 'City', 'AQI_Bucket']:
                features[col] = current_data[col]
        
        # Update lag features
        for i in range(1, 8):
            features[f'AQI_lag_{i}'] = history[i-1]
            
        X_new = pd.DataFrame([features])
        pred_aqi = model.predict(X_new)[0]
        forecasts.append({'Date': date, 'Predicted AQI': pred_aqi, 'AQI Category': get_aqi_category(pred_aqi)})
        
        history.insert(0, pred_aqi)
        history = history[:8]
        
    return pd.DataFrame(forecasts)

def main():
    # 1. Load and Process
    print("Loading and processing data...")
    train, test, config, full_df = load_and_process()
    
    if train is None:
        return

    # 2. Train Models
    print("\nTraining Models...")
    
    # Random Forest
    print("Training Random Forest...")
    rf = AQIRandomForest()
    rf.train(train, train['AQI'])
    
    # Prophet
    print("Training Prophet...")
    prophet = AQIProphet()
    prophet.train(train)
    
    # LSTM
    print("Training LSTM...")
    lstm = AQILSTM(epochs=10) # Reduced epochs for speed in demo
    lstm.train(train['AQI'].values)
    
    # 3. Save Models
    print(f"\nSaving models to {config['MODEL_SAVE_PATH']}...")
    rf.save_model(config['MODEL_SAVE_PATH'])
    prophet.save_model(config['MODEL_SAVE_PATH'])
    lstm.save_model(config['MODEL_SAVE_PATH'])
    
    # 4. Forecast (Using Random Forest as Best Model)
    print(f"\nGenerating {config['FORECAST_DAYS']}-Day Forecast (Random Forest)...")
    forecast_df = generate_forecast(rf, full_df, days=config['FORECAST_DAYS'])
    
    print("\nForecast Results:")
    print(forecast_df.to_string(index=False))
    
    # Plotting
    print("\nPlotting forecast...")
    plt.figure(figsize=(12, 6))
    last_30_days = full_df.iloc[-30:]
    plt.plot(last_30_days['Date'], last_30_days['AQI'], label='Actual History', color='black')
    
    last_date = last_30_days['Date'].iloc[-1]
    last_aqi = last_30_days['AQI'].iloc[-1]
    plot_dates = [last_date] + list(forecast_df['Date'])
    plot_aqi = [last_aqi] + list(forecast_df['Predicted AQI'])
    
    plt.plot(plot_dates, plot_aqi, label='Forecast', color='red', linestyle='--', marker='o')
    
    plt.title(f"AQI Forecast: Next {config['FORECAST_DAYS']} Days ({config['CITY_NAME']})")
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/final_forecast.png')
    print("Plot saved as 'results/final_forecast.png'")


if __name__ == "__main__":
    main()
