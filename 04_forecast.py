import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt
import os
from src.models import AQIRandomForest

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
        
        # Filter to only keep the features the model was trained on (lags)
        feature_cols = [f'AQI_lag_{i}' for i in range(1, 8)]
        X_new = X_new[feature_cols]
        
        pred_aqi = model.predict(X_new)[0]
        forecasts.append({'Date': date, 'Predicted AQI': pred_aqi, 'AQI Category': get_aqi_category(pred_aqi)})
        
        history.insert(0, pred_aqi)
        history = history[:8]
        
    return pd.DataFrame(forecasts)

def main():
    print("Step 3: Generating Forecast...")
    
    # Load config
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    # Load processed full data
    print("Loading processed data from 'data/processed_full.csv'...")
    try:
        full_df = pd.read_csv('data/processed_full.csv')
        full_df['Date'] = pd.to_datetime(full_df['Date'])
    except FileNotFoundError:
        print("Error: 'data/processed_full.csv' not found. Please run 01_process_data.py first.")
        return

    # Load Random Forest Model (Best Model)
    print(f"Loading Random Forest model from {config['MODEL_SAVE_PATH']}...")
    try:
        rf_model = joblib.load(os.path.join(config['MODEL_SAVE_PATH'], 'random_forest.pkl'))
        # Re-wrap in AQIRandomForest class structure if needed, but joblib loads the object.
        # Since we saved 'self.model' (the sklearn object) in AQIRandomForest.save_model,
        # we are loading a RandomForestRegressor, NOT an AQIRandomForest instance.
        # Wait, let's check src/models.py:
        # joblib.dump(self.model, ...) -> saves the sklearn model.
        # So 'rf_model' here is the sklearn model.
        # generate_forecast expects an object with .predict(). sklearn model has .predict().
        # However, generate_forecast passes a DataFrame. sklearn model expects array or DF with correct cols.
        # Our generate_forecast constructs a DataFrame with correct columns.
        # So it should work fine.
    except FileNotFoundError:
        print("Error: Model file not found. Please run 02_train.py first.")
        return

    # Generate Forecast
    print(f"\nGenerating {config['FORECAST_DAYS']}-Day Forecast...")
    forecast_df = generate_forecast(rf_model, full_df, days=config['FORECAST_DAYS'])
    
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
    
    # Ensure results directory exists
    os.makedirs(config['RESULTS_PATH'], exist_ok=True)
    save_path = os.path.join(config['RESULTS_PATH'], 'sequential_forecast.png')
    plt.savefig(save_path)
    print(f"Plot saved as '{save_path}'")

if __name__ == "__main__":
    main()
