import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from src.utils import load_config, generate_forecast, plot_forecast

def main():
    print("Step 3: Generating Forecast...")
    
    config = load_config()
        
    print("Loading processed data from 'data/processed_full.csv'...")
    try:
        full_df = pd.read_csv('data/processed_full.csv')
        full_df['Date'] = pd.to_datetime(full_df['Date'])
    except FileNotFoundError:
        print("Error: 'data/processed_full.csv' not found. Please run 01_process_data.py first.")
        return

    print(f"Loading Random Forest model from {config['MODEL_SAVE_PATH']}...")
    try:
        rf_model = joblib.load(os.path.join(config['MODEL_SAVE_PATH'], 'random_forest.pkl'))
    except FileNotFoundError:
        print("Error: Model file not found. Please run 02_train.py first.")
        return

    print(f"\nGenerating {config['FORECAST_DAYS']}-Day Forecast...")
    forecast_df = generate_forecast(rf_model, full_df, days=config['FORECAST_DAYS'])
    
    print("\nForecast Results:")
    print(forecast_df.to_string(index=False))
    
    print("\nPlotting forecast...")
    save_path = os.path.join(config['RESULTS_PATH'], 'sequential_forecast.png')
    os.makedirs(config['RESULTS_PATH'], exist_ok=True)
    
    plot_forecast(full_df, forecast_df, 
                 title=f"AQI Forecast: Next {config['FORECAST_DAYS']} Days ({config['CITY_NAME']})",
                 save_path=save_path)

if __name__ == "__main__":
    main()
