import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from src.utils import generate_forecast, plot_forecast
from src.preprocessing import load_and_process

def main():
    print("Step 5: Generating Forecast with Best Model...")
    
    # Validating we can reuse load_and_process here
    _, _, config, full_df = load_and_process()
    
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
    
    print("\nPlotting forecast...")
    save_path = os.path.join(config['RESULTS_PATH'], 'best_model_forecast.png')
    plot_forecast(full_df, forecast_df,
                 title=f"AQI Forecast: Next {config['FORECAST_DAYS']} Days ({config['CITY_NAME']})",
                 save_path=save_path)

if __name__ == "__main__":
    main()
