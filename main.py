import pandas as pd
import matplotlib.pyplot as plt
import importlib
import sys

from src.preprocessing import load_and_process
from src.models import AQIRandomForest, AQIProphet, AQILSTM
from src.utils import generate_forecast, plot_forecast, load_config

# Import analysis module
# Using importlib to handle the numeric filename
data_analysis = importlib.import_module("01_data_analysis")

# Import comparison module
# Using importlib to handle the numeric filename
compare_models = importlib.import_module("05_compare_models")

def main():
    config = load_config()

    # 0. Data Analysis
    print("Step 0: Running Data Analysis (EDA)...")
    data_analysis.generate_eda()

    # 1. Load and Process
    print("\nStep 1: Loading and processing data...")
    train, test, config, full_df = load_and_process()
    
    if train is None:
        return

    # 2. Tune & Train Models
    # Switched from basic training to Tuning to ensure best results
    print("\nStep 2: Tuning and Training Models...")
    
    # Random Forest
    print("Tuning Random Forest...")
    rf = AQIRandomForest()
    rf.tune(train, train['AQI']) # This tunes AND trains
    
    # Prophet
    print("Tuning Prophet...")
    prophet = AQIProphet()
    prophet.tune(train)
    
    # LSTM
    print("Tuning LSTM...")
    lstm = AQILSTM(epochs=100) # Increased to 100 for optimization
    lstm.tune(train['AQI'].values)
    print("Training LSTM with best config...")
    lstm.train(train['AQI'].values)
    
    # 3. Save Models
    print(f"\nStep 3: Saving models to {config['MODEL_SAVE_PATH']}...")
    rf.save_model(config['MODEL_SAVE_PATH'])
    prophet.save_model(config['MODEL_SAVE_PATH'])
    lstm.save_model(config['MODEL_SAVE_PATH'])
    
    # 4. Compare Models
    print("\nStep 4: Comparing Models (Generating Graphs)...")
    # We call the main function of the comparison script
    compare_models.main()
    
    # 5. Forecast (Using Random Forest as Best Model)
    print(f"\nStep 5: Generating {config['FORECAST_DAYS']}-Day Forecast (Random Forest)...")
    forecast_df = generate_forecast(rf.model, full_df, days=config['FORECAST_DAYS'])
    
    print("\nForecast Results:")
    print(forecast_df.to_string(index=False))
    
    # Plotting
    print("\nPlotting forecast...")
    plot_forecast(full_df, forecast_df, 
                 title=f"AQI Forecast: Next {config['FORECAST_DAYS']} Days ({config['CITY_NAME']})",
                 save_path='results/best_model_forecast.png')

if __name__ == "__main__":
    main()
