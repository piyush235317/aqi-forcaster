import pandas as pd
import numpy as np
import yaml

def load_and_process(config_path='configs/config.yaml'):
    """
    Loads configuration and processes the data.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    print(f"Loading data from {config['DATA_PATH']}...")
    try:
        df = pd.read_csv(config['DATA_PATH'])
    except FileNotFoundError:
        print(f"Error: File '{config['DATA_PATH']}' not found.")
        return None, None, None, None

    # Filter by City
    print(f"Filtering for {config['CITY_NAME']}...")
    df = df[df['City'] == config['CITY_NAME']].copy()
    
    # Handle Missing Values
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    # Infer objects to avoid FutureWarning
    df = df.infer_objects(copy=False)
    # Select numeric columns for interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')
    df = df.bfill().ffill().reset_index()
    
    # Feature Engineering (Lags)
    print(f"Creating {config['LAG_DAYS']} lag features...")
    for i in range(1, config['LAG_DAYS'] + 1):
        df[f'AQI_lag_{i}'] = df['AQI'].shift(i)
    df = df.dropna()
    
    # Splitting
    print("Splitting data (Cutoff: 2020-01-01)...")
    train = df[df['Date'] < '2020-01-01'].copy()
    test = df[df['Date'] >= '2020-01-01'].copy()
    
    return train, test, config, df # Returning full df for future forecasting context if needed
