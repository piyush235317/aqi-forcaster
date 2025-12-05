import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_aqi_category(aqi):
    """Determine category based on AQI value (Indian Standard)."""
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def generate_forecast(model, last_available_data, days=7):
    """
    Generates future forecast using recursive strategy.
    """
    future_dates = pd.date_range(start=last_available_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days)
    forecasts = []
    
    current_data = last_available_data.iloc[-1].copy()
    last_known_aqi = current_data['AQI']
    
    # Identify lag features dynamically
    lag_cols = [c for c in last_available_data.columns if c.startswith('AQI_lag_')]
    # Sort them to ensure order: lag_1, lag_2 ...
    lag_cols.sort(key=lambda x: int(x.split('_')[-1]))
    max_lag = len(lag_cols)
    
    if max_lag == 0:
         raise ValueError("No lag features found for recursive forecasting.")

    # History should contain [AQI_T, AQI_T-1, ... AQI_T-(max_lag-1)]
    # current_data has AQI (T), AQI_lag_1 (T-1), ...
    # We need history to populate next day's lag_1, lag_2...
    
    # Next Day:
    # lag_1 = current AQI
    # lag_2 = current lag_1
    # ...
    
    # So history state needed is [current_AQI, current_lag_1, current_lag_2, ...]
    history = [last_known_aqi] + [current_data[col] for col in lag_cols[:-1]]
    
    for date in future_dates:
        features = {}
        # Persistence for non-lag features
        for col in last_available_data.columns:
            if 'lag' not in col and col not in ['Date', 'AQI', 'City', 'AQI_Bucket']:
                features[col] = current_data[col]
        
        # Update lag features
        for i, col in enumerate(lag_cols):
            # lag_1 (i=0) takes history[0]
            features[col] = history[i]
            
        X_new = pd.DataFrame([features])
        
        if set(lag_cols).issubset(X_new.columns):
             X_new = X_new[lag_cols]

        pred_aqi = model.predict(X_new)[0]
        forecasts.append({'Date': date, 'Predicted AQI': pred_aqi, 'AQI Category': get_aqi_category(pred_aqi)})
        
        history.insert(0, pred_aqi)
        history = history[:max_lag]
        
    return pd.DataFrame(forecasts)

def evaluate_model_metrics(y_true, y_pred, model_name):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}%")
    return mae, rmse, r2, mape

def evaluate_classification_metrics(y_true, y_pred, model_name):
    y_true_cat = [get_aqi_category(val) for val in y_true]
    y_pred_cat = [get_aqi_category(val) for val in y_pred]
    
    acc = accuracy_score(y_true_cat, y_pred_cat)
    f1 = f1_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
    
    report_str = f"\n[{model_name}] Classification Report (Regression-to-Classification):\n"
    report_str += f"Accuracy: {acc:.2%}\n"
    report_str += f"Weighted F1-Score: {f1:.2f}\n"
    report_str += "Confusion Matrix:\n"
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'])
    report_str += str(cm)
    
    print(report_str)
    return acc, f1, report_str, y_true_cat, y_pred_cat

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir='results'):
    cm = confusion_matrix(y_true, y_pred, labels=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'],
                yticklabels=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'])
    
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to '{filename}'")

def plot_forecast(full_df, forecast_df, title, save_path):
    plt.figure(figsize=(12, 6))
    last_30_days = full_df.iloc[-30:]
    plt.plot(last_30_days['Date'], last_30_days['AQI'], label='Actual History', color='black')
    
    last_date = last_30_days['Date'].iloc[-1]
    last_aqi = last_30_days['AQI'].iloc[-1]
    plot_dates = [last_date] + list(forecast_df['Date'])
    plot_aqi = [last_aqi] + list(forecast_df['Predicted AQI'])
    
    plt.plot(plot_dates, plot_aqi, label='Forecast', color='red', linestyle='--', marker='o')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved as '{save_path}'")
