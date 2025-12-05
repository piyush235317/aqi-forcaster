import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.preprocessing import load_and_process
from src.models import AQIRandomForest, AQIProphet, AQILSTM

def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[{model_name}] MAE: {mae:.2f}, RMSE: {rmse:.2f}")

def main():
    print("Step 2 (Tune): Tuning Models...")
    
    # 1. Load and Process Data
    # load_and_process returns train, test, config, full_df
    train, test, config, _ = load_and_process()
    
    if train is None:
        print("Error: Could not load data.")
        return

    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # 2. Tune and Train Models
    
    # Random Forest
    print("\nTuning Random Forest...")
    rf = AQIRandomForest()
    # Note: AQIRandomForest.tune refits the model on the provided data with best params
    rf.tune(train, train['AQI']) 
    
    # Prophet
    print("\nTuning Prophet...")
    prophet = AQIProphet()
    prophet.tune(train) 
    
    # LSTM
    print("\nTuning LSTM...")
    lstm = AQILSTM(epochs=20, batch_size=32)
    lstm.tune(train['AQI'].values)
    print("Training LSTM with best config...")
    lstm.train(train['AQI'].values)
    
    # 3. Generate Predictions on Test Set for Quick Check
    print("\nGenerating predictions for validation...")
    
    # Random Forest
    rf_pred = rf.predict(test)
    evaluate(test['AQI'], rf_pred, 'Random Forest (Tuned)')
    
    # Prophet
    prophet_forecast = prophet.predict(periods=len(test))
    prophet_pred = prophet_forecast['yhat'].tail(len(test)).values
    evaluate(test['AQI'], prophet_pred, 'Prophet (Tuned)')
    
    # LSTM
    # For LSTM validation here, we do a quick check. 
    # 05_compare_models.py has more robust sliding window logic.
    # Here we just want to ensure it runs.
    # We need context for LSTM.
    window_size = 30
    train_aqi = train['AQI'].values
    test_aqi = test['AQI'].values
    combined_data = np.concatenate((train_aqi[-window_size:], test_aqi))
    lstm_pred = lstm.predict(combined_data)
    lstm_pred = lstm_pred.flatten()
    
    if len(lstm_pred) == len(test):
        evaluate(test['AQI'], lstm_pred, 'LSTM (Tuned)')
    else:
        print(f"[LSTM] Warning: Pred length {len(lstm_pred)} != Test length {len(test)}")

    # 4. Save Best Model (Random Forest is usually best, but we save all tuned versions)
    print(f"\nSaving tuned models to {config['MODEL_SAVE_PATH']}...")
    rf.save_model(config['MODEL_SAVE_PATH'])
    prophet.save_model(config['MODEL_SAVE_PATH'])
    lstm.save_model(config['MODEL_SAVE_PATH'])
    
    # Save specific 'best_model.pkl' for the forecast script to pick up
    # We assume Random Forest is best based on typical performance on this data
    joblib.dump(rf.model, os.path.join(config['MODEL_SAVE_PATH'], 'best_model.pkl'))
    print(f"Best model (Random Forest) saved as '{os.path.join(config['MODEL_SAVE_PATH'], 'best_model.pkl')}'")

if __name__ == "__main__":
    main()
