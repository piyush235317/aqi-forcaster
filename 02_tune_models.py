import pandas as pd
import numpy as np
import os
import joblib
from src.preprocessing import load_and_process
from src.models import AQIRandomForest, AQIProphet, AQILSTM
from src.utils import evaluate_model_metrics

def main():
    print("Step 2 (Tune): Tuning Models...")
    
    train, test, config, _ = load_and_process()
    
    if train is None:
        print("Error: Could not load data.")
        return

    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # --- Random Forest ---
    print("\nTuning Random Forest...")
    rf = AQIRandomForest()
    rf.tune(train, train['AQI']) 
    
    # --- Prophet ---
    print("\nTuning Prophet...")
    prophet = AQIProphet()
    prophet.tune(train) 
    
    # --- LSTM ---
    print("\nTuning LSTM...")
    lstm = AQILSTM(epochs=20, batch_size=32)
    lstm.tune(train['AQI'].values)
    print("Training LSTM with best config...")
    lstm.train(train['AQI'].values)
    
    # --- Validation ---
    print("\nGenerating predictions for validation...")
    
    # RF
    rf_pred = rf.predict(test)
    evaluate_model_metrics(test['AQI'], rf_pred, 'Random Forest (Tuned)')
    
    # Prophet
    prophet_forecast = prophet.predict(periods=len(test))
    prophet_pred = prophet_forecast['yhat'].tail(len(test)).values
    evaluate_model_metrics(test['AQI'], prophet_pred, 'Prophet (Tuned)')
    
    # LSTM
    window_size = 30
    combined_data = np.concatenate((train['AQI'].values[-window_size:], test['AQI'].values))
    lstm_pred = lstm.predict(combined_data).flatten()
    
    if len(lstm_pred) == len(test):
        evaluate_model_metrics(test['AQI'], lstm_pred, 'LSTM (Tuned)')
    else:
        print(f"[LSTM] Warning: Pred length {len(lstm_pred)} != Test length {len(test)}")

    # --- Save Models ---
    print(f"\nSaving tuned models to {config['MODEL_SAVE_PATH']}...")
    rf.save_model(config['MODEL_SAVE_PATH'])
    prophet.save_model(config['MODEL_SAVE_PATH'])
    lstm.save_model(config['MODEL_SAVE_PATH'])
    
    best_model_path = os.path.join(config['MODEL_SAVE_PATH'], 'best_model.pkl')
    joblib.dump(rf.model, best_model_path)
    print(f"Best model (Random Forest) saved as '{best_model_path}'")

if __name__ == "__main__":
    main()
