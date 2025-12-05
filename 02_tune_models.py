import pandas as pd
import numpy as np
from data_processor import load_data, filter_by_city, handle_missing_values, create_lag_features, get_train_test_split
from models.random_forest import AQIRandomForest
from models.prophet_model import AQIProphet
from models.lstm import AQILSTM
from evaluation import evaluate_models

def main():
    # 1. Load and Process Data
    print("Loading and processing data...")
    df = load_data()
    df = filter_by_city(df)
    df = handle_missing_values(df)
    df = create_lag_features(df)
    
    # 2. Split Data
    print("Splitting data...")
    train, test = get_train_test_split(df)
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # 3. Tune and Train Models
    
    # Random Forest
    print("\nTuning Random Forest...")
    rf = AQIRandomForest()
    rf.tune(train, train['AQI']) 
    # rf.model is now the best estimator refitted on the training data
    
    # Prophet
    print("\nTuning Prophet...")
    prophet = AQIProphet()
    prophet.tune(train) 
    print("Training Prophet with best params...")
    prophet.train(train)
    
    # LSTM
    print("\nTuning LSTM...")
    lstm = AQILSTM(epochs=20, batch_size=32)
    lstm.tune(train['AQI'].values)
    print("Training LSTM with best config...")
    lstm.train(train['AQI'].values)
    
    # 4. Generate Predictions
    print("\nGenerating predictions...")
    
    # Random Forest
    preds_rf = rf.predict(test)
    
    # Prophet
    preds_prophet_df = prophet.predict(periods=len(test))
    preds_prophet = preds_prophet_df['yhat'].tail(len(test)).values
    
    # LSTM
    window_size = 30
    train_aqi = train['AQI'].values
    test_aqi = test['AQI'].values
    combined_data = np.concatenate((train_aqi[-window_size:], test_aqi))
    preds_lstm = lstm.predict(combined_data)
    preds_lstm = preds_lstm.flatten()
    
    # 5. Evaluation & Visualization
    print("\nEvaluation Results (Tuned Models):")
    y_test = test['AQI'].values
    dates = test['Date']
    
    predictions_dict = {
        'Random Forest (Tuned)': preds_rf,
        'Prophet (Tuned)': preds_prophet,
        'LSTM (Tuned)': preds_lstm
    }
    
    evaluate_models(dates, y_test, predictions_dict, filename='tuned_model_comparison.png')
    
    # 6. Save Best Model (Random Forest)
    import joblib
    import os
    print("\nSaving best model (Random Forest)...")
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(rf, 'saved_models/best_model.pkl')
    print("Model saved as 'saved_models/best_model.pkl'")

if __name__ == "__main__":
    main()
