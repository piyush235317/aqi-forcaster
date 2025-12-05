import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import yaml
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import load_model
from src.preprocessing import load_and_process

def evaluate_model(y_true, y_pred, model_name):
    # Ensure standard types
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Handle division by zero for MAPE if any y_true is 0 (though unlikely for AQI)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100 # percentage
    
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}%")
    return mae, rmse, r2, mape

def main():
    print("Step 4: Comparing Models...")
    
    # Load config
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    print("Loading and splitting data...")
    train, test, _, full_df = load_and_process()
    
    if test is None:
        print("Error: Could not load data.")
        return
        
    y_test = test['AQI'].values
    results = {}
    
    # 1. Random Forest
    print("\nEvaluating Random Forest...")
    try:
        rf_path = os.path.join(config['MODEL_SAVE_PATH'], 'random_forest.pkl')
        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
            feature_cols = [f'AQI_lag_{i}' for i in range(1, 8)]
            X_test_rf = test[feature_cols]
            rf_pred = rf_model.predict(X_test_rf)
            results['Random Forest'] = evaluate_model(y_test, rf_pred, 'Random Forest')
        else:
            print("Random Forest model not found.")
    except Exception as e:
        print(f"Error evaluating Random Forest: {e}")

    # 2. Prophet
    print("\nEvaluating Prophet...")
    try:
        prophet_path = os.path.join(config['MODEL_SAVE_PATH'], 'prophet.pkl')
        if os.path.exists(prophet_path):
            prophet_model = joblib.load(prophet_path)
            future = pd.DataFrame({'ds': test['Date']})
            forecast = prophet_model.predict(future)
            prophet_pred = forecast['yhat'].values
            results['Prophet'] = evaluate_model(y_test, prophet_pred, 'Prophet')
        else:
            print("Prophet model not found.")
    except Exception as e:
        print(f"Error evaluating Prophet: {e}")

    # 3. LSTM
    print("\nEvaluating LSTM...")
    try:
        lstm_path = os.path.join(config['MODEL_SAVE_PATH'], 'lstm.keras')
        if os.path.exists(lstm_path):
            lstm_model = load_model(lstm_path)
            
            # Robust Window Construction
            window_size = 30
            # Get data spanning from [first_test_date - window] to [last_test_date - 1]
            first_test_date = test['Date'].min()
            start_date = first_test_date - pd.Timedelta(days=window_size)
            
            # Filter full_df by date range
            # We need strictly contiguous data. 'full_df' from load_and_process is interpolated and filled.
            mask = (full_df['Date'] >= start_date) & (full_df['Date'] <= test['Date'].max())
            lstm_data_slice = full_df.loc[mask, 'AQI'].values
            
            # Verify we have enough data
            expected_min_len = len(test) + window_size
            # It might be slightly less if start_date < full_df.Date.min(), but load_and_process usually handles ample history.
            
            X_lstm = []
            # We iterate through the TEST portion
            # If lstm_data_slice starts exactly 30 days before test:
            # Index 0 to 29 -> input for prediction 0
            # Index 1 to 30 -> input for prediction 1
            # We need to align carefully. 
            
            # Re-align: Let's find index of first_test_date in this slice
            # Actually, simpler: just iterate over the test dates and look up past 30 days in full_df
            # But that's slow. 
            
            # Let's rely on the slice being correct.
            # We need exactly len(test) predictions.
            # X_lstm[0] should be data[t-30 : t] where t is time of first test sample.
            
            # Check length
            if len(lstm_data_slice) < window_size + 1:
                 print("Not enough history for LSTM testing.")
            else:
                 # Construct sequences using sliding window on the slice
                 # The slice includes history + test data
                 # We want sequences ending JUST BEFORE each test point?
                 # Standard is: input [t-w ... t-1], target [t]
                 
                 # The slice contains [H_30 ... H_1, T_0, T_1 ... T_N]
                 # Seq 0: slice[0:30] -> Predicts slice[30] (which is T_0, correct)
                 # Seq 1: slice[1:31] -> Predicts slice[31] (which is T_1)
                 # ...
                 # Last Seq: ... -> Predicts slice[last]
                 
                 # Number of sequences = len(lstm_data_slice) - window_size
                 # This should equal len(test) if slice bounds are exact.
                 
                 # However, mask might return more data or less data depending on exact timestamps.
                 # Let's force alignment:
                 # We want to predict 'y_test'.
                 
                 preds = []
                 valid_y = [] # In case we miss some points
                 
                 inputs = []
                 for i in range(len(test)):
                     current_date = test['Date'].iloc[i]
                     # Get last 30 days BEFORE current_date
                     past_30 = full_df[(full_df['Date'] < current_date) & (full_df['Date'] >= current_date - pd.Timedelta(days=window_size))]
                     
                     if len(past_30) == window_size:
                         inputs.append(past_30['AQI'].values)
                     else:
                         # Padding or skipping? Skipping means y_test is mismatched.
                         # Better to pad with mean or fill?
                         # Given earlier interpolation, this should only happen at very start of dataset.
                         # Since test is 2020+, and data starts 2015, we are safe.
                         pass

                 if inputs:
                     inputs = np.array(inputs)
                     inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1))
                     inputs = np.asarray(inputs).astype('float32') # Ensure type for TF
                     
                     lstm_pred = lstm_model.predict(inputs, verbose=0).flatten()
                     
                     # Ensure lengths match
                     if len(lstm_pred) == len(y_test):
                        results['LSTM'] = evaluate_model(y_test, lstm_pred, 'LSTM')
                     else:
                         print(f"LSTM dimension mismatch: pred {len(lstm_pred)} vs true {len(y_test)}")
                 else:
                     print("Could not construct valid inputs for LSTM.")

        else:
            print("LSTM model not found.")
    except Exception as e:
        print(f"Error evaluating LSTM: {e}")
        import traceback
        traceback.print_exc()

    # Plotting
    if results:
        print("\nPlotting comparison...")
        models = list(results.keys())
        
        # metrics = (mae, rmse, r2, mape)
        mae = [results[m][0] for m in models]
        rmse = [results[m][1] for m in models]
        r2 = [results[m][2] for m in models]
        mape = [results[m][3] for m in models]
        
        
        # Plot 1: Errors (MAE, RMSE)
        x = np.arange(len(models))
        width = 0.35
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        rects1 = ax1.bar(x - width/2, mae, width, label='MAE', color='skyblue')
        rects2 = ax1.bar(x + width/2, rmse, width, label='RMSE', color='salmon')
        
        ax1.set_ylabel('Error Value')
        ax1.set_title('Model Errors (Lower is Better)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Annotations for Plot 1
        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -12),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1, ax1)
        autolabel(rects2, ax1)
        
        fig1.tight_layout()
        save_path_errors = 'results/model_errors.png'
        plt.figure(fig1.number) # Set current figure
        plt.savefig(save_path_errors)
        print(f"Error plot saved to '{save_path_errors}'")
        plt.close(fig1)

        # Plot 2: R2 Score (Accuracy)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        rects3 = ax2.bar(models, r2, color='lightgreen', width=0.5)
        ax2.set_ylabel('R2 Score (Higher is Better)')
        ax2.set_title('Model Fit Quality (R2 Score)')
        ax2.set_ylim(bottom=min(min(r2)*1.2, 0), top=1.0) 
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        autolabel(rects3, ax2)
        
        fig2.tight_layout()
        save_path_r2 = 'results/model_r2_score.png'
        plt.figure(fig2.number)
        plt.savefig(save_path_r2)
        print(f"R2 score plot saved to '{save_path_r2}'")
        plt.close(fig2)

        
        # Also print summary table
        print("\nSummary Validation Table:")
        print(f"{'Model':<15} | {'MAE':<10} | {'RMSE':<10} | {'R2':<10} | {'MAPE (%)':<10}")
        print("-" * 65)
        for m in models:
            metrics = results[m]
            print(f"{m:<15} | {metrics[0]:<10.2f} | {metrics[1]:<10.2f} | {metrics[2]:<10.2f} | {metrics[3]:<10.2f}")
            
    else:
        print("No models were evaluated successfully.")

if __name__ == "__main__":
    main()
