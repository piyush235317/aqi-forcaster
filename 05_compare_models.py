import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from tensorflow.keras.models import load_model

from src.preprocessing import load_and_process
from src.utils import load_config, evaluate_model_metrics, evaluate_classification_metrics, plot_confusion_matrix

def main():
    print("Step 4: Comparing Models...")
    
    config = load_config()
        
    print("Loading and splitting data...")
    train, test, _, full_df = load_and_process()
    
    if test is None:
        print("Error: Could not load data.")
        return
        
    y_test = test['AQI'].values
    results = {}
    cls_results = {}

    # --- 1. Random Forest ---
    print("\nEvaluating Random Forest...")
    try:
        rf_path = os.path.join(config['MODEL_SAVE_PATH'], 'random_forest.pkl')
        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
            # Ensure features match what RF expects (lags)
            feature_cols = [f'AQI_lag_{i}' for i in range(1, 8)]
            if set(feature_cols).issubset(test.columns):
                X_test_rf = test[feature_cols]
                rf_pred = rf_model.predict(X_test_rf)
                results['Random Forest'] = evaluate_model_metrics(y_test, rf_pred, 'Random Forest')
                acc, f1, _, y_true_cat, rf_pred_cat = evaluate_classification_metrics(y_test, rf_pred, 'Random Forest')
                cls_results['Random Forest'] = (acc, f1)
                plot_confusion_matrix(y_true_cat, rf_pred_cat, 'Random Forest')
            else:
                 print("Lag features missing in test set for Random Forest.")
        else:
            print("Random Forest model not found.")
    except Exception as e:
        print(f"Error evaluating Random Forest: {e}")

    # --- 2. Prophet ---
    print("\nEvaluating Prophet...")
    try:
        prophet_path = os.path.join(config['MODEL_SAVE_PATH'], 'prophet.pkl')
        if os.path.exists(prophet_path):
            prophet_model = joblib.load(prophet_path)
            future = pd.DataFrame({'ds': test['Date']})
            forecast = prophet_model.predict(future)
            prophet_pred = forecast['yhat'].values
            results['Prophet'] = evaluate_model_metrics(y_test, prophet_pred, 'Prophet')
            acc, f1, _, y_true_cat, prophet_pred_cat = evaluate_classification_metrics(y_test, prophet_pred, 'Prophet')
            cls_results['Prophet'] = (acc, f1)
            plot_confusion_matrix(y_true_cat, prophet_pred_cat, 'Prophet')
        else:
            print("Prophet model not found.")
    except Exception as e:
        print(f"Error evaluating Prophet: {e}")

    # --- 3. LSTM ---
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
            
            mask = (full_df['Date'] >= start_date) & (full_df['Date'] <= test['Date'].max())
            lstm_data_slice = full_df.loc[mask, 'AQI'].values
            
            if len(lstm_data_slice) < window_size + 1:
                 print("Not enough history for LSTM testing.")
            else:
                 inputs = []
                 # Naive sliding window for test comparison
                 # Note: Ideally should be vectorized, but loop is safe for clarity here
                 for i in range(len(test)):
                     current_date = test['Date'].iloc[i]
                     past_30 = full_df[(full_df['Date'] < current_date) & (full_df['Date'] >= current_date - pd.Timedelta(days=window_size))]
                     
                     if len(past_30) == window_size:
                         inputs.append(past_30['AQI'].values)

                 if inputs:
                     inputs = np.array(inputs)
                     inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1))
                     inputs = np.asarray(inputs).astype('float32') # Ensure type for TF
                     
                     lstm_pred = lstm_model.predict(inputs, verbose=0).flatten()
                     
                     # Ensure lengths match
                     if len(lstm_pred) == len(y_test): # Assuming one-to-one for valid inputs
                        results['LSTM'] = evaluate_model_metrics(y_test, lstm_pred, 'LSTM')
                        acc, f1, _, y_true_cat, lstm_pred_cat = evaluate_classification_metrics(y_test, lstm_pred, 'LSTM')
                        cls_results['LSTM'] = (acc, f1)
                        plot_confusion_matrix(y_true_cat, lstm_pred_cat, 'LSTM')
                     else:
                         print(f"LSTM dimension mismatch: pred {len(lstm_pred)} vs true {len(y_test)}")
                 else:
                     print("Could not construct valid inputs for LSTM.")
        else:
            print("LSTM model not found.")
    except Exception as e:
        print(f"Error evaluating LSTM: {e}")


    # --- Plotting Comparison ---
    if results:
        print("\nPlotting comparison...")
        models = list(results.keys())
        
        # metrics = (mae, rmse, r2, mape)
        mae = [results[m][0] for m in models]
        rmse = [results[m][1] for m in models]
        r2 = [results[m][2] for m in models]
        
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
        plt.figure(fig1.number)
        plt.savefig(save_path_errors)
        plt.close(fig1)

        # Plot 2: R2 Score
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
        plt.close(fig2)

        # Plot 3: Classification Metrics
        if cls_results:
            cls_models = list(cls_results.keys())
            accuracies = [cls_results[m][0] for m in cls_models]
            f1_scores = [cls_results[m][1] for m in cls_models]
            
            x_cls = np.arange(len(cls_models))
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            rects_acc = ax3.bar(x_cls - width/2, accuracies, width, label='Accuracy', color='lightskyblue')
            rects_f1 = ax3.bar(x_cls + width/2, f1_scores, width, label='Weighted F1', color='plum')
            
            ax3.set_ylabel('Score (0-1)')
            ax3.set_title('Classification Performance (Accuracy & F1)')
            ax3.set_xticks(x_cls)
            ax3.set_xticklabels(cls_models)
            ax3.legend()
            ax3.set_ylim(0, 1.05)
            ax3.grid(axis='y', linestyle='--', alpha=0.7)
            
            autolabel(rects_acc, ax3)
            autolabel(rects_f1, ax3)
            
            fig3.tight_layout()
            save_path_cls = 'results/model_accuracy_comparison.png'
            plt.figure(fig3.number)
            plt.savefig(save_path_cls)
            plt.close(fig3)

        # Plot 4: Time Series Comparison (Actual vs All Models)
        print("\nPlotting time series comparison...")
        plt.figure(figsize=(15, 6))
        
        # Plot Actual
        plt.plot(test['Date'], y_test, label='Actual', color='black', linewidth=2, linestyle='--')
        
        # Plot Models (using gathered data)
        # We need to access the variables from the try-except blocks. 
        # Since they are local variables, we assume they are available if results has the key.
        # However, because of scoping in python functions, we need to ensure we stored them or they are accessible.
        # The safest way is to store them in a dictionary during evaluation.
        
        # NOTE: I need to retroactively ensure predictions are stored in a dict in previous steps.
        # But for this replace_file_content, I cannot see the scope variables easily if I don't change the blocks.
        # I will rely on locals() or better, I will assume I need to modify the blocks above.
        # Wait, I can't modify blocks above in *this* call easily if I am appending here.
        # So I will use a separate replace_tool call to modify the blocks to store predictions, 
        # OR I can just access them if they bound (Python 3 leaks list comp vars but try-except vars? 
        # Variables defined in try block ARE available after if no error).
        
        # Let's check if 'rf_pred', 'prophet_pred', 'lstm_pred' are available. 
        # They will be available if the corresponding block executed successfully.
        
        if 'Random Forest' in results:
             plt.plot(test['Date'], rf_pred, label='Random Forest', alpha=0.8)
        
        if 'Prophet' in results:
             plt.plot(test['Date'], prophet_pred, label='Prophet', alpha=0.8)
             
        if 'LSTM' in results:
             plt.plot(test['Date'], lstm_pred, label='LSTM', alpha=0.8)

        plt.title('Model Predictions Comparison')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/model_comparison_timeseries.png')
        plt.close()
        
        print("\nSummary Validation Table:")
        print(f"{'Model':<15} | {'MAE':<10} | {'RMSE':<10} | {'R2':<10} | {'MAPE (%)':<10}")
        print("-" * 65)
        for m in models:
            metrics = results[m]
            print(f"{m:<15} | {metrics[0]:<10.2f} | {metrics[1]:<10.2f} | {metrics[2]:<10.2f} | {metrics[3]:<10.2f}")

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
