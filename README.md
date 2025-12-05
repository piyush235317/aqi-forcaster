# AQI Forecasting Project - Group 14

This project forecasts Air Quality Index (AQI) for Delhi using three machine learning models:
1.  **Random Forest** (using lag features)
2.  **Facebook Prophet** (Time-series forecasting)
3.  **LSTM** (Deep Learning with TensorFlow/Keras)

## Project Structure
- `data_processor.py`: Handles data loading, cleaning, and feature engineering.
- `models/`: Contains the model classes (`random_forest.py`, `prophet_model.py`, `lstm.py`).
- `evaluation.py`: Helper module for calculating metrics and plotting results.
- `main.py`: Runs the baseline models with default hyperparameters.
- `fine_tuning.py`: Runs the hyperparameter tuning process to find the best models and evaluates them.

## Setup Instructions

1.  **Install Dependencies**:
    Ensure you have Python installed. Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib tensorflow prophet
    ```

2.  **Data**:
    Ensure `city_day.csv` is present in the project root directory.

## How to Run Manually

### Option 1: Run Baseline Models
To train and evaluate the models with default settings:
```bash
python main.py
```
This will generate `aqi_forecast_comparison.png`.

### Option 2: Run Hyperparameter Tuning (Recommended)
To tune the models for better performance and evaluate them:
```bash
python fine_tuning.py
```
This will:
1.  Tune Random Forest (Grid Search).
2.  Tune Prophet (Validation Split).
3.  Tune LSTM (Architecture Search).
4.  Train the best versions on the full training set.
5.  Evaluate on the test set.
6.  Generate `tuned_model_comparison.png`.

## Results
Check the generated `.png` files for visual comparison and the console output for RMSE and MAE scores.
