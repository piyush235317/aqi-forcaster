# AQI Forecasting Project - Group 14

This project forecasts Air Quality Index (AQI) for Delhi using three machine learning models:
1.  **Random Forest** (Regression with Lag Features)
2.  **Facebook Prophet** (Time-series Forecasting)
3.  **LSTM** (Long Short-Term Memory Neural Network)

## Project Structure
The project is organized into a modular structure:

-   **`configs/`**: Contains `config.yaml` for centralizing parameters (paths, lag days, etc.).
-   **`src/`**: Source code modules.
    -   `preprocessing.py`: Data loading and feature engineering logic.
    -   `models/`: Project specific model package (`random_forest.py`, `prophet_model.py`, `lstm.py`).
-   **`trained_models/`**: Stores trained binary models (`.pkl`, `.keras`).
-   **`results/`**: Stores generated plots and evaluation metrics.
-   **`data/`**: Stores raw and processed CSV data.

## Quick Start (Recommended)
Run the entire pipeline (Analysis -> Tuning -> Comparison -> Forecast) with a single command:
```bash
python main.py
```
This script handles everything automatically using `src.utils` and optimized model parameters.

## Detailed Workflow & Scripts
The project pipeline is divided into sequential modules, which can also be run individually:

### 1. Data Preparation
-   **`00_process_data.py`**: Loads raw data, fills missing values, creates lag features, and splits into train/test sets. Saves to `data/`.
-   **`01_data_analysis.py`**: Generates Exploratory Data Analysis (EDA) plots to `results/`.

### 2. Model Tuning & Development
-   **`02_tune_models.py`**: Fine-tunes all models (RF, Prophet, LSTM) using Grid Search / Validation splits. Saves the best versions to `trained_models/`.
-   **`03_train_models.py`**: A faster alternative to tuning. Trains models with default parameters.

### 3. Forecasting & Evaluation
-   **`04_forecast.py`**: Generates a fast forecast using the default models.
-   **`05_compare_models.py`**: Loads saved models and performs a detailed metric comparison (MAE, RMSE, R2), classification analysis, and **Time Series Plotting**.
-   **`06_forecast_best.py`**: Uses the **best performing model** (typically Random Forest) tuned in step 02 to generate the final future forecast.

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn tensorflow prophet pyyaml joblib
    ```

2.  **Data**:
    Ensure the raw data file is present at `data/city_day.csv` (or update `configs/config.yaml`).

## Usage Example
```bash
# Run the full automated pipeline
python main.py
```
Check the `results/` folder for all generated plots and `trained_models/` for the saved model files.
