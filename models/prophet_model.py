from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class AQIProphet:
    def __init__(self, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                             seasonality_prior_scale=seasonality_prior_scale)

    def train(self, df_train):
        """
        Trains the Prophet model.
        Expects df_train to have 'Date' and 'AQI' columns.
        """
        # Prepare data for Prophet
        df = df_train[['Date', 'AQI']].rename(columns={'Date': 'ds', 'AQI': 'y'})
        self.model.fit(df)

    def predict(self, periods=30):
        """
        Forecasts AQI for the next 'periods' days.
        Returns the forecast DataFrame.
        """
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast

    def tune(self, df_train):
        """
        Tunes Prophet hyperparameters using a validation split.
        """
        # Split into train and validation (last 20%)
        n = len(df_train)
        split_idx = int(n * 0.8)
        train_sub = df_train.iloc[:split_idx].copy()
        val_sub = df_train.iloc[split_idx:].copy()
        
        param_grid = {
            'changepoint_prior_scale': [0.01, 0.1, 0.5],
            'seasonality_prior_scale': [1.0, 10.0]
        }
        
        best_params = {}
        best_rmse = float('inf')
        
        print("Tuning Prophet...")
        for cps in param_grid['changepoint_prior_scale']:
            for sps in param_grid['seasonality_prior_scale']:
                # Suppress Prophet output
                model = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps)
                df_prophet = train_sub[['Date', 'AQI']].rename(columns={'Date': 'ds', 'AQI': 'y'})
                model.fit(df_prophet)
                
                future = model.make_future_dataframe(periods=len(val_sub))
                forecast = model.predict(future)
                preds = forecast['yhat'].tail(len(val_sub)).values
                
                rmse = np.sqrt(mean_squared_error(val_sub['AQI'], preds))
                print(f"Params: cps={cps}, sps={sps} -> RMSE: {rmse:.2f}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps}
        
        print(f"Best Prophet Params: {best_params}")
        self.changepoint_prior_scale = best_params['changepoint_prior_scale']
        self.seasonality_prior_scale = best_params['seasonality_prior_scale']
        self.model = Prophet(changepoint_prior_scale=self.changepoint_prior_scale,
                             seasonality_prior_scale=self.seasonality_prior_scale)
        return self.model
