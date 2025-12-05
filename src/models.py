import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split

class AQIRandomForest:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.feature_cols = [f'AQI_lag_{i}' for i in range(1, 8)]

    def train(self, X_train, y_train):
        X = X_train[self.feature_cols]
        self.model.fit(X, y_train)

    def predict(self, X_test):
        X = X_test[self.feature_cols]
        return self.model.predict(X)

    def tune(self, X_train, y_train):
        X = X_train[self.feature_cols]
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_leaf': [1, 2, 4]
        }
        print("Tuning Random Forest...")
        grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                                   param_grid=param_grid,
                                   cv=3,
                                   scoring='neg_root_mean_squared_error',
                                   n_jobs=-1,
                                   verbose=1)
        grid_search.fit(X, y_train)
        print(f"Best Random Forest Params: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_
        return self.model

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.model, os.path.join(save_path, 'random_forest.pkl'))
        print(f"Random Forest saved to {save_path}")

class AQIProphet:
    def __init__(self, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                             seasonality_prior_scale=seasonality_prior_scale)

    def train(self, df_train):
        df = df_train[['Date', 'AQI']].rename(columns={'Date': 'ds', 'AQI': 'y'})
        self.model.fit(df)

    def predict(self, periods=30):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        # Prophet models can be pickled
        joblib.dump(self.model, os.path.join(save_path, 'prophet.pkl'))
        print(f"Prophet saved to {save_path}")

class AQILSTM:
    def __init__(self, window_size=30, units=50, epochs=50, batch_size=32):
        self.window_size = window_size
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.config = 'A'

    def create_sequences(self, data):
        X, y = [], []
        if len(data) <= self.window_size:
            return np.array(X), np.array(y)
        for i in range(len(data) - self.window_size):
            X.append(data[i:(i + self.window_size)])
            y.append(data[i + self.window_size])
        return np.array(X), np.array(y)

    def build_model(self, input_shape, config='A'):
        model = Sequential()
        if config == 'A':
            model.add(LSTM(50, activation='relu', input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(Dense(1))
        elif config == 'B':
            model.add(LSTM(100, activation='relu', input_shape=input_shape))
            model.add(Dropout(0.3))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))
        elif config == 'C':
            model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, train_data):
        data = np.array(train_data)
        X_train, y_train = self.create_sequences(data)
        if X_train.ndim == 2:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]), config=self.config)
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, data):
        data = np.array(data)
        X_test, _ = self.create_sequences(data)
        if len(X_test) == 0:
             if len(data) == self.window_size:
                 X_test = data.reshape((1, self.window_size, 1))
             else:
                 return np.array([])
        if X_test.ndim == 2:
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        return self.model.predict(X_test)

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.model.save(os.path.join(save_path, 'lstm.keras'))
        print(f"LSTM saved to {save_path}")
