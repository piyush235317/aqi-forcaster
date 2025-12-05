from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

class AQIRandomForest:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.feature_cols = [f'AQI_lag_{i}' for i in range(1, 8)]

    def train(self, X_train, y_train):
        """
        Trains the Random Forest model.
        Expects X_train to contain the lag features.
        """
        # Ensure we only use the lag features
        X = X_train[self.feature_cols]
        self.model.fit(X, y_train)

    def predict(self, X_test):
        """
        Predicts AQI using the trained model.
        """
        X = X_test[self.feature_cols]
        return self.model.predict(X)

    def tune(self, X_train, y_train):
        """
        Tunes hyperparameters using GridSearchCV.
        """
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
