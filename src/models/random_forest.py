import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

class AQIRandomForest:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.feature_cols = None

    def train(self, X_train, y_train):
        # Dynamically Identify lag features
        self.feature_cols = [c for c in X_train.columns if c.startswith('AQI_lag_')]
        if not self.feature_cols:
             # Fallback if no lags, use all numeric except target (simplistic)
             self.feature_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
             if 'AQI' in self.feature_cols: self.feature_cols.remove('AQI')
        
        X = X_train[self.feature_cols]
        self.model.fit(X, y_train)

    def predict(self, X_test):
        if self.feature_cols is None:
             # If loaded without training (shouldn't happen in this flow usually, but safe to check)
             raise ValueError("Model has not been trained yet.")
        X = X_test[self.feature_cols]
        return self.model.predict(X)

    def tune(self, X_train, y_train):
        # Identify features if not already done
        if self.feature_cols is None:
            self.feature_cols = [c for c in X_train.columns if c.startswith('AQI_lag_')]
            
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
