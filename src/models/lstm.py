import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split

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

    def tune(self, train_data):
        """
        Tunes LSTM architecture.
        """
        data = np.array(train_data)
        X, y = self.create_sequences(data)
        if X.ndim == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        configs = ['A', 'B', 'C']
        best_loss = float('inf')
        best_config = 'A'
        
        print("Tuning LSTM...")
        for config in configs:
            print(f"Testing Config {config}...")
            model = self.build_model((X_train.shape[1], X_train.shape[2]), config=config)
            history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), verbose=0)
            val_loss = history.history['val_loss'][-1]
            print(f"Config {config} Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_config = config
                
        print(f"Best LSTM Config: {best_config}")
        self.config = best_config
        return self.config
