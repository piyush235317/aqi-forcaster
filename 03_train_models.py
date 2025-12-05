import pandas as pd
import yaml
from src.models import AQIRandomForest, AQIProphet, AQILSTM

def main():
    print("Step 2: Training Models...")
    
    # Load config
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    # Load training data
    print("Loading training data from 'data/train.csv'...")
    try:
        train = pd.read_csv('data/train.csv')
        train['Date'] = pd.to_datetime(train['Date'])
    except FileNotFoundError:
        print("Error: 'data/train.csv' not found. Please run 01_process_data.py first.")
        return

    # Train Models
    print("\nTraining Random Forest...")
    rf = AQIRandomForest()
    rf.train(train, train['AQI'])
    
    print("Training Prophet...")
    prophet = AQIProphet()
    prophet.train(train)
    
    print("Training LSTM...")
    lstm = AQILSTM(epochs=10)
    lstm.train(train['AQI'].values)
    
    # Save Models
    print(f"\nSaving models to {config['MODEL_SAVE_PATH']}...")
    rf.save_model(config['MODEL_SAVE_PATH'])
    prophet.save_model(config['MODEL_SAVE_PATH'])
    lstm.save_model(config['MODEL_SAVE_PATH'])
    
    print("Training complete. Models saved.")

if __name__ == "__main__":
    main()
