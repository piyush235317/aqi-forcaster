import pandas as pd
import os
from src.preprocessing import load_and_process

def main():
    print("Step 1: Processing Data...")
    
    train, test, config, full_df = load_and_process()
    
    if train is None:
        print("Data processing failed.")
        return

    os.makedirs('data', exist_ok=True)
    
    print("Saving datasets to 'data/'...")
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    full_df.to_csv('data/processed_full.csv', index=False)
    
    print("Data processing complete. Files saved: train.csv, test.csv, processed_full.csv")

if __name__ == "__main__":
    main()
