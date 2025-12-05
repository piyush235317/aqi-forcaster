import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

# Set style
plt.style.use('ggplot')
sns.set_palette("tab10")

from src.utils import load_config

def generate_eda():
    print("Generating EDA visualizations...")
    os.makedirs('results', exist_ok=True)
    
    # Load processed data if available, else raw
    if os.path.exists('data/processed_full.csv'):
        df = pd.read_csv('data/processed_full.csv')
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        print("Processed data not found, trying raw data...")
        config = load_config()
        df = pd.read_csv(config['DATA_PATH'])
        df = df[df['City'] == config['CITY_NAME']].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df['AQI'] = df['AQI'].interpolate(method='time') # Basic fix for plotting

    # 1. Full Time Series Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['AQI'], color='#1f77b4', linewidth=1)
    plt.title('Daily AQI Trends in Delhi (2015-2020)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.tight_layout()
    plt.savefig('results/eda_time_series.png', dpi=300)
    plt.close()

    # 2. Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df['AQI'], bins=50, kde=True, color='#2ca02c')
    plt.title('Distribution of AQI Values', fontsize=14)
    plt.xlabel('AQI')
    plt.axvline(df['AQI'].mean(), color='red', linestyle='--', label=f'Mean: {df["AQI"].mean():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/eda_distribution.png', dpi=300)
    plt.close()

    # 3. Monthly Seasonality (Boxplot)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Month', y='AQI', data=df, palette='coolwarm')
    plt.title('Seasonal AQI Patterns (Monthly Distribution)', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('AQI')
    plt.tight_layout()
    plt.savefig('results/eda_seasonality.png', dpi=300)
    plt.close()

    # 4. Correlation Heatmap (if features exist)
    # Check for lag features
    cols_to_plot = ['AQI'] + [c for c in df.columns if 'lag' in c]
    if len(cols_to_plot) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[cols_to_plot].corr()
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
        plt.title('Correlation Matrix: AQI vs Lag Features', fontsize=14)
        plt.tight_layout()
        plt.savefig('results/eda_correlation.png', dpi=300)
        plt.close()
    
    print("EDA plots saved to 'results/' directory.")

if __name__ == "__main__":
    generate_eda()
