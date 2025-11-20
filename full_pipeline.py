import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# "C:/Users/omcha/AppData/Local/Microsoft/WindowsApps/python3.11.exe" "d:/Cryptocurrency Liquidity/run_app.py" -- to run the site
# 1. Setup Environment
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
print(f'Working directory: {current_dir}')

# 2. Load Raw Data
print('Loading raw data...')
file_16_path = os.path.join(current_dir, 'coin_gecko_2022-03-16.csv')
file_17_path = os.path.join(current_dir, 'coin_gecko_2022-03-17.csv')

try:
    df_16 = pd.read_csv(file_16_path)
    df_17 = pd.read_csv(file_17_path)
    df_combined = pd.concat([df_16, df_17], ignore_index=True)
except FileNotFoundError:
    print('Error: CSV files not found.')
    print(f'Looking for: {file_16_path}')
    exit()
except OSError as e:
    print(f'Error: {e}')
    exit()

# 3. Data Cleaning & Imputation
print('Cleaning data...')
df_combined['date'] = pd.to_datetime(df_combined['date'])
df_combined.drop_duplicates(inplace=True)

numerical_cols = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap']

for col in numerical_cols:
    df_combined[col] = df_combined[col].fillna(df_combined.groupby('symbol')[col].transform('mean'))
    df_combined[col] = df_combined[col].fillna(df_combined[col].mean())

# 4. Fit & Save Scaler
print('Fitting and saving Scaler...')
scaler = StandardScaler()
scaler.fit(df_combined[numerical_cols])

with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 5. Transform Data
df_scaled = df_combined.copy()
df_scaled[numerical_cols] = scaler.transform(df_combined[numerical_cols])

# 6. Feature Engineering
print('Creating features...')
df_scaled['target_liquidity_volume'] = df_scaled.groupby('symbol')['24h_volume'].shift(-1)
df_scaled.dropna(inplace=True)

# 7. Train Model
print('Training Random Forest Model...')
drop_cols = ['symbol', 'target_liquidity_volume', 'date', 'coin']
existing_drop_cols = [c for c in drop_cols if c in df_scaled.columns]
X = df_scaled.drop(columns=existing_drop_cols)
Y = df_scaled['target_liquidity_volume']

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, Y)

# 8. Save Model
print('Saving Model...')
with open('final_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Success! final_rf_model.pkl and feature_scaler.pkl have been generated.')
