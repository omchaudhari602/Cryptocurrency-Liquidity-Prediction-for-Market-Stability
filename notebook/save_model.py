
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- Load Data and Prepare ---
# Files are now loaded from the same directory as the script.
try:
    df_ml = pd.read_csv('"D:\Cryptocurrency Liquidity\notebook\cryptocurrency_ml_ready.csv"')
    df_processed = pd.read_csv('"D:\Cryptocurrency Liquidity\notebook\cryptocurrency_preprocessed.csv"')
except FileNotFoundError as e:
    print(f"FATAL ERROR: Missing critical file. Ensure '{e.filename}' is in the D:\\Cryptocurrency Liquidity directory.")
    exit()

X = df_ml.drop(columns=['symbol', 'target_liquidity_volume'])
Y = df_ml['target_liquidity_volume']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 1. Train the Final Model
final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
final_model.fit(X_train, Y_train)

# 2. Fit the StandardScaler
numerical_cols = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap']
scaler = StandardScaler()
scaler.fit(df_processed[numerical_cols])


# 3. Save Artifacts (Saved to the main D:\Cryptocurrency Liquidity\ folder)
with open('final_rf_model.pkl', 'wb') as file:
    pickle.dump(final_model, file)

with open('feature_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("\nSUCCESS: Model and Scaler files have been created. Proceed to run Streamlit.")