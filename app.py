import streamlit as st
import pandas as pd
import pickle
import os

# --- Page Configuration ---
st.set_page_config(page_title="Crypto Liquidity Predictor", layout="centered")

# --- Robust File Loading Function ---
@st.cache_resource
def load_artifacts():
    # 1. Try to get the path from __file__ (works in standard scripts)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 2. Fallback: Use current working directory (works in Jupyter/Interactive)
        current_dir = os.getcwd()
        print("DEBUG: __file__ not found, using os.getcwd() instead.")
    
    # Construct absolute paths
    model_path = os.path.join(current_dir, 'final_rf_model.pkl')
    scaler_path = os.path.join(current_dir, 'feature_scaler.pkl')
    
    print(f"DEBUG: Looking for model at: {model_path}")
    
    # Load the model
    if not os.path.exists(model_path):
        # LAST RESORT: Try looking in the specific folder if relative paths fail
        # (Useful if you are running from C:\Users... but files are in D:\...)
        hardcoded_dir = r"D:\Cryptocurrency Liquidity"
        model_path = os.path.join(hardcoded_dir, 'final_rf_model.pkl')
        scaler_path = os.path.join(hardcoded_dir, 'feature_scaler.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler

# --- Main App Logic ---
try:
    model, scaler = load_artifacts()
except FileNotFoundError as e:
    st.error(f"🚨 Error: {e}")
    st.info("Please ensure 'final_rf_model.pkl' and 'feature_scaler.pkl' are in the correct folder.")
    st.stop()

# --- App UI ---
st.title("🪙 Cryptocurrency Liquidity Predictor")
st.markdown("""
This tool uses a Random Forest model to predict the **next day's trading volume** (liquidity) 
based on current market metrics.
""")

st.divider()

st.header("Input Current Market Data")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Price (USD)", value=40000.0, step=10.0)
    change_1h = st.number_input("1h Change (Decimal)", value=0.022, format="%.4f", help="0.01 = 1%")
    change_24h = st.number_input("24h Change (Decimal)", value=0.030, format="%.4f")

with col2:
    change_7d = st.number_input("7d Change (Decimal)", value=0.055, format="%.4f")
    volume_24h = st.number_input("Current 24h Volume (USD)", value=35000000000.0, step=1000000.0)
    mkt_cap = st.number_input("Market Cap (USD)", value=700000000000.0, step=1000000.0)

# --- Prediction Logic ---
if st.button("Predict Next Day Liquidity", type="primary"):
    feature_names = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap']
    input_df = pd.DataFrame([[price, change_1h, change_24h, change_7d, volume_24h, mkt_cap]], 
                            columns=feature_names)
    
    input_scaled = scaler.transform(input_df)
    pred_scaled = model.predict(input_scaled)[0]
    
    # Inverse transform logic
    vol_index = 4 
    vol_mean = scaler.mean_[vol_index]
    vol_scale = scaler.scale_[vol_index]
    pred_dollar = (pred_scaled * vol_scale) + vol_mean
    
    st.divider()
    st.subheader("Prediction Results")
    
    c1, c2 = st.columns(2)
    c1.metric(label="Predicted Volume (USD)", value=f"${pred_dollar:,.2f}")
    c2.metric(label="Model Score (Z-Score)", value=f"{pred_scaled:.4f}")
    
    diff = pred_dollar - volume_24h
    if diff > 0:
        st.success(f"📈 The model predicts an INCREASE in liquidity (+${diff:,.2f})")
    else:
        st.warning(f"📉 The model predicts a DECREASE in liquidity (-${abs(diff):,.2f})")