import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import holidays
import datetime
from catboost import CatBoostRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DC Bike Predictor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CREATIVE RED & WHITE THEME CSS ---
st.markdown("""
    <style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        color: #333333;
    }
    
    /* Main Background */
    .stApp {
        background-color: #ffffff;
        background-image: radial-gradient(#f2f2f2 1px, transparent 1px);
        background-size: 20px 20px;
    }

    /* Headlines */
    h1, h2,h3 {
        color: #D32F2F !important;
        font-weight: 900 !important;
        letter-spacing: -1px;
    }
    
    /* Metric Cards (Top Row) */
    .metric-card {
        background-color: #ffffff;
        border-left: 4px solid #D32F2F; /* Red Accent */
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(211, 47, 47, 0.15);
    }
    .metric-title {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #D32F2F;
        font-size: 2rem;
        font-weight: 900;
    }

    /* Prediction Hero Card (Main Result) */
    .prediction-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #FF5252 0%, #D32F2F 100%);
        width: 100%;
        padding: 40px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 50px rgba(211, 47, 47, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    /* Decorative Circle inside Card */
    .prediction-card::before {
        content: "";
        position: absolute;
        top: -50px;
        right: -50px;
        width: 150px;
        height: 150px;
        background: rgba(255,255,255,0.1);
        border-radius: 50%;
    }

    .pred-number {
        font-size: 6rem;
        font-weight: 900;
        margin: 0;
        color:white;
        line-height: 1.1;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    .pred-label {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        background: rgba(255,255,255,0.2);
        display: inline-block;
        padding: 5px 20px;
        border-radius: 50px;
        margin-top: 10px;
    }
    
    /* Sidebar styling tweaks */
    section[data-testid="stSidebar"] {
        background-color: #f9f9f9;
        border-right: 1px solid #eee;
    }
    
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL (Keep logic identical) ---
@st.cache_resource
def load_assets():
    features = joblib.load('features_champion.pkl')
    model = CatBoostRegressor()
    model.load_model('catboost_champion.cbm')
    return model, features

try:
    model, feature_list = load_assets()
except Exception as e:
    st.error(f" Error loading model files. Please check paths. Details: {e}")
    st.stop()

# --- HEADER ---
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("<h1>WASHINGTON DC <span style='color:#333'>BIKESHARE AI</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.1rem; color:#666;'>Intelligent Mobility & Demand Forecasting System</p>", unsafe_allow_html=True)

st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("<h2 style='color:#D32F2F;'>Configuration</h2>", unsafe_allow_html=True)

# 1. Date & Time Selection
st.sidebar.subheader("1. Timeline")
selected_date = st.sidebar.date_input("Select Date", datetime.date.today())
selected_hour = st.sidebar.slider("Select Hour (24h)", 0, 23, 8)

# 2. Weather Data Source
st.sidebar.subheader("2. Atmosphere")
use_api = st.sidebar.toggle("Live Satellite Data", value=True)

# Initialize weather variables
temp, precip, humidity, wind = 20.0, 0.0, 60.0, 10.0
api_success = False

if use_api:
    try:
        # Washington DC Coordinates
        lat, lon = 38.9072, -77.0369
        date_str = selected_date.strftime("%Y-%m-%d")
        
        # Open-Meteo API
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&start_date={date_str}&end_date={date_str}"
        
        r = requests.get(url).json()
        
        # Extract hourly data
        temp = r['hourly']['temperature_2m'][selected_hour]
        humidity = r['hourly']['relative_humidity_2m'][selected_hour]
        precip = r['hourly']['precipitation'][selected_hour]
        wind = r['hourly']['wind_speed_10m'][selected_hour]
        
        api_success = True
        st.sidebar.success("Data Synced")
    except:
        st.sidebar.warning("API unavailable. Manual Mode.")

if not api_success:
    temp = st.sidebar.slider("Temperature (°C)", -10.0, 40.0, 20.0)
    precip = st.sidebar.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
    wind = st.sidebar.slider("Wind Speed (km/h)", 0, 100, 10)

# --- BACKEND CALCULATIONS ---

# 1. Temporal Features from Date
year = selected_date.year
month = selected_date.month
day_of_week = selected_date.weekday() # 0=Mon, 6=Sun

# 2. Holiday Logic (Automatic)
us_holidays = holidays.US()
is_holiday = 1 if selected_date in us_holidays else 0

# 3. Weekend Logic
is_weekend = 1 if day_of_week >= 5 else 0

# 4. Rush Hour Logic
is_rush_hour = 0

if (7 <= selected_hour <= 9) or (16 <= selected_hour <= 19):
    is_rush_hour = 1

# 5. Cyclic Features (Sin/Cos)
hour_sin = np.sin(2 * np.pi * selected_hour / 24)
hour_cos = np.cos(2 * np.pi * selected_hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
day_sin = np.sin(2 * np.pi * day_of_week / 7)
day_cos = np.cos(2 * np.pi * day_of_week / 7)

# --- DASHBOARD METRICS ---

# Row 1: Context Data
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Temp</div>
            <div class="metric-value">{temp}°</div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Weather</div>
            <div class="metric-value">{'Rain' if precip > 0 else 'Clear'}</div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">DAY</div>
            <div class="metric-value">{'Off' if is_holiday or is_weekend else 'Work'}</div>
        </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Context</div>
            <div class="metric-value">{'Rush' if is_rush_hour else 'Std'}</div>
        </div>
    """, unsafe_allow_html=True)

# --- PREDICTION ENGINE ---

# Prepare Data Frame
input_data = pd.DataFrame({
    'year': [year],
    'is_weekend': [is_weekend],
    'is_rush_hour': [is_rush_hour],
    'is_holiday': [is_holiday],
    'hour_sin': [hour_sin],
    'hour_cos': [hour_cos],
    'month_sin': [month_sin],
    'month_cos': [month_cos],
    'day_sin': [day_sin],
    'day_cos': [day_cos],
    'temp': [temp],
    'precip': [precip],
    'humidity': [humidity],
    'windspeed': [wind]
})

# Ensure exact column order
input_data = input_data[feature_list]

# Predict
pred_log = model.predict(input_data)[0]
pred_real = np.expm1(pred_log) # Inverse Log1p
final_prediction = int(max(0, pred_real))

# --- MAIN RESULT DISPLAY (RED THEME) ---
st.markdown("<br>", unsafe_allow_html=True)

col_spacer, col_main, col_spacer2 = st.columns([1, 6, 1])

with col_main:
    status_text = "Standard Flow"
    if final_prediction > 1000:
        status_text = "Maximum Saturation"
    elif final_prediction > 500:
        status_text = "High Demand"
        
    st.markdown(f"""
        <div class="prediction-container">
            <div class="prediction-card">
                <p style="opacity:0.9; font-size: 1.1rem; margin-bottom: 10px;">PREDICTED BIKE RENTALS</p>
                <div style="color:white;" class="pred-number">{final_prediction}</div>
                <div class="pred-label">{status_text}</div>
                <p style="margin-top: 25px; font-size: 0.9rem; opacity: 0.8;">
                    Forecast target: {selected_date.strftime('%B %d, %Y')} @ {selected_hour}:00
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- FOOTER / DEBUG ---
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander(" System Logs & Model Input"):
    st.write("Current Input Vector:")
    st.dataframe(input_data)
    st.info(f"Model: CatBoost Regressor ")