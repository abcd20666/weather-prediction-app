import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# PAGE CONFIG (Dark Mode Style)
# -------------------------------
st.set_page_config(page_title="Weather App", page_icon="🌦️", layout="wide")

# Custom CSS for dark theme
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.stApp {
    background-color: #0e1117;
    color: white;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1c1f26;
    text-align: center;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
try:
    model = pickle.load(open("weather_model.pkl", "rb"))
except:
    st.error("Model file not found!")
    st.stop()

# Load dataset
try:
    df = pd.read_csv("seattle-weather.csv")
except:
    st.warning("Dataset not found! Charts disabled.")
    df = None

# -------------------------------
# HEADER
# -------------------------------
st.title("🌦️ Weather Prediction Dashboard")

# -------------------------------
# INPUT SECTION
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    precipitation = st.slider("Precipitation", 0.0, 50.0, 0.0)
    temp_max = st.slider("Max Temperature", -10.0, 50.0, 25.0)

with col2:
    temp_min = st.slider("Min Temperature", -10.0, 40.0, 15.0)
    wind = st.slider("Wind Speed", 0.0, 20.0, 5.0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Weather"):

    features = np.array([[precipitation, temp_max, temp_min, wind]])
    prediction = model.predict(features)[0]

    st.subheader("Prediction Result")

    # Card UI
    if prediction == "rain":
        st.markdown('<div class="card">🌧️ Rainy Weather<br>Carry umbrella ☔</div>', unsafe_allow_html=True)
    elif prediction == "sun":
        st.markdown('<div class="card">☀️ Sunny Weather<br>Enjoy your day 😎</div>', unsafe_allow_html=True)
    elif prediction == "fog":
        st.markdown('<div class="card">🌫️ Foggy Weather<br>Drive carefully 🚗</div>', unsafe_allow_html=True)
    elif prediction == "drizzle":
        st.markdown('<div class="card">🌦️ Drizzle<br>Light rain expected</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="card">🌍 {prediction}</div>', unsafe_allow_html=True)

    # -------------------------------
    # CHARTS (Only if dataset exists)
    # -------------------------------
    if df is not None:
        st.subheader("📊 Weather Insights")

        col1, col2 = st.columns(2)

        # Chart 1: Weather Distribution
        with col1:
            st.write("Weather Distribution")
            st.bar_chart(df["weather"].value_counts())

        # Chart 2: Temperature Trend
        with col2:
            st.write("Temperature Trend")
            st.line_chart(df[["temp_max", "temp_min"]])

        # Chart 3: Precipitation
        st.write("Precipitation Over Time")
        st.area_chart(df["precipitation"])

        # Chart 4: Wind Speed
        st.write("Wind Speed Trend")
        st.line_chart(df["wind"])
