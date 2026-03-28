import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Weather App", page_icon="🌦️")

# Custom CSS (Dark + Button Text Black)
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

/* Button text black */
div.stButton > button {
    color: black !important;
    background-color: #4CAF50;
    border-radius: 10px;
    height: 3em;
    width: 100%;
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
    df = None
    st.warning("Dataset not found. Charts disabled.")

# -------------------------------
# TITLE
# -------------------------------
st.title("🌦️ Weather Prediction App")

# -------------------------------
# INPUTS
# -------------------------------
precipitation = st.slider("Precipitation", 0.0, 50.0, 0.0)
temp_max = st.slider("Max Temperature", -10.0, 50.0, 25.0)
temp_min = st.slider("Min Temperature", -10.0, 40.0, 15.0)
wind = st.slider("Wind Speed", 0.0, 20.0, 5.0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Weather"):

    features = np.array([[precipitation, temp_max, temp_min, wind]])
    prediction = model.predict(features)[0]

    st.subheader("Prediction")

    if prediction == "rain":
        st.write("🌧️ Rainy Weather")
    elif prediction == "sun":
        st.write("☀️ Sunny Weather")
    elif prediction == "fog":
        st.write("🌫️ Foggy Weather")
    elif prediction == "drizzle":
        st.write("🌦️ Drizzle")
    else:
        st.write(prediction)

    # -------------------------------
    # SIMPLE CHARTS (3 ONLY)
    # -------------------------------
    if df is not None:
        st.subheader("📊 Charts")

        # 1. Weather Count
        st.write("Weather Distribution")
        st.bar_chart(df["weather"].value_counts())

        # 2. Temperature Chart
        st.write("Temperature Trend")
        st.line_chart(df[["temp_max", "temp_min"]])

        # 3. Precipitation Chart
        st.write("Precipitation Trend")
        st.area_chart(df["precipitation"])
