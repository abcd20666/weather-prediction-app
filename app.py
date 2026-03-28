import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Weather App", page_icon="🌦️")

# -------------------------------
# CUSTOM CSS (Dark + Button Style)
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

/* Button styling */
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

# -------------------------------
# LOAD DATASET
# -------------------------------
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
# INPUT SECTION
# -------------------------------
st.subheader("Enter Weather Details")

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

    st.subheader("Prediction Result")

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
    # CLEAN SIMPLE CHARTS
    # -------------------------------
    if df is not None:
        st.subheader("📊 Charts")

        # Use only first 50 rows (clean display)
        df_small = df.head(50)

        # 1. Weather Distribution
        st.write("Weather Distribution")
        st.bar_chart(df["weather"].value_counts())

        # 2. Temperature Trend
        st.write("Temperature Trend (First 50 Days)")
        st.line_chart(df_small[["temp_max", "temp_min"]])

        # 3. Precipitation Trend
        st.write("Precipitation Trend (First 50 Days)")
        st.line_chart(df_small["precipitation"])
