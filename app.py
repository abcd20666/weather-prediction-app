import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("../model/weather_model.pkl", "rb"))

st.title("🌦️ Weather Prediction App")

st.write("Enter weather details:")

precipitation = st.number_input("Precipitation")
temp_max = st.number_input("Max Temperature")
temp_min = st.number_input("Min Temperature")
wind = st.number_input("Wind Speed")

if st.button("Predict Weather"):
    features = np.array([[precipitation, temp_max, temp_min, wind]])
    prediction = model.predict(features)

    st.success(f"Predicted Weather: {prediction[0]}")
