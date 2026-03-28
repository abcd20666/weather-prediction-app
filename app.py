import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
try:
    model = pickle.load(open("weather_model.pkl", "rb"))
except:
    st.error("Model file not found!")
    st.stop()

# Load dataset for graphs
df = pd.read_csv("seattle-weather.csv")

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
# GRAPHS SECTION
# -------------------------------
st.subheader("📊 Data Visualization")

# 1. Weather Count Chart
st.write("Weather Distribution")
weather_counts = df['weather'].value_counts()

fig1 = plt.figure()
weather_counts.plot(kind='bar')
st.pyplot(fig1)

# 2. Temperature vs Weather
st.write("Max Temperature vs Weather")

fig2 = plt.figure()
for w in df['weather'].unique():
    subset = df[df['weather'] == w]
    plt.scatter(subset['temp_max'], [w]*len(subset))
st.pyplot(fig2)

# 3. Precipitation Distribution
st.write("Precipitation Distribution")

fig3 = plt.figure()
df['precipitation'].plot(kind='hist')
st.pyplot(fig3)
