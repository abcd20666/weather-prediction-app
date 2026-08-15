# 🌦️ Weather Prediction App

A web-based **Weather Prediction App** that provides weather information and predicts upcoming weather conditions based on location and weather data.

The application provides a simple and user-friendly interface to check weather conditions and view useful weather insights.

---

## 📌 Project Overview

The Weather Prediction App is designed to provide users with weather information in an easy-to-understand format.

Users can enter a location and view weather details such as temperature, humidity, wind speed, and weather conditions.

---

## 🚀 Try the Project

🔗 **[Try Weather Prediction App](https://weather-prediction-app-59vm6kneecgyfoqtdpyqe3.streamlit.app/)**

---

## 🎥 Project Demo

▶️ **[Watch Weather Prediction App Demo](https://drive.google.com/file/d/1l3yY8mdqZzv7rb98DfS6j0xXX6TLTUJv/view?usp=sharing)**

---

## 🎯 Objectives

- 🌤️ Provide weather information based on location.
- 🌡️ Display temperature and weather conditions.
- 💧 Show humidity and other weather parameters.
- 💨 Display wind speed information.
- 📊 Provide weather data in an easy-to-understand format.
- 🖥️ Provide a simple and user-friendly interface.

---

## ✨ Key Features

- 🌍 Location-based weather information
- 🌡️ Temperature display
- 💧 Humidity information
- 💨 Wind speed details
- ☁️ Weather condition display
- 📊 Weather data visualization
- 🔍 Search weather by location
- 🖥️ User-friendly interface
- 📱 Responsive design

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Weather API

---

## 🏗️ System Architecture

```text
                    👤 User
                      │
                      ▼
              ┌───────────────┐
              │   Streamlit   │
              │  Web Interface│
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  User Location│
              │     Input     │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Weather Data  │
              │   / API       │
              └───────┬───────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
        Weather Analysis   Prediction
              │                │
              └───────┬────────┘
                      ▼
              📊 Weather Results
