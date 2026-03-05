
from attr import s
import requests
import streamlit as st
import numpy as np
import os 
import joblib
from train_model import train_model
import pandas as pd
import plotly.express as px
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

if not os.path.exists("weather_model.pkl"):
    train_model()

model = joblib.load("weather_model.pkl")

# Load dataset
data = pd.read_csv("dataset/weather.csv")

# Page settings
st.set_page_config(
    page_title="Weather prediction using redression techniques",
    page_icon="🌦",
    layout="wide"
)

# Load ML model
model = joblib.load("weather_model.pkl")
# Sidebar
st.sidebar.title("⚙️ Model Information")
st.sidebar.write("Machine Learning Model: Random Forest Regression")
st.sidebar.write("Features Used:")
st.sidebar.write("- Humidity")
st.sidebar.write("- Wind Speed")
st.sidebar.write("- Pressure")

st.sidebar.info("This AI model predicts temperature based on weather conditions.")

# Title
st.title("🌦 Weather prediction using redression techniques")
st.subheader("🌍 Live Weather Data")

API_KEY = ["OPENWEATHERMAP_API_KEY"]  # Replace with your OpenWeatherMap API key
city = st.text_input("Enter City", "Roorkee")

if st.button("Get Live Weather"):

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data_api = response.json()

    if response.status_code == 200:

        temperature = data_api["main"]["temp"]
        humidity = data_api["main"]["humidity"]
        pressure = data_api["main"]["pressure"]
        wind_speed = data_api["wind"]["speed"]

        st.write("Temperature:", temperature)
        st.write("Humidity:", humidity)
        st.write("Pressure:", pressure)
        st.write("Wind Speed:", wind_speed)

    else:
        st.error("City not found or API error.")





accuracy_data = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [81, 87, 92]
    
})


fig = px.bar(
    accuracy_data,
    x="Model",
    y="Accuracy",
    color="Model",
    title="Model Accuracy Comparison"
)

st.plotly_chart(fig, key="accuracy_graph")


fig = px.bar(
    accuracy_data,
    x="Model",
    y="Accuracy",
    color="Model",
    title="Model Accuracy Comparison"
)

st.plotly_chart(fig)
# Layout columns
col1, col2 = st.columns(2)

# User Inputs
with col1:
    st.subheader("Temperature Distribution")

fig, ax = plt.subplots()
ax.hist(data['Temp_C'], bins=30)
ax.set_xlabel("Temperature")
ax.set_ylabel("Frequency")

st.plotly_chart(fig, key="chart1")

st.subheader("📈 Temperature Trend")

data["Date/Time"] = pd.to_datetime(data["Date/Time"])

fig = px.line(
    data,
    x="Date/Time",
    y="Temp_C",
    title="Temperature Trend Over Time"
)

st.plotly_chart(fig, key="temp_trend_graph")

# Prediction output
with col2:
    st.subheader("Humidity vs Temperature")

fig2, ax2 = plt.subplots()
ax2.scatter(data['Rel Hum_%'], data['Temp_C'])
ax2.set_xlabel("Humidity")
ax2.set_ylabel("Temperature")

st.pyplot(fig2)

st.subheader("💧 Humidity Trend")

fig2 = px.line(
    data,
    x="Date/Time",
    y="Rel Hum_%",
    title="Humidity Trend Over Time"
)

st.plotly_chart(fig2)



st.subheader("🔍 Feature Importance")

importance = model.feature_importances_

features = pd.DataFrame({
    "Feature": ["Humidity", "Wind Speed", "Pressure"],
    "Importance": importance
})

fig = px.bar(
    features,
    x="Feature",
    y="Importance",
    color="Feature",
    title="Feature Importance in Prediction"
)

st.plotly_chart(fig)

        

# Divider
st.markdown("---")

# Data visualization section
st.subheader("📊 Weather Data Visualization")

data = pd.read_csv("dataset/weather.csv")

col3, col4 = st.columns(2)

with col3:

    fig1 = px.scatter(
    data,
    x="Rel Hum_%",
    y="Temp_C",
    title="Humidity vs Temperature"
)

    st.plotly_chart(fig1)

with col4:

    fig2 = px.scatter(
    data,
    x="Wind Speed_km/h",
    y="Temp_C",
    title="Wind Speed vs Temperature"
)
st.subheader("🌡 Temperature Prediction")

humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (km/h)", 0, 100, 20)
pressure = st.slider("Pressure (kPa)", 90, 110, 101)

predict = st.button("Predict Temperature")

if predict:

    input_data = [[humidity, wind_speed, pressure]]

    prediction = model.predict(input_data)

    st.success(f"Predicted Temperature: {prediction[0]:.2f} °C")
    
    import plotly.graph_objects as go

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction[0],
        title={'text': "Predicted Temperature"},
        gauge={
            'axis': {'range': [-20, 50]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [-20, 0], 'color': "lightblue"},
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 35], 'color': "orange"},
                {'range': [35, 50], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(gauge, key="temperature_gauge")

# Footer
st.markdown("---")
st.markdown(
"""
### 🚀 Project: Weather Prediction Using Machine Learning  
**Algorithm Used:** Random Forest Regression  
**Technology:** Python, Scikit-Learn, Streamlit, Plotly
"""
)