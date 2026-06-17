# Streamlit code here
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("crop_yield_model.pkl")

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Crop Yield Prediction System")
st.write(
    "Predict crop yield using climatic and agricultural features."
)

# Crop list
crops = [
    'Arecanut', 'Arhar/Tur', 'Castor seed', 'Coconut ',
    'Cotton(lint)', 'Dry chillies', 'Gram', 'Jute',
    'Linseed', 'Maize', 'Mesta', 'Niger seed',
    'Onion', 'Other  Rabi pulses', 'Potato',
    'Rapeseed &Mustard', 'Rice', 'Sesamum',
    'Small millets', 'Sugarcane', 'Sweet potato',
    'Tapioca', 'Tobacco', 'Turmeric', 'Wheat',
    'Bajra', 'Black pepper', 'Cardamom',
    'Coriander', 'Garlic', 'Ginger', 'Groundnut',
    'Horse-gram', 'Jowar', 'Ragi', 'Cashewnut',
    'Banana', 'Soyabean', 'Barley', 'Khesari',
    'Masoor', 'Moong(Green Gram)',
    'Other Kharif pulses', 'Safflower',
    'Sannhamp', 'Sunflower', 'Urad',
    'Peas & beans (Pulses)', 'other oilseeds',
    'Other Cereals', 'Cowpea(Lobia)',
    'Oilseeds total', 'Guar seed',
    'Other Summer Pulses', 'Moth'
]

states = [
    'Assam', 'Karnataka', 'Kerala', 'Meghalaya',
    'West Bengal', 'Puducherry', 'Goa',
    'Andhra Pradesh', 'Tamil Nadu', 'Odisha',
    'Bihar', 'Gujarat', 'Madhya Pradesh',
    'Maharashtra', 'Mizoram', 'Punjab',
    'Uttar Pradesh', 'Haryana',
    'Himachal Pradesh', 'Tripura',
    'Nagaland', 'Chhattisgarh',
    'Uttarakhand', 'Jharkhand',
    'Delhi', 'Manipur',
    'Jammu and Kashmir',
    'Telangana',
    'Arunachal Pradesh',
    'Sikkim'
]

seasons = [
    "Whole Year",
    "Kharif",
    "Rabi",
    "Summer",
    "Winter",
    "Autumn"
]

crop = st.selectbox("Crop", crops)
state = st.selectbox("State", states)
season = st.selectbox("Season", seasons)

crop_year = st.number_input(
    "Crop Year",
    min_value=1997,
    max_value=2035,
    value=2025
)

area = st.number_input(
    "Area",
    min_value=0.0,
    value=100.0
)

annual_rainfall = st.number_input(
    "Annual Rainfall (mm)",
    min_value=0.0,
    value=1000.0
)

fertilizer = st.number_input(
    "Fertilizer Usage",
    min_value=0.0,
    value=1000.0
)

pesticide = st.number_input(
    "Pesticide Usage",
    min_value=0.0,
    value=100.0
)

avg_temp = st.number_input(
    "Average Temperature",
    value=25.0
)

max_temp = st.number_input(
    "Maximum Temperature",
    value=35.0
)

min_temp = st.number_input(
    "Minimum Temperature",
    value=15.0
)

if st.button("Predict Yield"):

    input_df = pd.DataFrame({
        "Crop": [crop],
        "Crop_Year": [crop_year],
        "Season": [season],
        "State": [state],
        "Area": [area],
        "Annual_Rainfall": [annual_rainfall],
        "Fertilizer": [fertilizer],
        "Pesticide": [pesticide],
        "Avg_Temperature": [avg_temp],
        "Max_Temperature": [max_temp],
        "Min_Temperature": [min_temp]
    })

    pred_log = model.predict(input_df)[0]

    predicted_yield = np.expm1(pred_log)

    st.success(
        f"Predicted Yield: {predicted_yield:.2f}"
    )
