import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Simple Styling
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    h1 {color: #4f46e5;}
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('car_prediction_model.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# Header
st.title("🚗 Car Price Prediction System")
st.markdown("### Get Instant Valuation for Your Used Car")

if model is None:
    st.error("Model file not found! Please train the model first.")
    st.stop()

# Sidebar Inputs
st.sidebar.title("Car Details")

year = st.sidebar.slider('Manufacturing Year', 2000, 2024, 2015)
present_price = st.sidebar.number_input('Current Ex-Showroom Price (Lakhs)', 0.0, 50.0, 5.0, 0.1)
kms_driven = st.sidebar.number_input('Kilometers Driven', 0, 500000, 50000, 1000)

fuel_type = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
seller_type = st.sidebar.selectbox('Seller Type', ['Dealer', 'Individual'])
transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.sidebar.selectbox('Number of Previous Owners', [0, 1, 2, 3])

# Predict Button
predict_btn = st.sidebar.button("Get Price Estimate", type="primary")

if predict_btn:
    
    # Encoding
    fuel_encoded = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}[fuel_type]
    seller_encoded = {'Dealer': 0, 'Individual': 1}[seller_type]
    transmission_encoded = {'Manual': 0, 'Automatic': 1}[transmission]

    input_data = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_encoded],
        'Seller_Type': [seller_encoded],
        'Transmission': [transmission_encoded],
        'Owner': [owner]
    })

    predicted_price = model.predict(input_data)[0]

    depreciation = present_price - predicted_price
    depreciation_percent = (depreciation / present_price) * 100 if present_price > 0 else 0

    st.markdown("---")
    st.header("Price Estimation Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Estimated Selling Price", f"₹{predicted_price:.2f} Lakhs")
    col2.metric("Current Showroom Price", f"₹{present_price:.2f} Lakhs")
    col3.metric("Total Depreciation", f"₹{depreciation:.2f} Lakhs", f"-{depreciation_percent:.1f}%")

    st.markdown("---")
    st.subheader("Price Position Indicator")

    if present_price > 0:
        percent_value = min(int((predicted_price / present_price) * 100), 100)
        st.progress(percent_value)
        st.write(f"Resale Value: {percent_value}% of original price")

    st.markdown("---")
    st.subheader("Car Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"Year: {year}")
        st.write(f"Kilometers Driven: {kms_driven:,} km")
        st.write(f"Fuel Type: {fuel_type}")

    with col2:
        st.write(f"Transmission: {transmission}")
        st.write(f"Seller Type: {seller_type}")
        st.write(f"Previous Owners: {owner}")

else:
    st.info("Enter details in sidebar and click **Get Price Estimate**")