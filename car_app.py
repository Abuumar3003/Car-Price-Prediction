import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("car_prediction_model.pkl")

data = load_model()
model = data["model"]
feature_names = data["features"]

st.title("🚗 Car Price Prediction System")

st.sidebar.header("Enter Car Details")

year = st.sidebar.slider("Year", 2000, 2024, 2015)
present_price = st.sidebar.number_input("Present Price (Lakhs)", 0.0, 50.0, 5.0)
kms_driven = st.sidebar.number_input("Kms Driven", 0, 500000, 50000)

fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Owner", [0,1,2,3])

if st.sidebar.button("Predict Price"):

    car_age = 2024 - year

    input_dict = {
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Owner": owner,
        "Car_Age": car_age,
        f"Fuel_Type_{fuel}": 1,
        f"Seller_Type_{seller}": 1,
        f"Transmission_{transmission}": 1
    }

    # Initialize all features as 0
    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = 0

    # Fill matching features
    for key in input_dict:
        if key in input_data.columns:
            input_data.at[0, key] = input_dict[key]

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Selling Price: ₹{prediction:.2f} Lakhs")