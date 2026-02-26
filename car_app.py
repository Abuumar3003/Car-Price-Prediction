import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Car Price Prediction Pro",
    page_icon="🚗",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.main {padding: 1rem;}
h1 {color: #1e3a8a; text-align: center;}
h2 {color: #2563eb;}
.section-card {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    padding: 20px;
    font-size: 14px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
# ---------------- TRAIN MODEL (CLOUD SAFE) ---------------- #
@st.cache_resource
def load_model():
    from sklearn.ensemble import RandomForestRegressor
    
    if not os.path.exists("car_data.csv"):
        return None

    df = pd.read_csv("car_data.csv")

    # Split features & target
    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)
    return model

model = load_model()

# ---------------- SIDEBAR NAVIGATION ---------------- #
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🚗 Predict Price", "📊 Model Insights", "📁 Dataset Overview", "ℹ About Project", "👨‍💻 About Developer"]
)

# =====================================================
# 🏠 HOME PAGE
# =====================================================
if page == "🏠 Home":

    st.title("🚗 Car Price Prediction System")
    st.markdown("### AI Powered Used Car Valuation Platform")

    st.markdown("""
    <div class="section-card">
    <h3>📌 What This App Does</h3>
    <ul>
    <li>Predicts used car selling price using Machine Learning</li>
    <li>Calculates depreciation</li>
    <li>Shows resale value percentage</li>
    <li>Provides car summary insights</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
    <h3>⚙ Technology Stack</h3>
    <ul>
    <li>Python</li>
    <li>Streamlit</li>
    <li>Scikit-learn</li>
    <li>Pandas & NumPy</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 🚗 PREDICTION PAGE
# =====================================================
elif page == "🚗 Predict Price":

    st.title("🚗 Predict Car Selling Price")

    if model is None:
        st.error("Model file not found. Please ensure car_prediction_model.pkl exists.")
        st.stop()

    st.sidebar.header("Enter Car Details")

    year = st.sidebar.slider("Manufacturing Year", 2000, 2024, 2015)
    present_price = st.sidebar.number_input("Current Ex-Showroom Price (Lakhs)", 0.0, 50.0, 5.0, 0.1)
    kms_driven = st.sidebar.number_input("Kilometers Driven", 0, 500000, 50000, 1000)

    fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 2, 3])

    predict_btn = st.sidebar.button("🚀 Predict Price")

    if predict_btn:

        encoding = {
            "Fuel_Type": {"Petrol": 0, "Diesel": 1, "CNG": 2},
            "Seller_Type": {"Dealer": 0, "Individual": 1},
            "Transmission": {"Manual": 0, "Automatic": 1}
        }

        input_df = pd.DataFrame({
            "Year": [year],
            "Present_Price": [present_price],
            "Kms_Driven": [kms_driven],
            "Fuel_Type": [encoding["Fuel_Type"][fuel_type]],
            "Seller_Type": [encoding["Seller_Type"][seller_type]],
            "Transmission": [encoding["Transmission"][transmission]],
            "Owner": [owner]
        })

        predicted_price = model.predict(input_df)[0]
        depreciation = present_price - predicted_price
        depreciation_percent = (depreciation / present_price * 100) if present_price > 0 else 0

        st.markdown("---")
        st.header("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Price", f"₹ {predicted_price:.2f} Lakhs")
        col2.metric("Original Price", f"₹ {present_price:.2f} Lakhs")
        col3.metric("Depreciation", f"₹ {depreciation:.2f}", f"-{depreciation_percent:.1f}%")

        st.progress(min(int((predicted_price / present_price) * 100), 100) if present_price > 0 else 0)

# =====================================================
# 📊 MODEL INSIGHTS
# =====================================================
elif page == "📊 Model Insights":

    st.title("📊 Model Insights")

    if model is not None and hasattr(model, "feature_importances_"):
        features = ["Year", "Present_Price", "Kms_Driven", "Fuel_Type",
                    "Seller_Type", "Transmission", "Owner"]

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("Feature importance not available for this model.")

# =====================================================
# 📁 DATASET OVERVIEW
# =====================================================
elif page == "📁 Dataset Overview":

    st.title("📁 Dataset Overview")

    if os.path.exists("car_data.csv"):
        df = pd.read_csv("car_data.csv")
        st.write("### Sample Data")
        st.dataframe(df.head(), use_container_width=True)

        st.write("### Dataset Shape")
        st.write(df.shape)

        st.write("### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.warning("Dataset file not found.")

# =====================================================
# ℹ ABOUT PROJECT
# =====================================================
elif page == "ℹ About Project":

    st.title("ℹ About This Project")

    st.markdown("""
    ### 🎯 Objective
    Build a Machine Learning model to predict used car prices accurately.

    ### 📌 Problem Statement
    Used car pricing is complex due to multiple influencing factors like:
    - Age of vehicle
    - Fuel type
    - Transmission
    - Ownership history
    - Market demand

    ### 🤖 Machine Learning Approach
    - Data preprocessing
    - Feature encoding
    - Model training using Regression
    - Evaluation & deployment

    ### 🚀 Deployment
    - Streamlit Web Application
    - Model serialized using Joblib
    """)

# =====================================================
# 👨‍💻 ABOUT DEVELOPER
# =====================================================
elif page == "👨‍💻 About Developer":

    st.title("👨‍💻 About Developer")

    st.markdown("""
    ### Abu Umar
    Artificial Intelligence & Data Science Student  

    ### 💡 Skills
    - Python
    - Machine Learning
    - SQL
    - Data Analysis
    - Streamlit App Development

    ### 🎯 Goal
    To build AI-driven intelligent systems and real-world ML applications.
    """)

# ---------------- FOOTER ---------------- #
st.markdown("""
<div class="footer">
© 2026 Car Price Prediction System | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)