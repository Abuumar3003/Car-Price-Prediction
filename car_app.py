import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

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
@st.cache_resource
def load_model():
    if os.path.exists("car_prediction_model.pkl"):
        return joblib.load("car_prediction_model.pkl")
    return None

data = load_model()

if data is not None:
    model = data["model"]
    feature_names = data["features"]
else:
    model = None
    feature_names = None

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
    <li>Uses dynamic feature alignment</li>
    <li>Production-ready deployment structure</li>
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
    <li>Pandas</li>
    <li>Joblib</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 🚗 PREDICTION PAGE
# =====================================================
elif page == "🚗 Predict Price":

    st.title("🚗 Predict Car Selling Price")

    if model is None:
        st.error("Model file not found.")
        st.stop()

    st.sidebar.header("Enter Car Details")

    year = st.sidebar.slider("Manufacturing Year", 2000, datetime.now().year, 2015)
    present_price = st.sidebar.number_input("Present Price (Lakhs)", 0.0, 50.0, 5.0)
    kms_driven = st.sidebar.number_input("Kms Driven", 0, 500000, 50000)

    fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.sidebar.selectbox("Owner", [0, 1, 2, 3])

    if st.sidebar.button("🚀 Predict Price"):

        car_age = datetime.now().year - year

        input_dict = {
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Owner": owner,
            "Car_Age": car_age,
            f"Fuel_Type_{fuel}": 1,
            f"Seller_Type_{seller}": 1,
            f"Transmission_{transmission}": 1
        }

        # Create empty dataframe with correct feature order
        input_data = pd.DataFrame(columns=feature_names)
        input_data.loc[0] = 0

        for key in input_dict:
            if key in input_data.columns:
                input_data.at[0, key] = input_dict[key]

        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.header("📊 Prediction Results")

        col1, col2 = st.columns(2)

        col1.metric("Estimated Selling Price", f"₹ {prediction:.2f} Lakhs")
        col2.metric("Car Age", f"{car_age} Years")

        resale_percent = (prediction / present_price * 100) if present_price > 0 else 0
        st.progress(min(int(resale_percent), 100))

# =====================================================
# 📊 MODEL INSIGHTS
# =====================================================
elif page == "📊 Model Insights":

    st.title("📊 Model Insights")

    if model is not None and hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("Feature importance not available.")

# =====================================================
# 📁 DATASET OVERVIEW
# =====================================================
elif page == "📁 Dataset Overview":

    st.title("📁 Dataset Overview")

    if os.path.exists("car_data.csv"):
        df = pd.read_csv("car_data.csv")
        st.dataframe(df.head(), use_container_width=True)
        st.write("Shape:", df.shape)
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.warning("Dataset not found.")

# =====================================================
# ℹ ABOUT PROJECT
# =====================================================
elif page == "ℹ About Project":

    st.title("ℹ About This Project")

    st.markdown("""
    ### Objective
    Predict used car selling price using Machine Learning.

    ### Methodology
    - Feature Engineering (Car Age)
    - One-hot Encoding
    - Random Forest Regression
    - Feature Alignment at Deployment

    ### Deployment Strategy
    - Separate Training Script
    - Model + Feature Names Saved Together
    - Streamlit Cloud Deployment
    """)

# =====================================================
# 👨‍💻 ABOUT DEVELOPER
# =====================================================
elif page == "👨‍💻 About Developer":

    st.title("👨‍💻 About Developer")

    st.markdown("""
    **M S Mohammad Abu Umar**  
    Artificial Intelligence & Data Science Student  

    Skills:
    - Python
    - Machine Learning
    - SQL
    - Data Analysis
    - Streamlit Deployment
    """)

# ---------------- FOOTER ---------------- #
st.markdown("""
<div class="footer">
© 2026 Car Price Prediction System | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)