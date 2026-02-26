import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
PRIMARY_COLOR = "#1E3A8A"      # Dark Blue
SECONDARY_COLOR = "#0EA5E9"    # Sky Blue
ACCENT_COLOR = "#F97316"       # Orange
DARK_BG = "#0F172A"            # Very Dark Blue
LIGHT_BG = "#F8FAFC"           # LightSlate

MODEL_PATH = "car_prediction_model.pkl"


def apply_styles() -> None:
    """Inject refined professional CSS with minimal colors."""
    st.markdown(
        """
        <style>
            /* Base styling */
            body {
                background: #f9fafb;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #2c3e50;
            }
            .main {
                background-color: #ffffff;
                padding: 2rem 1rem;
            }
            h1, h2, h3, h4, h5 {
                color: #1e293b;
                font-weight: 600;
            }

            /* Card containers */
            .page-card {
                background-color: #ffffff;
                border-radius: 12px;
                padding: 28px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                margin-bottom: 20px;
            }

            /* Highlighted prediction box */
            .prediction-box {
                background: #1e3a8a;
                color: #ffffff;
                padding: 24px;
                border-radius: 12px;
                text-align: center;
                font-size: 1.8em;
                font-weight: 600;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15);
                margin: 20px 0;
            }

            /* Info and success messages */
            .info-box {
                background-color: #f1f5f9;
                padding: 14px;
                border-left: 4px solid #1e3a8a;
                border-radius: 6px;
                margin: 15px 0;
            }
            .success-box {
                background-color: #ecfdf5;
                padding: 14px;
                border-left: 4px solid #10b981;
                border-radius: 6px;
                margin: 15px 0;
            }

            /* Buttons */
            .stButton>button {
                background-color: #1e3a8a;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 0.5rem 1.4rem;
                font-size: 1rem;
                font-weight: 500;
                transition: background-color .2s ease;
            }
            .stButton>button:hover {
                background-color: #163667;
                cursor: pointer;
            }

            /* Inputs */
            .stTextInput>div>input,
            .stNumberInput>div>input,
            .stSelectbox>div>div {
                border-radius: 8px;
                border: 1px solid #d1d5db;
                padding: 0.4rem 0.6rem;
                box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
            }
            .stSlider>div>div>div>span {
                background-color: #1e3a8a !important;
            }

            /* Sidebar */
            .css-1lcbmhc .sidebar-content,
            .css-1v0mbdj {
                background-color: #ffffff;
                padding: 1rem 0;
            }
            .stMarkdown h1 {
                color: #1e293b;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_model(path: str):
    """Attempt to load the trained model from disk."""
    try:
        model_ = joblib.load(path)
        st.sidebar.markdown(
            "<div class='success-box'>✅ Model loaded successfully</div>",
            unsafe_allow_html=True,
        )
        return model_
    except Exception:
        st.sidebar.error("⚠️ Model file not found. Train the model first.")
        return None


def page_home():
    """Render the home/landing page with a professional design."""
    st.markdown("<div class='page-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-bottom: 1rem;'>Welcome to AutoValue Pro</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class='info-box'>
                <h3>How it works</h3>
                <ul style='margin: 0; padding-left: 1.2rem;'>
                    <li>Provide your car's specifications</li>
                    <li>Algorithm analyses them</li>
                    <li>Receive an estimated resale price</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='info-box'>
                <h3>Key features</h3>
                <ul style='margin: 0; padding-left: 1.2rem;'>
                    <li>Random Forest regressor</li>
                    <li>High accuracy on test data</li>
                    <li>Real-time interaction</li>
                    <li>Comprehensive analytics</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style='background: #1e3a8a; 
                        padding: 50px 20px; border-radius: 12px; text-align: center; 
                        color: white; min-height: 380px; display: flex; 
                        flex-direction: column; justify-content: center; align-items: center;
                        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15);'>
                <div style='font-size: 100px; margin-bottom: 15px;'>🚗</div>
                <h2 style='margin: 0; font-size: 26px; font-weight: 600;'>Car Price Prediction</h2>
                <p style='margin-top: 12px; font-size: 15px; opacity: 0.85;'>
                    Powered by Machine Learning
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='success-box'>
                <h3>Get started</h3>
                <p>Select <strong>“Make Prediction”</strong> from the sidebar.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def page_predict(model):
    """Render the prediction page; requires a loaded model."""
    st.markdown("<div class='page-card'>", unsafe_allow_html=True)

    if model is None:
        st.error("Model unavailable – check the sidebar.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("<h2>Predict Resale Price</h2>", unsafe_allow_html=True)

    # inputs
    year = st.slider("Year of manufacture", 2000, datetime.now().year, 2015, 1)
    present_price = st.number_input(
        "Present price (lakhs)", 0.1, 50.0, 7.0, 0.1
    )
    kms_driven = st.number_input(
        "Kilometres driven", 0, 500_000, 50_000, step=1_000
    )

    fuel_type = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.slider("Number of previous owners", 0, 3, 1, 1)

    mapping = {
        "Fuel_Type": {"Petrol": 0, "Diesel": 1, "CNG": 2},
        "Seller_Type": {"Dealer": 0, "Individual": 1},
        "Transmission": {"Manual": 0, "Automatic": 1},
    }

    if st.button("Predict price", use_container_width=True):
        df = pd.DataFrame(
            {
                "Year": [year],
                "Present_Price": [present_price],
                "Kms_Driven": [kms_driven],
                "Fuel_Type": [mapping["Fuel_Type"][fuel_type]],
                "Seller_Type": [mapping["Seller_Type"][seller_type]],
                "Transmission": [mapping["Transmission"][transmission]],
                "Owner": [owner],
            }
        )
        pred = model.predict(df)[0]
        st.markdown(
            f"""
            <div class='prediction-box'>
                Predicted selling price<br>₹{pred:.2f} lakhs
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='info-box'><h4>Input summary</h4>", unsafe_allow_html=True)
            st.write(f"• Year: {year}")
            st.write(f"• Present price: ₹{present_price:.2f} lakhs")
            st.write(f"• Kilometres driven: {kms_driven:,} km")
            st.write(f"• Fuel type: {fuel_type}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='info-box'><h4>Configuration</h4>", unsafe_allow_html=True)
            st.write(f"• Seller type: {seller_type}")
            st.write(f"• Transmission: {transmission}")
            st.write(f"• Owners: {owner}")
            st.write("• Model: Random Forest Regressor")
            st.markdown("</div>", unsafe_allow_html=True)

        change = ((pred - present_price) / present_price) * 100
        if change > 0:
            st.success(f"Price appreciation: {change:.2f}%")
        else:
            st.info(f"Price depreciation: {abs(change):.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)


def page_performance():
    """Show model performance metrics and plots."""
    st.markdown("<div class='page-card'>", unsafe_allow_html=True)
    st.markdown("<h2>Model performance</h2>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='info-box'>
        <h3>Model specifications</h3>
        <ul>
            <li>Algorithm: Random Forest Regressor</li>
            <li>Trees: 100</li>
            <li>Training samples: 240</li>
            <li>Test samples: 60</li>
            <li>Features: 7</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Train R²", "0.9841", delta="Excellent")
    col2.metric("Test R²", "0.9625", delta="Excellent")
    col3.metric("Test RMSE", "0.929", delta="Low error")

    st.markdown("---")
    st.subheader("Model comparison")
    comparison_df = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Lasso Regression", "Random Forest"],
            "Train R²": [0.8840, 0.8726, 0.9841],
            "Test R²": [0.8468, 0.8448, 0.9625],
            "MAE": [1.176, 1.158, 0.620],
            "RMSE": [1.751, 1.835, 0.929],
        }
    )
    st.dataframe(comparison_df, use_container_width=True)

    st.subheader("Feature importance")
    feat_df = pd.DataFrame(
        {
            "Feature": [
                "Present_Price",
                "Year",
                "Kms_Driven",
                "Transmission",
                "Fuel_Type",
                "Seller_Type",
                "Owner",
            ],
            "Importance": [0.879, 0.064, 0.034, 0.013, 0.006, 0.002, 0.001],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_df)))
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="black")
    ax.set_xlabel("Importance score", fontsize=12, fontweight="bold")
    ax.set_title("Feature importance", fontsize=14, fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", ha="left", va="center")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(
        """
        <div class='success-box'>
        <h4>Key insights</h4>
        <ul>
            <li>Present price dominates (87.9%).</li>
            <li>Year and kilometres also matter.</li>
            <li>Other factors have minor but useful contributions.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def page_about():
    """Render the about page with project metadata."""
    st.markdown("<div class='page-card'>", unsafe_allow_html=True)
    st.markdown("<h2>About this project</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class='info-box'>
            <h3>Project overview</h3>
            <p>
            Predict car resale values using historical data and a random forest model.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='info-box'>
            <h3>Technology stack</h3>
            <ul>
                <li>Python / scikit‑learn / pandas</li>
                <li>Streamlit for UI</li>
                <li>Matplotlib &amp; Seaborn for graphics</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class='info-box'>
            <h3>Dataset</h3>
            <ul>
                <li>300 records (240 train / 60 test)</li>
                <li>Seven input features</li>
                <li>Target: selling price (lakhs)</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='success-box'>
            <h3>Model results</h3>
            <p>
            96.25 % test accuracy, RMSE 0.929 lakhs – suitable for rough estimates.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        <div class='info-box'>
        <h3>How it works</h3>
        <p>
        The model learns from past transactions. A random forest ensemble 
        reduces overfitting and generalises well to new inputs.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='info-box'>
        <h3>Usage tips</h3>
        <ul>
            <li>Enter accurate current price and mileage.</li>
            <li>Choose correct fuel/transmission.</li>
            <li>Use prediction as an estimate; actual prices vary.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    """Render a simple footer at the bottom of every page."""
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #95a5a6; font-size:0.9em;'>"
        "AutoValue Pro – Car Price Prediction | Built with Streamlit & ML | 2026"
        "</p>",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="AutoValue Pro – Car Price Predictor",
        page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    # banner
    st.markdown(
        """
        <div class="header-banner">
            <h1>AutoValue Pro</h1>
            <p>Accurate car resale price estimates</p>
        </div>
        <style>
        .header-banner {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            border-radius: 0 0 20px 20px;
            text-align: center;
            color: white;
            margin-bottom: 20px;
        }
        .header-banner p {
            font-size: 1.2rem;
            margin: 0;
            opacity: 0.9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "🛠️ Make Prediction", "📊 Model Performance", "ℹ️ About"],
    )

    model = load_model(MODEL_PATH)

    if page.startswith("🏠"):
        page_home()
    elif page.startswith("🛠"):
        page_predict(model)
    elif page.startswith("📊"):
        page_performance()
    elif page.startswith("ℹ"):
        page_about()

    render_footer()


if __name__ == "__main__":
    main()