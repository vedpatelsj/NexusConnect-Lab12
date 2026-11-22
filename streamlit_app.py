
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
import os

# ==========================================
# NEXUSCONNECT - CUSTOMER INTELLIGENCE
# ==========================================

st.set_page_config(
    page_title="NexusConnect AI",
    page_icon="📡",
    layout="wide"
)

# Styling
st.markdown("""
<style>
.main {
    background-color: #F5F8FC;
}
h1, h2, h3 {
    color: #12304A;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E2EAF2;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("📡 NexusConnect")
st.subheader("AI-Powered Customer Intelligence Platform")
st.caption(
    "Predict customer churn and support data-driven retention decisions."
)

st.divider()

MODEL_PATH = "nexusconnect_churn_model.pkl.gz"

# ==========================================
# LOAD COMPRESSED MODEL
# ==========================================

@st.cache_resource
def load_model():
    with gzip.open(MODEL_PATH, "rb") as f:
        return joblib.load(f)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Upload "
        "nexusconnect_churn_model.pkl.gz "
        "to the root of your GitHub repository."
    )
    st.stop()

try:
    model = load_model()
    st.success("AI model loaded successfully.")
except Exception as e:
    st.error("Could not load the model. Check that the compressed file is valid.")
    st.exception(e)
    st.stop()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("NexusConnect")
st.sidebar.caption("Customer Intelligence")

page = st.sidebar.radio(
    "Navigation",
    ["Churn Prediction", "Product Whitespace", "About"]
)

# ==========================================
# CHURN PREDICTION
# ==========================================

if page == "Churn Prediction":

    st.header("Customer Churn Prediction")

    st.write(
        "Upload a customer CSV containing the same feature columns "
        "used to train the Random Forest model."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer Dataset",
        type=["csv"]
    )

    threshold = st.slider(
        "High-risk prediction threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.60,
        step=0.05
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded dataset is empty.")
            st.stop()

        # Preserve customer IDs for display
        if "customerID" in df.columns:
            customer_ids = df["customerID"].astype(str)
        else:
            customer_ids = pd.Series(
                [f"Customer {i+1}" for i in range(len(df))]
            )

        # Remove ID and actual outcome from model input
        X = df.drop(
            columns=["customerID", "Churn"],
            errors="ignore"
        )

        try:
            probabilities = model.predict_proba(X)[:, 1]

        except Exception as e:
            st.error(
                "The uploaded CSV does not match the model's "
                "training features. Please use the original "
                "IBM Telco Customer Churn dataset."
            )
            st.exception(e)
            st.stop()

        # Create results
        results = pd.DataFrame({
            "Customer ID": customer_ids,
            "Churn Probability": probabilities
        })

        results["Churn Risk"] = (
            results["Churn Probability"] >= threshold
        ).map({
            True: "High Risk",
            False: "Lower Risk"
        })

        results["Churn Probability"] = (
            results["Churn Probability"] * 100
        ).round(2)

        # Metrics
        high = int(
            (results["Churn Risk"] == "High Risk").sum()
        )
        lower = len(results) - high
        average = results["Churn Probability"].mean()

        c1, c2, c3 = st.columns(3)

        c1.metric("Customers Analyzed", f"{len(results):,}")
        c2.metric("High-Risk Customers", f"{high:,}")
        c3.metric("Average Churn Probability", f"{average:.1f}%")

        st.divider()

        # Results
        st.subheader("Customer Retention Worklist")

        sorted_results = results.sort_values(
            "Churn Probability",
            ascending=False
        )

        st.dataframe(
            sorted_results,
            use_container_width=True,
            hide_index=True
        )

        # Chart
        st.subheader("Churn Risk Distribution")

        chart_data = pd.DataFrame({
            "Risk Category": ["High Risk", "Lower Risk"],
            "Customers": [high, lower]
        })

        st.bar_chart(
            chart_data.set_index("Risk Category")
        )

        # Download
        csv = sorted_results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction Results",
            data=csv,
            file_name="nexusconnect_churn_predictions.csv",
            mime="text/csv"
        )

        st.info(
            "Predictions are decision-support estimates, not "
            "certain outcomes. Validate the model before using "
            "it with real customers."
        )

    else:
        st.info(
            "Upload a customer CSV to begin analyzing churn risk."
        )

# ==========================================
# PRODUCT WHITESPACE
# ==========================================

elif page == "Product Whitespace":

    st.header("Product Whitespace Identification")

    st.write(
        "Identify products that customers may be eligible to "
        "purchase but do not currently own."
    )

    st.warning(
        "This module is currently a demonstration interface. "
        "The UCI Online Retail association-rule analysis is "
        "separate from the telecom churn model."
    )

    st.subheader("How the system works")

    st.markdown("""
    1. Analyze historical customer-product purchases.
    2. Identify frequently co-purchased products.
    3. Compare recommendations against existing ownership.
    4. Apply eligibility and customer-consent rules.
    5. Evaluate recommendations using conversion rates and lift.
    """)

    st.info(
        "A production version requires a product catalog and "
        "customer ownership data."
    )

# ==========================================
# ABOUT
# ==========================================

elif page == "About":

    st.header("About NexusConnect")

    st.write("""
    NexusConnect is an AI-powered customer intelligence
    prototype designed to help businesses understand customer
    behavior and make better retention decisions.
    """)

    st.subheader("Machine Learning Architecture")

    st.markdown("""
    - **Algorithm:** Random Forest Classifier
    - **Task:** Binary customer churn classification
    - **Training data:** IBM Telco Customer Churn benchmark
    - **Preprocessing:** Median/mode imputation and one-hot encoding
    - **Output:** Estimated customer churn probability
    """)

    st.subheader("Important Limitations")

    st.write("""
    This educational prototype uses public benchmark data.
    It has not been validated for operational business decisions.
    Do not upload confidential customer information.
    """)
