import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="CLV Predictor — AutoShield Insurance",
    page_icon="🛡️",
    layout="centered"
)

# =============================================
# LOAD MODEL
# =============================================
@st.cache_resource
def load_model():
    model_path = "clv_gradient_boosting_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# =============================================
# HELPER FUNCTIONS
# =============================================
def predict_clv(input_data: dict) -> float:
    df_input = pd.DataFrame([input_data])
    log_clv  = model.predict(df_input)[0]
    return np.exp(log_clv)

def get_segment(clv: float) -> dict:
    if clv < 5000:
        return {
            "name"  : "🥉 Bronze",
            "range" : "< $5,000",
            "color" : "#CD7F32",
            "bg"    : "#FFF5E6",
            "action": "Educate & nurture. Focus on upselling to additional policies.",
            "cac"   : f"${clv * 0.20:,.0f}"
        }
    elif clv < 10000:
        return {
            "name"  : "🥈 Silver",
            "range" : "$5,000 – $10,000",
            "color" : "#808080",
            "bg"    : "#F5F5F5",
            "action": "Cross-sell higher coverage tiers. Moderate retention budget.",
            "cac"   : f"${clv * 0.20:,.0f}"
        }
    elif clv < 20000:
        return {
            "name"  : "🥇 Gold",
            "range" : "$10,000 – $20,000",
            "color" : "#B8860B",
            "bg"    : "#FFFDE7",
            "action": "Priority retention. Offer loyalty rewards and dedicated support.",
            "cac"   : f"${clv * 0.20:,.0f}"
        }
    else:
        return {
            "name"  : "💎 Platinum",
            "range" : "> $20,000",
            "color" : "#4A4A8A",
            "bg"    : "#EDE7F6",
            "action": "White-glove service. Zero churn tolerance. VIP treatment.",
            "cac"   : f"${clv * 0.20:,.0f}"
        }

# =============================================
# CUSTOM CSS
# =============================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid;
    }
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        flex: 1;
        margin: 0 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f3a5f;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
    }
    .info-box {
        background: #EBF5FB;
        border-left: 4px solid #2E86C1;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER
# =============================================
st.markdown("""
<div class='main-header'>
    <h1>🛡️ AutoShield Insurance</h1>
    <h3>Customer Lifetime Value Predictor</h3>
    <p style='color: #666;'>Predict CLV and assign the right customer segment instantly</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =============================================
# MODEL CHECK
# =============================================
if model is None:
    st.error("⚠️ Model file not found. Please ensure `clv_gradient_boosting_model.joblib` is in the same directory.")
    st.info("💡 Run the notebook to generate the model file first.")
    st.stop()

# =============================================
# INPUT FORM
# =============================================
st.subheader("📋 Customer Information")
st.caption("Fill in the customer details below to predict their CLV.")

with st.form("clv_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🚗 Vehicle & Coverage**")
        vehicle_class = st.selectbox(
            "Vehicle Class",
            ["Two-Door Car", "Four-Door Car", "SUV",
             "Sports Car", "Luxury Car", "Luxury SUV"],
            help="Type of vehicle owned by the customer"
        )
        coverage = st.selectbox(
            "Coverage Type",
            ["Basic", "Extended", "Premium"],
            help="Insurance coverage level selected by the customer"
        )
        renew_offer = st.selectbox(
            "Renew Offer Type",
            ["Offer1", "Offer2", "Offer3", "Offer4"],
            help="Type of renewal offer given to the customer"
        )

    with col2:
        st.markdown("**👤 Customer Profile**")
        employment = st.selectbox(
            "Employment Status",
            ["Employed", "Unemployed", "Medical Leave",
             "Disabled", "Retired"],
            help="Current employment status"
        )
        marital = st.selectbox(
            "Marital Status",
            ["Married", "Single", "Divorced"],
            help="Customer's marital status"
        )
        education = st.selectbox(
            "Education Level",
            ["High School or Below", "College",
             "Bachelor", "Master", "Doctor"],
            help="Highest education level attained"
        )

    st.markdown("**💰 Financial Information**")
    col3, col4 = st.columns(2)

    with col3:
        num_policies = st.slider(
            "Number of Policies",
            min_value=1, max_value=9, value=2,
            help="Total number of insurance policies held"
        )
        monthly_premium = st.number_input(
            "Monthly Premium Auto ($)",
            min_value=61, max_value=300, value=90,
            help="Monthly premium amount paid (between $61–$300)"
        )

    with col4:
        total_claim = st.number_input(
            "Total Claim Amount ($)",
            min_value=0.0, max_value=3000.0, value=400.0,
            step=10.0,
            help="Total claim amount submitted so far"
        )
        income = st.number_input(
            "Annual Income ($)",
            min_value=0, max_value=100000, value=35000,
            step=1000,
            help="Annual income (0 for unemployed customers)"
        )

    submitted = st.form_submit_button(
        "🔍 Predict CLV",
        use_container_width=True
    )

# =============================================
# PREDICTION
# =============================================
if submitted:
    input_data = {
        "Vehicle Class"       : vehicle_class,
        "Coverage"            : coverage,
        "Renew Offer Type"    : renew_offer,
        "EmploymentStatus"    : employment,
        "Marital Status"      : marital,
        "Education"           : education,
        "Number of Policies"  : num_policies,
        "Monthly Premium Auto": monthly_premium,
        "Total Claim Amount"  : total_claim,
        "Income"              : income
    }

    with st.spinner("Calculating CLV..."):
        clv       = predict_clv(input_data)
        segment   = get_segment(clv)

    st.divider()
    st.subheader("📊 Prediction Results")

    # --- CLV Result ---
    st.markdown(f"""
    <div class='result-card' style='border-color:{segment["color"]}; background:{segment["bg"]};'>
        <h2 style='text-align:center; color:{segment["color"]}; margin-bottom:0.5rem;'>
            {segment["name"]} Customer
        </h2>
        <h1 style='text-align:center; color:#1f3a5f; font-size:3rem; margin:0;'>
            ${clv:,.2f}
        </h1>
        <p style='text-align:center; color:#666; margin-top:0.25rem;'>
            Predicted Customer Lifetime Value
        </p>
        <p style='text-align:center; color:{segment["color"]}; font-weight:bold;'>
            Segment Range: {segment["range"]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Key Metrics ---
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("📦 Segment", segment["name"].split(" ", 1)[1])
    with col_b:
        st.metric("💵 Predicted CLV", f"${clv:,.0f}")
    with col_c:
        st.metric("🎯 Max Recommended CAC", segment["cac"])

    # --- Action Recommendation ---
    st.markdown("**📌 Recommended Action**")
    st.info(f"**{segment['name']}** → {segment['action']}")

    # --- Input Summary ---
    with st.expander("📋 View Input Summary"):
        summary_df = pd.DataFrame({
            "Feature": list(input_data.keys()),
            "Value"  : [str(v) for v in input_data.values()]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # --- Segment Guide ---
    st.divider()
    st.subheader("📖 Segment Reference Guide")

    seg_col1, seg_col2, seg_col3, seg_col4 = st.columns(4)

    with seg_col1:
        st.markdown("""
        **🥉 Bronze**
        - CLV < $5,000
        - 37.3% of customers
        - 16.7% of total CLV
        - Strategy: Nurture & Educate
        """)
    with seg_col2:
        st.markdown("""
        **🥈 Silver**
        - CLV $5K – $10K
        - 38.1% of customers
        - 35.9% of total CLV
        - Strategy: Cross-sell
        """)
    with seg_col3:
        st.markdown("""
        **🥇 Gold**
        - CLV $10K – $20K
        - 22.9% of customers
        - 42.1% of total CLV
        - Strategy: Retain & Reward
        """)
    with seg_col4:
        st.markdown("""
        **💎 Platinum**
        - CLV > $20K
        - 1.7% of customers
        - 5.4% of total CLV
        - Strategy: VIP Service
        """)

# =============================================
# FOOTER
# =============================================
st.divider()
st.markdown("""
<p style='text-align:center; color:#999; font-size:0.8rem;'>
    Powered by Gradient Boosting Regressor · R² = 0.9060 · MAPE = 9.89% ·
    AutoShield Insurance Co. © 2024
</p>
""", unsafe_allow_html=True)
