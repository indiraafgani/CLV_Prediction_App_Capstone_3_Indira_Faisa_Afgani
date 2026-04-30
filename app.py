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
            "color" : "#8B6914",
            "bg"    : "#FFF8E7",
            "border": "#D4A017",
            "action": "Educate & nurture. Focus on upselling to additional policies.",
            "cac"   : f"${clv * 0.20:,.0f}"
        }
    elif clv < 10000:
        return {
            "name"  : "🥈 Silver",
            "range" : "$5,000 – $10,000",
            "color" : "#4A4A5A",
            "bg"    : "#F2F2F5",
            "border": "#9E9EB8",
            "action": "Cross-sell higher coverage tiers. Moderate retention budget.",
            "cac"   : f"${clv * 0.20:,.0f}"
        }
    elif clv < 20000:
        return {
            "name"  : "🥇 Gold",
            "range" : "$10,000 – $20,000",
            "color" : "#7A5C00",
            "bg"    : "#FFFBEA",
            "border": "#F0C000",
            "action": "Priority retention. Offer loyalty rewards and dedicated support.",
            "cac"   : f"${clv * 0.20:,.0f}"
        }
    else:
        return {
            "name"  : "💎 Platinum",
            "range" : "> $20,000",
            "color" : "#0a66b7",
            "bg"    : "#EBF4FF",
            "border": "#0a66b7",
            "action": "White-glove service. Zero churn tolerance. VIP treatment.",
            "cac"   : f"${clv * 0.20:,.0f}"
        }

# =============================================
# CUSTOM CSS
# =============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background gradient */
    .stApp {
        background: linear-gradient(160deg, #8fa8c8 0%, #6b8db5 40%, #4a6f9e 100%);
        min-height: 100vh;
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #222338 0%, #0a66b7 65%, #1a85d4 100%);
        border-radius: 20px;
        padding: 2.8rem 2rem 2.2rem;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(10, 102, 183, 0.4);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 65%);
        pointer-events: none;
    }

    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -10%;
        width: 50%;
        height: 150%;
        background: radial-gradient(circle, rgba(26,133,212,0.15) 0%, transparent 65%);
        pointer-events: none;
    }

    .hero-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        display: block;
        filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3));
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 16px rgba(0,0,0,0.25);
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0.5rem 0 0 0;
        letter-spacing: 0.3px;
        text-shadow: 0 1px 8px rgba(0,0,0,0.3);
    }

    .hero-desc {
        font-size: 0.88rem;
        color: rgba(255,255,255,0.88);
        margin: 0.5rem 0 0 0;
        font-weight: 500;
        text-shadow: 0 1px 6px rgba(0,0,0,0.2);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 0.28rem 1rem;
        font-size: 0.78rem;
        color: rgba(255,255,255,0.88);
        margin-top: 1rem;
        backdrop-filter: blur(6px);
        letter-spacing: 0.2px;
    }

    /* Section title */
    .section-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #222338;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.6rem;
        padding-bottom: 0.35rem;
        border-bottom: 2.5px solid #0a66b7;
        display: inline-block;
    }

    /* Form wrapper */
    .stForm {
        background: rgba(255,255,255,0.72) !important;
        backdrop-filter: blur(14px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(10, 102, 183, 0.18) !important;
        box-shadow: 0 4px 24px rgba(34, 35, 56, 0.09) !important;
        padding: 1.5rem !important;
    }

    /* Result card */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid;
        box-shadow: 0 6px 28px rgba(10, 102, 183, 0.14);
        backdrop-filter: blur(8px);
    }

    .clv-value {
        font-size: 3.4rem;
        font-weight: 800;
        color: #222338;
        text-align: center;
        margin: 0.2rem 0;
        letter-spacing: -1.5px;
    }

    .segment-name {
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .segment-range {
        text-align: center;
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.4rem;
    }

    /* Metrics */
    .metric-container {
        display: flex;
        gap: 0.8rem;
        margin: 1rem 0;
    }

    .metric-box {
        flex: 1;
        background: rgba(255,255,255,0.82);
        border-radius: 12px;
        padding: 1rem 0.8rem;
        text-align: center;
        border: 1px solid rgba(10, 102, 183, 0.18);
        box-shadow: 0 2px 10px rgba(34, 35, 56, 0.06);
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0a66b7;
    }

    .metric-label {
        font-size: 0.72rem;
        color: #777;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }

    /* Action box */
    .action-box {
        background: linear-gradient(135deg, rgba(10,102,183,0.07), rgba(34,35,56,0.04));
        border-left: 4px solid #0a66b7;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.2rem;
        margin: 0.8rem 0;
    }

    .action-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #0a66b7;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .action-text {
        font-size: 0.92rem;
        color: #222338;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    /* Segment guide cards */
    .seg-card {
        background: rgba(255,255,255,0.72);
        border-radius: 14px;
        padding: 1.1rem 0.8rem;
        text-align: center;
        border: 1px solid rgba(10, 102, 183, 0.14);
        box-shadow: 0 2px 12px rgba(34,35,56,0.06);
        transition: transform 0.2s;
    }

    .seg-emoji  { font-size: 2rem; display: block; }
    .seg-title  { font-weight: 700; color: #222338; font-size: 0.95rem; margin: 0.35rem 0 0.2rem; }
    .seg-range  { font-size: 0.75rem; color: #0a66b7; font-weight: 600; }
    .seg-stat   { font-size: 0.72rem; color: #666; margin: 0.12rem 0; }
    .seg-strat  { font-size: 0.75rem; color: #444; margin-top: 0.4rem; font-style: italic; }

    /* Submit button */
    .stButton > button {
        background: linear-gradient(135deg, #0a66b7 0%, #222338 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        width: 100% !important;
        box-shadow: 0 4px 16px rgba(10, 102, 183, 0.4) !important;
    }

    .stButton > button:hover {
        box-shadow: 0 6px 22px rgba(10, 102, 183, 0.55) !important;
        transform: translateY(-1px) !important;
    }

    /* Divider */
    hr { border-color: rgba(10, 102, 183, 0.18) !important; }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.6) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #222338 !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #555;
        font-size: 0.78rem;
        padding: 1rem 0 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HERO BANNER
# =============================================
st.markdown("""
<div class='hero-banner'>
    <span class='hero-icon'>🛡️</span>
    <div class='hero-title'>AutoShield Insurance</div>
    <div class='hero-subtitle'>Customer Lifetime Value Predictor</div>
    <div class='hero-desc'>Enter customer information to predict CLV and assign the right segment instantly</div>
    <div class='hero-badge'>⚡ Gradient Boosting &nbsp;·&nbsp; R² = 0.906 &nbsp;·&nbsp; MAPE = 9.89%</div>
</div>
""", unsafe_allow_html=True)

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
with st.form("clv_form"):

    # Vehicle & Coverage
    st.markdown("<div class='section-title'>🚗 Vehicle & Coverage</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        vehicle_class = st.selectbox("Vehicle Class",
            ["Two-Door Car", "Four-Door Car", "SUV", "Sports Car", "Luxury Car", "Luxury SUV"])
    with c2:
        coverage = st.selectbox("Coverage Type", ["Basic", "Extended", "Premium"])
    with c3:
        renew_offer = st.selectbox("Renew Offer Type", ["Offer1", "Offer2", "Offer3", "Offer4"])

    st.write("")

    # Customer Profile
    st.markdown("<div class='section-title'>👤 Customer Profile</div>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        employment = st.selectbox("Employment Status",
            ["Employed", "Unemployed", "Medical Leave", "Disabled", "Retired"])
    with c5:
        marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    with c6:
        education = st.selectbox("Education Level",
            ["High School or Below", "College", "Bachelor", "Master", "Doctor"])

    st.write("")

    # Financial Info
    st.markdown("<div class='section-title'>💰 Financial Information</div>", unsafe_allow_html=True)
    c7, c8, c9, c10 = st.columns(4)
    with c7:
        num_policies = st.slider("No. of Policies", min_value=1, max_value=9, value=2)
    with c8:
        monthly_premium = st.number_input("Monthly Premium ($)", min_value=61, max_value=300, value=90)
    with c9:
        total_claim = st.number_input("Total Claim ($)", min_value=0.0, max_value=3000.0, value=400.0, step=10.0)
    with c10:
        income = st.number_input("Annual Income ($)", min_value=0, max_value=100000, value=35000, step=1000)

    st.write("")
    submitted = st.form_submit_button("🔍 Predict Customer Lifetime Value", use_container_width=True)

# =============================================
# PREDICTION RESULTS
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
        clv     = predict_clv(input_data)
        segment = get_segment(clv)

    st.markdown("---")

    # Result card
    st.markdown(f"""
    <div class='result-card' style='border-color:{segment["border"]}; background:{segment["bg"]};'>
        <div class='segment-name' style='color:{segment["color"]};'>{segment["name"]}</div>
        <div class='segment-range'>Segment Range: {segment["range"]}</div>
        <div class='clv-value'>${clv:,.2f}</div>
        <p style='text-align:center; color:#888; font-size:0.82rem; margin:0.2rem 0 0;'>
            Predicted Customer Lifetime Value
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    st.markdown(f"""
    <div class='metric-container'>
        <div class='metric-box'>
            <div class='metric-value'>{segment["name"].split()[0]}</div>
            <div class='metric-label'>Segment</div>
        </div>
        <div class='metric-box'>
            <div class='metric-value'>${clv:,.0f}</div>
            <div class='metric-label'>Predicted CLV</div>
        </div>
        <div class='metric-box'>
            <div class='metric-value'>{segment["cac"]}</div>
            <div class='metric-label'>Max Recommended CAC</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action
    st.markdown(f"""
    <div class='action-box'>
        <div class='action-label'>📌 Recommended Action</div>
        <div class='action-text'>{segment["action"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # Input summary
    with st.expander("📋 View Input Summary"):
        summary_df = pd.DataFrame({
            "Feature": list(input_data.keys()),
            "Value"  : [str(v) for v in input_data.values()]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# =============================================
# SEGMENT GUIDE
# =============================================
st.markdown("---")
st.markdown("<div class='section-title'>📖 Segment Reference Guide</div>", unsafe_allow_html=True)
st.write("")

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("""<div class='seg-card'>
        <span class='seg-emoji'>🥉</span>
        <div class='seg-title'>Bronze</div>
        <div class='seg-range'>CLV &lt; $5,000</div>
        <div class='seg-stat'>37.3% of customers</div>
        <div class='seg-stat'>16.7% of total CLV</div>
        <div class='seg-strat'>Nurture & Educate</div>
    </div>""", unsafe_allow_html=True)

with s2:
    st.markdown("""<div class='seg-card'>
        <span class='seg-emoji'>🥈</span>
        <div class='seg-title'>Silver</div>
        <div class='seg-range'>$5K – $10K</div>
        <div class='seg-stat'>38.1% of customers</div>
        <div class='seg-stat'>35.9% of total CLV</div>
        <div class='seg-strat'>Cross-sell</div>
    </div>""", unsafe_allow_html=True)

with s3:
    st.markdown("""<div class='seg-card'>
        <span class='seg-emoji'>🥇</span>
        <div class='seg-title'>Gold</div>
        <div class='seg-range'>$10K – $20K</div>
        <div class='seg-stat'>22.9% of customers</div>
        <div class='seg-stat'>42.1% of total CLV</div>
        <div class='seg-strat'>Retain & Reward</div>
    </div>""", unsafe_allow_html=True)

with s4:
    st.markdown("""<div class='seg-card'>
        <span class='seg-emoji'>💎</span>
        <div class='seg-title'>Platinum</div>
        <div class='seg-range'>CLV &gt; $20K</div>
        <div class='seg-stat'>1.7% of customers</div>
        <div class='seg-stat'>5.4% of total CLV</div>
        <div class='seg-strat'>VIP Service</div>
    </div>""", unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.write("")
st.markdown("""
<div class='footer-text'>
    AutoShield Insurance Co. © 2024 &nbsp;·&nbsp;
    Gradient Boosting Regressor &nbsp;·&nbsp;
    R² = 0.9060 &nbsp;·&nbsp; MAPE = 9.89%
</div>
""", unsafe_allow_html=True)
