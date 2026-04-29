"""
app.py – Customer Lifetime Value Prediction App
Run: streamlit run app.py
"""

import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from preprocessing import preprocess_input, predict_clv


# ─── Auto-train if model.pkl not found ───────────────────────────────────────
def train_and_save_model(data_path: str = "data_customer_lifetime_value.csv",
                          model_path: str = "model.pkl"):
    from sklearn.preprocessing import OrdinalEncoder, RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor

    df = pd.read_csv(data_path)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    df_clean["Number of Policies"] = df_clean["Number of Policies"].astype(int)
    df_clean["CLV_log"] = np.log(df_clean["Customer Lifetime Value"])

    coverage_order  = ["Basic", "Extended", "Premium"]
    vehicle_order   = ["Two-Door Car", "Four-Door Car", "SUV",
                       "Sports Car", "Luxury Car", "Luxury SUV"]
    education_order = ["High School or Below", "College",
                       "Bachelor", "Master", "Doctor"]

    ordinal_enc = OrdinalEncoder(
        categories=[coverage_order, vehicle_order, education_order]
    )
    df_clean[["Coverage_enc", "VehicleClass_enc", "Education_enc"]] = \
        ordinal_enc.fit_transform(
            df_clean[["Coverage", "Vehicle Class", "Education"]]
        )

    df_clean = pd.get_dummies(
        df_clean,
        columns=["Renew Offer Type", "EmploymentStatus", "Marital Status"],
        drop_first=True,
        dtype=int,
    )

    cols_to_drop = ["Customer Lifetime Value", "Coverage", "Vehicle Class",
                    "Education", "CLV_log"]
    X = df_clean.drop(cols_to_drop, axis=1)
    y = df_clean["CLV_log"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_cols = ["Monthly Premium Auto", "Total Claim Amount",
                      "Number of Policies", "Income"]
    bounds = {}
    for col in numerical_cols:
        Q1  = X_train[col].quantile(0.25)
        Q3  = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    X_train_capped = X_train.copy()
    for col in numerical_cols:
        lo, hi = bounds[col]
        X_train_capped[col] = X_train[col].clip(lower=lo, upper=hi)

    scaler = RobustScaler()
    X_train_capped[numerical_cols] = scaler.fit_transform(
        X_train_capped[numerical_cols]
    )

    selected_features = [
        "Number of Policies",
        "Monthly Premium Auto",
        "VehicleClass_enc",
        "Coverage_enc",
        "Total Claim Amount",
    ]
    X_train_selected = X_train_capped[selected_features]

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_selected, y_train)

    artifact = {
        "model":             model,
        "ordinal_enc":       ordinal_enc,
        "scaler":            scaler,
        "bounds":            bounds,
        "selected_features": selected_features,
        "numerical_cols":    numerical_cols,
        "coverage_order":    coverage_order,
        "vehicle_order":     vehicle_order,
    }
    joblib.dump(artifact, model_path)
    return artifact


# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLV Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── FEVER Color Palette CSS ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #FCF5AF;
    }
    .stApp {
        background: #060608;
    }
    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        background: linear-gradient(135deg, #0a0a0f 0%, #0B1A2E 60%, #1a0a00 100%);
        border: 1px solid rgba(228,79,10,0.3);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -80px; right: -80px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(228,79,10,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -60px; left: -60px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(11,75,139,0.2) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        letter-spacing: 0.08em;
        background: linear-gradient(90deg, #F0A533, #E44F0A, #BA011A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
        line-height: 1;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #a89060;
        margin: 0;
        font-weight: 400;
    }

    .section-card {
        background: #0D0D12;
        border: 1px solid rgba(228,79,10,0.15);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1rem;
        letter-spacing: 0.2em;
        color: #F0A533;
        margin-bottom: 1.2rem;
    }

    label, .stSelectbox label, .stNumberInput label,
    .stTextInput label, [data-testid="stWidgetLabel"] {
        color: #c8a870 !important;
        font-size: 0.87rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: #13100a !important;
        border: 1px solid rgba(240,165,51,0.25) !important;
        border-radius: 10px !important;
        color: #FCF5AF !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #E44F0A !important;
        box-shadow: 0 0 0 2px rgba(228,79,10,0.2) !important;
    }
    [data-baseweb="select"] ul li {
        background: #13100a !important;
        color: #FCF5AF !important;
    }
    [data-baseweb="popover"] { background: #13100a !important; }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #BA011A, #E44F0A);
        color: #FCF5AF;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        font-family: 'Bebas Neue', sans-serif;
        letter-spacing: 0.12em;
        cursor: pointer;
        transition: all 0.25s ease;
        margin-top: 0.5rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #E44F0A, #F0A533);
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(228,79,10,0.35);
        color: #000000;
    }
    .stButton > button:active { transform: translateY(0); }

    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .clv-card {
        background: linear-gradient(135deg, #0B1A2E, #0e2040);
        border: 1px solid rgba(11,75,139,0.5);
    }
    .clv-label {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 0.9rem;
        letter-spacing: 0.2em;
        color: #5B9BD5;
        margin-bottom: 0.4rem;
    }
    .clv-customer {
        font-size: 0.88rem;
        color: #F0A533;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    .clv-value {
        font-size: 3rem;
        font-weight: 700;
        font-family: 'DM Mono', monospace;
        color: #FCF5AF;
        line-height: 1;
    }
    .clv-note {
        font-size: 0.76rem;
        color: #3a4a5a;
        margin-top: 0.5rem;
    }

    .segment-card { border-radius: 16px; padding: 1.8rem; text-align: center; }
    .seg-label {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 0.9rem;
        letter-spacing: 0.2em;
        margin-bottom: 0.5rem;
    }
    .seg-name {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.5rem;
        letter-spacing: 0.1em;
        line-height: 1;
    }
    .seg-desc { font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.75; }

    .seg-platinum {
        background: linear-gradient(135deg, #0B1A2E, #122040);
        border: 1px solid rgba(91,155,213,0.4);
    }
    .seg-platinum .seg-label { color: #5B9BD5; }
    .seg-platinum .seg-name  { color: #FCF5AF; }

    .seg-gold {
        background: linear-gradient(135deg, #1a1000, #271800);
        border: 1px solid rgba(240,165,51,0.4);
    }
    .seg-gold .seg-label { color: #F0A533; }
    .seg-gold .seg-name  { color: #FCF5AF; }

    .seg-silver {
        background: linear-gradient(135deg, #080e18, #0e1a28);
        border: 1px solid rgba(91,155,213,0.25);
    }
    .seg-silver .seg-label { color: #7aaac8; }
    .seg-silver .seg-name  { color: #c8dce8; }

    .seg-bronze {
        background: linear-gradient(135deg, #1a0500, #220800);
        border: 1px solid rgba(228,79,10,0.35);
    }
    .seg-bronze .seg-label { color: #E44F0A; }
    .seg-bronze .seg-name  { color: #F0A533; }

    .info-pill {
        display: inline-block;
        background: rgba(228,79,10,0.08);
        border: 1px solid rgba(228,79,10,0.2);
        border-radius: 999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.78rem;
        color: #E44F0A;
        margin: 0.25rem 0.2rem;
    }
    hr { border-color: rgba(228,79,10,0.1); }
    .stAlert { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifact():
    model_path = "model.pkl"
    data_path  = "data_customer_lifetime_value.csv"
    if os.path.exists(model_path):
        return joblib.load(model_path), None
    if not os.path.exists(data_path):
        return None, f"Dataset `{data_path}` tidak ditemukan di server."
    return train_and_save_model(data_path, model_path), None


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <p class="hero-title">🔥 Customer Lifetime Value Predictor</p>
        <p class="hero-sub">
            Enter customer details below to predict their estimated lifetime value
            and automatically classify them into a revenue segment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("⚙️ Setting up model for the first time, please wait..."):
    artifact, error_msg = load_artifact()

if artifact is None:
    st.error(f"❌ {error_msg}", icon="🚨")
    st.stop()

VEHICLE_OPTIONS  = artifact["vehicle_order"]
COVERAGE_OPTIONS = artifact["coverage_order"]

# ─── Layout ──────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📋 Customer Information</p>', unsafe_allow_html=True)

    customer_name = st.text_input(
        "Customer Name",
        placeholder="e.g. John Doe",
        help="Name of the customer being evaluated.",
    )

    r1a, r1b = st.columns(2)
    with r1a:
        number_of_policies = st.number_input(
            "Number of Policies",
            min_value=1, max_value=9, value=2, step=1,
            help="Total insurance policies held (1–9).",
        )
    with r1b:
        monthly_premium_auto = st.number_input(
            "Monthly Premium Auto ($)",
            min_value=61.0, max_value=298.0, value=100.0,
            step=1.0, format="%.2f",
            help="Monthly auto insurance premium in USD.",
        )

    r2a, r2b = st.columns(2)
    with r2a:
        vehicle_class = st.selectbox(
            "Vehicle Class", options=VEHICLE_OPTIONS, index=1,
            help="Type of vehicle insured.",
        )
    with r2b:
        coverage = st.selectbox(
            "Coverage Level", options=COVERAGE_OPTIONS, index=0,
            help="Insurance coverage tier.",
        )

    total_claim_amount = st.number_input(
        "Total Claim Amount ($)",
        min_value=0.0, max_value=2894.0, value=400.0,
        step=1.0, format="%.2f",
        help="Total dollar amount of claims filed.",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin-top:-0.5rem; margin-bottom:1rem;">
            <span class="info-pill">🧠 Gradient Boosting</span>
            <span class="info-pill">📐 RobustScaler</span>
            <span class="info-pill">🔢 Log-Target</span>
            <span class="info-pill">✂️ IQR Capping</span>
            <span class="info-pill">R² ≈ 0.90</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predict_clicked = st.button("🔥 Predict CLV & Segment", use_container_width=True)


# ─── Results ─────────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-card" style="height:100%">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Prediction Result</p>', unsafe_allow_html=True)

    SEGMENT_META = {
        "Platinum": {"css": "seg-platinum", "icon": "🏆", "range": "> $20,000",       "desc": "Top-tier, highest-value customer",        "color": "#5B9BD5"},
        "Gold":     {"css": "seg-gold",     "icon": "🥇", "range": "$10,000–$20,000", "desc": "High-value, strong retention priority",   "color": "#F0A533"},
        "Silver":   {"css": "seg-silver",   "icon": "🥈", "range": "$5,000–$10,000",  "desc": "Mid-tier, growth opportunity",            "color": "#7aaac8"},
        "Bronze":   {"css": "seg-bronze",   "icon": "🥉", "range": "< $5,000",        "desc": "Entry-level, cost-conscious segment",     "color": "#E44F0A"},
    }

    if predict_clicked:
        errors = []
        if not customer_name.strip():
            errors.append("Customer Name tidak boleh kosong.")
        if number_of_policies < 1:
            errors.append("Number of Policies must be ≥ 1.")
        if monthly_premium_auto <= 0:
            errors.append("Monthly Premium Auto must be > 0.")
        if total_claim_amount < 0:
            errors.append("Total Claim Amount cannot be negative.")

        if errors:
            for e in errors:
                st.error(f"❌ {e}")
        else:
            try:
                processed = preprocess_input(
                    number_of_policies=int(number_of_policies),
                    monthly_premium_auto=float(monthly_premium_auto),
                    total_claim_amount=float(total_claim_amount),
                    vehicle_class=vehicle_class,
                    coverage=coverage,
                    artifact=artifact,
                )
                clv, segment = predict_clv(processed, artifact)
                meta = SEGMENT_META[segment]
                name_display = customer_name.strip().title()

                st.markdown(
                    f"""
                    <div class="result-card clv-card">
                        <p class="clv-label">Predicted Customer Lifetime Value</p>
                        <p class="clv-customer">👤 {name_display}</p>
                        <p class="clv-value">${clv:,.0f}</p>
                        <p class="clv-note">Estimated total revenue from this customer</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="result-card segment-card {meta['css']}">
                        <p class="seg-label">Customer Segment</p>
                        <p class="seg-name">{meta['icon']} {segment}</p>
                        <p class="seg-desc">{meta['range']} &nbsp;|&nbsp; {meta['desc']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                st.markdown(
                    """<p style="font-family:'Bebas Neue',sans-serif;font-size:0.85rem;
                       color:#3a2a10;letter-spacing:0.15em;margin-bottom:0.6rem;">
                       Segment Thresholds</p>""",
                    unsafe_allow_html=True,
                )
                for icon, name, rng, color in [
                    ("🥉", "Bronze",   "< $5K",      "#E44F0A"),
                    ("🥈", "Silver",   "$5K–$10K",   "#7aaac8"),
                    ("🥇", "Gold",     "$10K–$20K",  "#F0A533"),
                    ("🏆", "Platinum", "> $20K",      "#5B9BD5"),
                ]:
                    active = "font-weight:700;" if name == segment else "opacity:0.35;"
                    st.markdown(
                        f"""<div style="display:flex;justify-content:space-between;
                            align-items:center;padding:0.3rem 0;{active}">
                            <span style="color:{color};font-size:0.88rem;">{icon} {name}</span>
                            <span style="color:#6a5030;font-size:0.82rem;
                                font-family:'DM Mono',monospace;">{rng}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            except Exception as exc:
                st.error(f"❌ Prediction failed: {exc}")
    else:
        st.markdown(
            """
            <div style="text-align:center; padding: 3rem 1rem;">
                <div style="font-size:3.5rem; margin-bottom:1rem;">🔥</div>
                <p style="font-size:1.1rem; font-weight:600; color:#3a2510;
                   font-family:'Bebas Neue',sans-serif; letter-spacing:0.12em;">
                    Awaiting Prediction
                </p>
                <p style="font-size:0.82rem; color:#2a1a08; margin-top:0.4rem;">
                    Fill in the customer details on the left<br>
                    and click <strong style="color:#3a2510;">🔥 Predict CLV &amp; Segment</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr>
    <p style="text-align:center;font-size:0.75rem;color:#2a1a08;margin-top:0.5rem;">
        Gradient Boosting · 5 features · R² ≈ 0.90 · Log-scale target
        &nbsp;|&nbsp; Built with Streamlit
    </p>
    """,
    unsafe_allow_html=True,
)
