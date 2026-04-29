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

# ─── Color Palette CSS ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #222338;
    }

    .stApp {
        background: #e1e1e1;
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* HERO */
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #f3f3f3 100%);
        border: 1px solid rgba(34,35,56,0.1);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.8rem;
        letter-spacing: 0.05em;
        color: #0a66b7;
        margin-bottom: 0.3rem;
    }

    .hero-sub {
        font-size: 0.95rem;
        color: #555;
    }

    /* CARD */
    .section-card {
        background: #ffffff;
        border: 1px solid rgba(34,35,56,0.1);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1rem;
        letter-spacing: 0.15em;
        color: #0a66b7;
        margin-bottom: 1.2rem;
    }

    /* INPUT */
    label {
        color: #222338 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #ccc !important;
        border-radius: 10px !important;
        color: #222338 !important;
    }

    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #0a66b7 !important;
        box-shadow: 0 0 0 2px rgba(10,102,183,0.15) !important;
    }

    /* BUTTON */
    .stButton > button {
        width: 100%;
        background: #0a66b7;
        color: white;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #084c87;
        transform: translateY(-1px);
    }

    /* RESULT */
    .result-card {
        background: #ffffff;
        border: 1px solid rgba(34,35,56,0.1);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .clv-label {
        font-size: 0.8rem;
        color: #666;
    }

    .clv-customer {
        font-size: 0.9rem;
        color: #0a66b7;
        font-weight: 600;
    }

    .clv-value {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'DM Mono', monospace;
        color: #222338;
    }

    /* SEGMENTS */
    .segment-card {
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }

    .seg-name {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.2rem;
    }

    .seg-label {
        font-size: 0.8rem;
        color: #666;
    }

    .seg-desc {
        font-size: 0.8rem;
        color: #777;
    }

    .seg-platinum { background:#eef5fb; border:1px solid #0a66b7; }
    .seg-gold     { background:#f7f3e8; border:1px solid #c9a227; }
    .seg-silver   { background:#f0f2f5; border:1px solid #999; }
    .seg-bronze   { background:#f9eee8; border:1px solid #cd7f32; }

    /* PILLS */
    .info-pill {
        background: #f0f0f0;
        border: 1px solid #ccc;
        color: #333;
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        font-size: 0.75rem;
    }

    hr {
        border-color: rgba(0,0,0,0.1);
    }

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
                       color:#F0A533;letter-spacing:0.15em;margin-bottom:0.6rem;">
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
                            <span style="color:#c8a870;font-size:0.82rem;
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
                <p style="font-size:1.15rem; font-weight:700; color:#F0A533;
                   font-family:'Bebas Neue',sans-serif; letter-spacing:0.15em;">
                    AWAITING PREDICTION
                </p>
                <p style="font-size:0.88rem; color:#c8a870; margin-top:0.5rem; line-height:1.6;">
                    Fill in the customer details on the left<br>
                    and click <strong style="color:#E44F0A;">🔥 Predict CLV &amp; Segment</strong>
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
    <p style="text-align:center;font-size:0.75rem;color:#8a7050;margin-top:0.5rem;">
        Gradient Boosting · 5 features · R² ≈ 0.90 · Log-scale target
        &nbsp;|&nbsp; Built with Streamlit
    </p>
    """,
    unsafe_allow_html=True,
)
