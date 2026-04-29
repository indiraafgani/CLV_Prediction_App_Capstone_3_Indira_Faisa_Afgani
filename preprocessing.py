"""
preprocessing.py
Applies the exact same preprocessing pipeline used during model training.
"""

import numpy as np
import pandas as pd


# ─── Segment thresholds (CLV in original $) ──────────────────────────────────
SEGMENT_BINS   = [0, 5_000, 10_000, 20_000, float("inf")]
SEGMENT_LABELS = ["Bronze", "Silver", "Gold", "Platinum"]


def preprocess_input(
    number_of_policies: int,
    monthly_premium_auto: float,
    total_claim_amount: float,
    vehicle_class: str,
    coverage: str,
    artifact: dict,
) -> pd.DataFrame:
    """
    Transform raw user inputs into the 5-feature scaled DataFrame
    that the trained GradientBoostingRegressor expects.

    Parameters
    ----------
    number_of_policies   : int
    monthly_premium_auto : float
    total_claim_amount   : float
    vehicle_class        : str  – one of artifact['vehicle_order']
    coverage             : str  – one of artifact['coverage_order']
    artifact             : dict – joblib bundle from model.pkl

    Returns
    -------
    pd.DataFrame with shape (1, 5) ready for model.predict()
    """
    ordinal_enc      = artifact["ordinal_enc"]
    scaler           = artifact["scaler"]
    bounds           = artifact["bounds"]
    selected_features = artifact["selected_features"]
    numerical_cols   = artifact["numerical_cols"]   # 4 cols used for scaler

    # ── Step 1: Ordinal-encode Coverage & Vehicle Class ───────────────────────
    # ordinal_enc was fit on [Coverage, Vehicle Class, Education]
    # We pass a dummy Education value (index 0) – it won't be used later.
    enc_input = pd.DataFrame(
        [[coverage, vehicle_class, artifact["coverage_order"][0]]],
        columns=["Coverage", "Vehicle Class", "Education"],
    )
    enc_vals = ordinal_enc.transform(enc_input)
    coverage_enc     = enc_vals[0, 0]   # column 0
    vehicle_class_enc = enc_vals[0, 1]  # column 1

    # ── Step 2: Build a row with ALL 4 numerical cols (needed for scaler) ─────
    # Income was kept during scaler fitting but dropped from selected features.
    # We use 0 as a neutral placeholder – it will be scaled but then discarded.
    row = {
        "Monthly Premium Auto": monthly_premium_auto,
        "Total Claim Amount":   total_claim_amount,
        "Number of Policies":   float(number_of_policies),
        "Income":               0.0,  # placeholder; not a selected feature
    }

    # ── Step 3: IQR capping ───────────────────────────────────────────────────
    for col in numerical_cols:
        lo, hi = bounds[col]
        row[col] = float(np.clip(row[col], lo, hi))

    num_df = pd.DataFrame([row], columns=numerical_cols)

    # ── Step 4: RobustScaler ─────────────────────────────────────────────────
    num_scaled = scaler.transform(num_df)
    num_scaled_df = pd.DataFrame(num_scaled, columns=numerical_cols)

    # ── Step 5: Assemble selected features ───────────────────────────────────
    final = pd.DataFrame(
        {
            "Number of Policies":   [num_scaled_df["Number of Policies"].iloc[0]],
            "Monthly Premium Auto": [num_scaled_df["Monthly Premium Auto"].iloc[0]],
            "VehicleClass_enc":     [vehicle_class_enc],
            "Coverage_enc":         [coverage_enc],
            "Total Claim Amount":   [num_scaled_df["Total Claim Amount"].iloc[0]],
        }
    )

    return final[selected_features]


def predict_clv(processed_df: pd.DataFrame, artifact: dict) -> tuple[float, str]:
    """
    Run inference and return (clv_dollars, segment_label).
    """
    model = artifact["model"]
    log_pred = model.predict(processed_df)[0]
    clv      = float(np.exp(log_pred))

    # Segment assignment
    segment = SEGMENT_LABELS[-1]
    for i, (lo, hi) in enumerate(
        zip(SEGMENT_BINS[:-1], SEGMENT_BINS[1:])
    ):
        if lo < clv <= hi:
            segment = SEGMENT_LABELS[i]
            break

    return clv, segment
