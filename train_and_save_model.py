"""
train_and_save_model.py
Run this script once to generate model.pkl from the original dataset.
Usage: python train_and_save_model.py
Requires: data_customer_lifetime_value.csv in the same directory.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data_customer_lifetime_value.csv")

# ── 2. Cleaning ───────────────────────────────────────────────────────────────
df_clean = df.drop_duplicates().reset_index(drop=True)
df_clean["Number of Policies"] = df_clean["Number of Policies"].astype(int)

# ── 3. Log-transform target ───────────────────────────────────────────────────
df_clean["CLV_log"] = np.log(df_clean["Customer Lifetime Value"])

# ── 4. Ordinal encoding ───────────────────────────────────────────────────────
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

# ── 5. One-hot encoding ───────────────────────────────────────────────────────
df_clean = pd.get_dummies(
    df_clean,
    columns=["Renew Offer Type", "EmploymentStatus", "Marital Status"],
    drop_first=True,
    dtype=int,
)

# ── 6. Define X / y ──────────────────────────────────────────────────────────
cols_to_drop = ["Customer Lifetime Value", "Coverage", "Vehicle Class",
                "Education", "CLV_log"]
X = df_clean.drop(cols_to_drop, axis=1)
y = df_clean["CLV_log"]

# ── 7. Train-test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 8. IQR capping (fit on train) ─────────────────────────────────────────────
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

# ── 9. RobustScaler (fit on train) ────────────────────────────────────────────
scaler = RobustScaler()
X_train_capped[numerical_cols] = scaler.fit_transform(
    X_train_capped[numerical_cols]
)

# ── 10. Select final features ─────────────────────────────────────────────────
selected_features = [
    "Number of Policies",
    "Monthly Premium Auto",
    "VehicleClass_enc",
    "Coverage_enc",
    "Total Claim Amount",
]
X_train_selected = X_train_capped[selected_features]

# ── 11. Train final model ─────────────────────────────────────────────────────
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# ── 12. Bundle & save ─────────────────────────────────────────────────────────
artifact = {
    "model":           model,
    "ordinal_enc":     ordinal_enc,
    "scaler":          scaler,
    "bounds":          bounds,
    "selected_features": selected_features,
    "numerical_cols":  numerical_cols,
    "coverage_order":  coverage_order,
    "vehicle_order":   vehicle_order,
}

joblib.dump(artifact, "model.pkl")
print("✅  model.pkl saved successfully.")
print(f"    Selected features : {selected_features}")
print(f"    Model R² (train)  : {model.score(X_train_selected, y_train):.4f}")
