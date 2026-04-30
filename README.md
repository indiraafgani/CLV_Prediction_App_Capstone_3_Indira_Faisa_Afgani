# 🚗 Customer Lifetime Value Prediction
### Enabling Data-Driven Marketing Strategies for AutoShield Insurance Co.

---

## 📋 Project Overview

This project builds a **supervised regression model** to predict Customer Lifetime Value (CLV) for AutoShield Insurance Co., a motor vehicle insurance company. By accurately predicting CLV, the company can move from intuition-based to data-driven marketing and retention strategies.

**Author:** Indira Faisa Afgani  
**Type:** Capstone Project 3 — Machine Learning (Regression)

---

## 🏢 Business Context

AutoShield Insurance Co. serves a diverse customer base across vehicle classes (Two-Door Cars, SUVs, Sports Cars, Luxury SUVs) with three coverage tiers: Basic, Extended, and Premium. Without a systematic way to quantify customer value, the company faces three critical inefficiencies:

- **Budget Misallocation** — No data-driven basis for acquisition/retention spend
- **Missed Retention Opportunities** — High-value at-risk customers go unidentified
- **One-Size-Fits-All Marketing** — Same treatment regardless of customer value

---

## 🎯 Goals

1. Accurately predict CLV for each customer based on demographics, product types, and payment behavior
2. Identify the key drivers of CLV
3. Segment customers into **Bronze / Silver / Gold / Platinum** tiers
4. Optimize marketing ROI by concentrating retention budgets on high-value segments

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total records (after cleaning) | ~5,634 rows |
| Features | 10 |
| Target variable | Customer Lifetime Value (continuous, in $) |
| CLV range | $1,898 – $83,325 |
| Average CLV | ~$8,004 |

### Features

| Column | Type | Description |
|---|---|---|
| Vehicle Class | Categorical | Type of vehicle (Two-Door Car, SUV, Luxury SUV, etc.) |
| Coverage | Categorical | Insurance coverage tier (Basic, Extended, Premium) |
| Renew Offer Type | Categorical | Renewal offer type (Offer1–Offer4) |
| EmploymentStatus | Categorical | Customer employment status |
| Marital Status | Categorical | Married, Single, Divorced |
| Education | Categorical | Highest education level |
| Number of Policies | Numerical | Total insurance policies held |
| Monthly Premium Auto | Numerical | Monthly premium amount ($) |
| Total Claim Amount | Numerical | Total claims submitted ($) |
| Income | Numerical | Annual income ($); 0 for unemployed customers |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- CLV is highly right-skewed (skewness = 3.06); log-transform applied
- Spearman correlation used (data non-normal, with outliers)
- 618 duplicate rows (10.9%) identified and removed

### 2. Preprocessing (Inside sklearn Pipeline)
- **RobustScaler** for numerical features (robust to outliers)
- **OrdinalEncoder** for ordinal features (Coverage, Vehicle Class, Education)
- **OneHotEncoder** with `drop='first'` for nominal features

### 3. Feature Selection (3 Methods)
- **Spearman Correlation** — monotonic relationship with CLV
- **Mutual Information** — captures non-linear relationships
- **Recursive Feature Elimination (RFE)** — wrapper method with LinearRegression

### 4. Model Benchmarking
Seven models compared with the same pipeline structure (5-fold CV):

| Model | Val R² | Test R² | Overfit Gap |
|---|---|---|---|
| Linear Regression | — | — | — |
| KNN | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | — | — | — |
| **Gradient Boosting** | — | **0.9058** | **0.0149** ✅ |
| LightGBM | — | — | — |
| XGBoost | — | — | — |

> Gradient Boosting selected as best model: highest R², lowest RMSE, lowest overfit gap.

### 5. Hyperparameter Tuning
RandomizedSearchCV with 50 iterations × 5-fold CV applied to Gradient Boosting. Default parameters found to be near-optimal.

### 6. Model Explainability (SHAP)
SHAP values used to explain feature contributions at both global and individual customer levels.

---

## 📈 Final Model Performance

| Metric | Value |
|---|---|
| R² | **0.9060** |
| RMSE (log scale) | 0.2001 |
| MAE (original scale) | **$1,445** |
| MAPE | **9.89%** ✅ |

> Target thresholds: R² > 0.80 ✅ and MAPE < 15% ✅ — both achieved.

---

## 🔍 Key Findings

### Top CLV Drivers (SHAP Analysis)

| Feature | Mean \|SHAP\| | Contribution |
|---|---|---|
| Number of Policies | 0.441 | ~57.8% |
| Monthly Premium Auto | 0.235 | ~30.8% |
| All other features | < 0.02 each | ~11.4% |

> Just 2 features drive ~88% of the model's predictive power.

### Customer Segmentation

| Segment | % of Customers | % of Total CLV | Avg CLV |
|---|---|---|---|
| 🥉 Bronze (<$5K) | 37.3% | 16.7% | $3,324 |
| 🥈 Silver ($5K–$10K) | 38.1% | 35.9% | $7,006 |
| 🥇 Gold ($10K–$20K) | 22.9% | 42.1% | $13,615 |
| 💎 Platinum (>$20K) | 1.7% | 5.4% | $23,671 |

> **Gold + Platinum = 24.6% of customers → 47.5% of total CLV**

---

## 💡 Business Recommendations

1. **Prioritize Gold & Platinum retention** — nearly half of total revenue comes from just ~25% of customers; retention spend here yields maximum ROI
2. **Target multi-policy acquisition** — Number of Policies is the strongest CLV driver; bundle products at onboarding to increase policies per customer
3. **Segment-based budget allocation** — avoid overspending on Bronze customers whose CLV does not justify high acquisition or retention costs

---

## 🛠️ Tech Stack

```
Python 3.x
├── pandas, numpy          — data manipulation
├── matplotlib, seaborn    — visualization
├── scipy                  — statistical analysis
├── scikit-learn           — modeling pipeline, preprocessing, evaluation
│   ├── GradientBoostingRegressor
│   ├── RandomForestRegressor
│   ├── LinearRegression, KNN, DecisionTree
│   ├── Pipeline, ColumnTransformer
│   ├── RandomizedSearchCV
│   └── SelectPercentile (RFE, Mutual Information)
├── lightgbm, xgboost      — additional boosting models
├── shap                   — model explainability
└── joblib                 — model serialization
```

---

## 📁 Project Structure

```
├── Customer_Lifetime_Value_Capstone.ipynb   # Main notebook
├── data_customer_lifetime_value.csv         # Raw dataset
├── data_clv_cleaned.csv                     # Cleaned dataset
├── clv_gradient_boosting_model.joblib       # Saved final model
└── outputs/
    ├── clv_distribution.png
    ├── log_clv_distribution.png
    ├── numerical_features.png
    ├── categorical_vs_clv.png
    ├── correlation_matrix.png
    ├── feature_selection_filter.png
    ├── rfe_results.png
    ├── feature_importance.png
    ├── selectpercentile_experiment.png
    ├── model_benchmarking.png
    ├── learning_curve.png
    ├── shap_summary.png
    ├── shap_bar.png
    ├── final_model_analysis.png
    └── customer_segmentation.png
```

---

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone <repo-url>
   cd clv-prediction
   ```

2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost shap joblib
   ```

3. Place the dataset (`data_customer_lifetime_value.csv`) in the project root

4. Open and run the notebook
   ```bash
   jupyter notebook Customer_Lifetime_Value_Capstone.ipynb
   ```

5. To load the saved model for inference:
   ```python
   import joblib
   model = joblib.load('clv_gradient_boosting_model.joblib')
   clv_log_pred = model.predict(X_new)

   import numpy as np
   clv_pred = np.exp(clv_log_pred)  # convert back from log scale
   ```

---

## 📌 Notes

- The model predicts **log(CLV)**; apply `np.exp()` to convert predictions back to dollar values
- The pipeline handles all preprocessing internally — pass raw (unencoded) features to `model.predict()`
- The final model was retrained on the full dataset (train + test) before saving to maximize learned information
