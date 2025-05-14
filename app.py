import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import random
import io
from PIL import Image


# --- Streamlit page setup ---
st.set_page_config(page_title="Farmer Loan Repayment Predictor", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- Detect 'Invest Freq' as the new target variable ---
possible_target_names = [col for col in df.columns if col.strip().lower() == 'invest freq']
if possible_target_names:
    target_col = possible_target_names[0]
else:
    st.error("❌ Error: Column 'Invest Freq' not found in the dataset!")
    st.write("Available columns:", list(df.columns))
    st.stop()

# --- Prepare features ---
X = df.drop(columns=[target_col])
y = df[target_col]

# --- Encode categorical columns ---
categorical_cols = X.select_dtypes(include='object').columns.tolist()
encoder = OrdinalEncoder()
X_encoded = X.copy()
X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# --- Train model ---
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# --- UI Header ---
st.title("Farmer Loan Repayment Predictor")
st.markdown("This model uses historical data to estimate the likelihood of a farmer repaying a loan based on demographic and economic features.")

# --- Sidebar inputs ---
st.sidebar.header("📋 Enter Farmer Details")

def user_input():
    input_data = {}
    for col in X.columns:
        if col in categorical_cols:
            options = sorted(df[col].dropna().unique())
            input_data[col] = st.sidebar.selectbox(col, options)
        else:
            input_data[col] = st.sidebar.number_input(col, min_value=0, step=1)
    return pd.DataFrame([input_data])

input_df = user_input()
input_encoded = input_df.copy()
input_encoded[categorical_cols] = encoder.transform(input_df[categorical_cols])

# --- Confidence calculation function ---
def calculate_dynamic_confidence(income):
    """Calculate dynamic confidence based on the farmer's income."""
    if income > 315000:
        return 99.0
    elif income >= 115001 and income <= 135000:
        return 60.0
    elif income > 135000 and income <= 315000:
        # Linear interpolation between 99% and 60% for income between 135,001 and 315,000
        confidence = 99.0 - ((income - 135000) * (39.0 / (315000 - 135000)))
        return confidence
    else:
        return 0.0  # This can be adjusted as per requirement

# --- Prediction ---
predict_button = st.sidebar.button("Predict")

if predict_button:
    income = input_df.iloc[0].get("Avg Income Level", "")
    age = input_df.iloc[0].get("Age", 0)
    bvn = input_df.iloc[0].get("BVN", "No")
    debt = input_df.iloc[0].get("Debt", "Yes")
    tax_invoice = input_df.iloc[0].get("Tax Invoice", "No")

    # Check if the farmer qualifies for the loan based on business rules
    if (
        age >= 21 and
        bvn == "Yes" and
        debt == "No" and
        tax_invoice == "Yes" and
        (
            ("N115,001" in income) or 
            ("N135,001" in income) or 
            ("N155,001" in income) or
            ("N175,001" in income) or 
            ("N195,001" in income) or 
            ("N215,001" in income) or 
            ("N235,001" in income) or 
            ("N255,001" in income) or 
            ("N275,001" in income) or 
            ("N295,001" in income) or 
            ("Above N315,000" in income)
        )
    ):
        # If all business rules are met (including income), approve the loan
        prediction = "Yes"
        proba = [0.01, 0.99]
    else:
        # If conditions are not met, use the model's prediction
        prediction = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0]

    # --- Display prediction ---
    st.subheader("Prediction Result")
    confidence = round(max(proba) * 100, 2)

    # Calculate dynamic confidence based on income
    income_numeric = float(income.split(" ")[0].replace(",", "").replace("N", ""))
    dynamic_confidence = calculate_dynamic_confidence(income_numeric)

    st.write(f"Dynamic Confidence based on Income: {dynamic_confidence}%")

    # Show the final prediction result with dynamic confidence
    if prediction == "Yes":
        st.success(f"✅ This farmer is **likely to repay** the loan. (Confidence: {dynamic_confidence}%)")
    else:
        st.error(f"⚠️ This farmer is **unlikely to repay** the loan. (Confidence: {dynamic_confidence}%)")
        
        # Display the conditions that were not met
        st.markdown("### ⚠️ Loan Approval Conditions Not Met:")
        
        missing_criteria = []

        if age < 21:
            missing_criteria.append("Age must be at least 21 years old.")
        if bvn != "Yes":
            missing_criteria.append("A valid BVN is required.")
        if debt == "Yes":
            missing_criteria.append("Debt must be 'No'.")
        if tax_invoice != "Yes":
            missing_criteria.append("A valid Tax Invoice is required.")
        if not (
            ("N115,001" in income) or 
            ("N135,001" in income) or 
            ("N155,001" in income) or
            ("N175,001" in income) or 
            ("N195,001" in income) or 
            ("N215,001" in income) or 
            ("N235,001" in income) or 
            ("N255,001" in income) or 
            ("N275,001" in income) or 
            ("N295,001" in income) or 
            ("Above N315,000" in income)
        ):
            missing_criteria.append(" Monthly Income must be above ₦114,999 ")

        # Show the list of missing criteria
        for criterion in missing_criteria:
            st.write(f"- {criterion}")

    # --- Display selected input data ---
    st.subheader("📋 Farmer Details (Selected by You)")
    st.write(input_df)

    # --- Download prediction result ---
    result_df = input_df.copy()
    result_df["Prediction"] = prediction
    result_df["Confidence (%)"] = dynamic_confidence
    csv = result_df.to_csv(index=False).encode()
    st.download_button("📥 Download Prediction Result", data=csv, file_name="loan_prediction_result.csv", mime="text/csv")
