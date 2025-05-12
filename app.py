import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# --- Streamlit UI ---
st.set_page_config(page_title="Farmer Loan Repayment Predictor", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()  # Clean up column names
    return df

df = load_data()

# DEBUG: Show columns
# st.write("Raw column names:", list(df.columns))

# Try to automatically find the correct 'Debt' column name
possible_target_names = [col for col in df.columns if col.strip().lower() == 'debt']
if possible_target_names:
    target_col = possible_target_names[0]
else:
    st.error("❌ Error: Column 'Debt' not found in the dataset!")
    st.write("Available columns:", list(df.columns))
    st.stop()

# Prepare features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
encoder = OrdinalEncoder()
X_encoded = X.copy()
X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# --- Streamlit UI ---
st.title("💸 Farmer Loan Repayment Predictor")
st.markdown("Use this tool to predict whether a farmer is likely to repay a loan based on their demographics and economic activity.")

# Sidebar - Input form
st.sidebar.header("Enter Farmer Details")

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

# --- Custom rule to override prediction if conditions are met ---
def meets_repayment_rules(user_input_row):
    row = user_input_row.iloc[0]
    return (
        row.get("Age", 0) >= 21 and
        row.get("Debt", "Yes") == "No" and
        row.get("BVN", "No") == "Yes" and
        row.get("Tax Invoice", "No") == "Yes" and
        row.get("Avg Income Level", "") in ["N115,001 - N135,000 per month", "Above N135,000 per month"]
    )

# Prediction
if meets_repayment_rules(input_df):
    prediction = "Yes"
    proba = [0.01, 0.99]
else:
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]

# Output
st.subheader("Prediction Result")
if prediction == "Yes":
    st.success(f"✅ This farmer is **likely to repay** the loan. (Confidence: {round(max(proba)*100, 2)}%)")
else:
    st.error(f"⚠️ This farmer is **unlikely to repay** the loan. (Confidence: {round(max(proba)*100, 2)}%)")

# --- Dashboard ---
st.markdown("---")
st.subheader("📊 Farmer Dataset Overview")

# Select variables to plot
st.markdown("### Visualize Any Two Variables")
var_x = st.selectbox("Select X-axis variable", df.columns)
var_color = st.selectbox("Select color group (e.g., Debt, Gender, etc.)", df.columns)

fig = px.histogram(df, x=var_x, color=var_color, barmode='group',
                   title=f"{var_x} grouped by {var_color}",
                   category_orders={var_x: sorted(df[var_x].dropna().unique(), key=str)})
st.plotly_chart(fig, use_container_width=True)

# Optional: Summary counts
st.markdown("### Target Variable Distribution")
st.bar_chart(df[target_col].value_counts())

# Loan eligibility tips section
st.markdown("### Loan Eligibility Tips")
st.markdown("""
Here are some tips to improve your eligibility for a loan:

- **Age**: Ensure the applicant is at least 21 years old to be eligible.
- **Debt Status**: Ensure no existing debt (Debt = 'No').
- **Tax Invoice**: A valid Tax Invoice is required.
- **BVN**: A valid BVN (Bank Verification Number) is required.
- **Income Level**: Aim for an income level above N115,000 per month for better chances.
""")

# --- End of Dashboard ---
st.markdown("---")
st.markdown("This app predicts whether a farmer will repay a loan based on various factors. Use the sidebar to input farmer details and explore relationships in the dataset.")
