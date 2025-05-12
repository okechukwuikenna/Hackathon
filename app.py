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

@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Identify target column (e.g. "Debt")
possible_target_names = [col for col in df.columns if col.strip().lower() == 'debt']
if possible_target_names:
    target_col = possible_target_names[0]
else:
    st.error("❌ Error: Column 'Debt' not found!")
    st.write("Available columns:", list(df.columns))
    st.stop()

# Prepare data
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical encoding
categorical_cols = X.select_dtypes(include='object').columns.tolist()
encoder = OrdinalEncoder()
X_encoded = X.copy()
X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# --- UI Form ---
st.title("💸 Farmer Loan Repayment Predictor")
st.markdown("Predict if a farmer is likely to repay a loan based on input details.")

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

# --- Rule-based override ---
def meets_repayment_rules(row):
    row = row.iloc[0]
    return (
        row.get("Age", 0) >= 25 and
        row.get("Debt", "Yes") == "No" and
        row.get("Voters Card", "No") == "Yes" and
        row.get("BVN", "No") == "Yes" and
        row.get("Tax Invoice", "No") == "Yes" and
        row.get("Tax Clearance Cert", "No") == "Yes" and
        row.get("Invest Freq", "") in ["Always", "Sometimes"] and
        row.get("Avg Income Level", "") in ["N215,001 - N315,000 per month", "Above N315,000 per month"] and
        row.get("Own Agri Land", "") != "Do not own" and
        row.get("Own Agri Mech Tool", "") != "Do not own" and
        row.get("Educational Level", "") not in ["No education", "Primary complete"] and
        row.get("Drought Damage", "Yes") == "No" and
        row.get("Pest Infestation", "Yes") == "No"
    )

# --- Eligibility TIPS ---
def get_improvement_tips(row):
    tips = []
    if row.get("Debt", "Yes") != "No":
        tips.append("Clear existing debts or obligations.")
    if row.get("Voters Card", "No") != "Yes":
        tips.append("Obtain a Voter’s Card.")
    if row.get("BVN", "No") != "Yes":
        tips.append("Ensure BVN is registered.")
    if row.get("Tax Invoice", "No") != "Yes":
        tips.append("Provide a valid tax invoice.")
    if row.get("Tax Clearance Cert", "No") != "Yes":
        tips.append("Obtain a tax clearance certificate.")
    if row.get("Invest Freq", "") not in ["Always", "Sometimes"]:
        tips.append("Invest in your business more frequently.")
    if row.get("Avg Income Level", "") not in ["N215,001 - N315,000 per month", "Above N315,000 per month"]:
        tips.append("Increase your monthly income if possible.")
    if row.get("Own Agri Land", "") == "Do not own":
        tips.append("Consider acquiring agricultural land.")
    if row.get("Own Agri Mech Tool", "") == "Do not own":
        tips.append("Invest in agricultural tools or machinery.")
    if row.get("Educational Level", "") in ["No education", "Primary complete"]:
        tips.append("Consider basic adult education or training.")
    if row.get("Drought Damage", "Yes") == "Yes":
        tips.append("Mitigate drought risk using irrigation or other techniques.")
    if row.get("Pest Infestation", "Yes") == "Yes":
        tips.append("Prevent or manage pest infestations effectively.")
    return tips

# --- Prediction ---
if meets_repayment_rules(input_df):
    prediction = "Yes"
    proba = [0.01, 0.99]
else:
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]

# --- Results ---
st.subheader("Prediction Result")
if prediction == "Yes":
    st.success(f"✅ Likely to repay the loan. (Confidence: {round(max(proba)*100, 2)}%)")
else:
    st.error(f"⚠️ Unlikely to repay the loan. (Confidence: {round(max(proba)*100, 2)}%)")
    st.markdown("### 💡 Tips to Improve Loan Eligibility")
    for tip in get_improvement_tips(input_df.iloc[0]):
        st.markdown(f"- {tip}")

# --- Dashboard ---
st.markdown("---")
st.subheader("📊 Farmer Dataset Overview")

# Select vars for visualization
st.markdown("### Visualize Any Two Variables")
var_x = st.selectbox("Select X-axis variable", df.columns)
var_color = st.selectbox("Select color group (e.g., Debt, Gender)", df.columns)

fig = px.histogram(df, x=var_x, color=var_color, barmode='group',
                   title=f"{var_x} grouped by {var_color}",
                   category_orders={var_x: sorted(df[var_x].dropna().unique(), key=str)})
st.plotly_chart(fig, use_container_width=True)

# Feature importance
st.markdown("### 🔍 Feature Importance")
feat_importance = model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': feat_importance})
fig2 = px.bar(feat_df.sort_values('Importance', ascending=False), x='Importance', y='Feature', orientation='h')
st.plotly_chart(fig2, use_container_width=True)

# Target summary
st.markdown("### Target Variable Distribution")
st.bar_chart(df[target_col].value_counts())

st.markdown("---")
st.markdown("Use the form to input farmer details and explore insights to improve financial eligibility.")
