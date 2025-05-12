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
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Identify target column
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

# Custom rule for repayment

def meets_repayment_rules(user_input_row):
    row = user_input_row.iloc[0]
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

# Eligibility rules

def is_eligible(row_df):
    row = row_df.iloc[0]
    income_str = row.get("Avg Income Level", "")
    income_levels = [
        "Below N15,000 per month",
        "N15,001 - N35,000 per month",
        "N35,001 - N55,000 per month",
        "N55,001 - N115,000 per month",
        "N115,001 - N215,000 per month",
        "N215,001 - N315,000 per month",
        "Above N315,000 per month"
    ]
    min_required_index = income_levels.index("N115,001 - N215,000 per month")
    user_index = income_levels.index(income_str) if income_str in income_levels else -1

    return (
        row.get("Age", 0) >= 21 and
        row.get("BVN", "No") == "Yes" and
        row.get("Debt", "Yes") == "No" and
        row.get("Tax Invoice", "No") == "Yes" and
        user_index >= min_required_index
    )

# --- Prediction ---
st.subheader("\ud83d\udd2e Prediction Result")

if is_eligible(input_df):
    if meets_repayment_rules(input_df):
        prediction = "Yes"
        proba = [0.01, 0.99]
    else:
        prediction = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0]

    confidence = round(max(proba) * 100, 2)
    if prediction == "Yes":
        st.success(f"✅ This farmer is **likely to repay** the loan. (Confidence: {confidence}%)")
    else:
        st.error(f"⚠️ This farmer is **unlikely to repay** the loan. (Confidence: {confidence}%)")

    # Download button
    result_df = input_df.copy()
    result_df["Prediction"] = prediction
    result_df["Confidence (%)"] = confidence
    csv = result_df.to_csv(index=False).encode()
    st.download_button("📥 Download Prediction Result", data=csv, file_name="loan_prediction_result.csv", mime="text/csv")
else:
    st.warning("🚫 The applicant does not meet the **minimum eligibility criteria** to assess loan risk.")
    st.markdown("""
**Required Conditions:**
- Age **21 or older**
- BVN must be **Yes**
- Debt must be **No**
- Tax Invoice must be **Yes**
- Avg Income Level at least **₦115,001/month**
""")

# --- Dashboard ---
st.markdown("---")
st.subheader("📊 Farmer Dataset Overview")

# Select variables to plot
st.markdown("### Visualize Any Two Variables")
var_x = st.selectbox("Select X-axis variable", df.columns)
var_color = st.selectbox("Select color group (e.g., Debt, Gender, etc.)", df.columns)

fig = px.histogram(
    df, x=var_x, color=var_color, barmode='group',
    title=f"{var_x} grouped by {var_color}",
    color_discrete_sequence=px.colors.qualitative.Set1,
    category_orders={var_x: sorted(df[var_x].dropna().unique(), key=str)}
)
st.plotly_chart(fig, use_container_width=True)

# --- Loan Tips ---
st.markdown("### 💡 Tips to Improve Loan Eligibility")
st.info("""
- Maintain accurate financial records and documents like **Tax Invoice** and **Tax Clearance**.
- Register for a **BVN** if you haven't.
- Ensure you have **no outstanding debt** before applying.
- Increase monthly income through stable sources (target above ₦115,000).
- Own agricultural land or mechanized tools to show farm capacity.
""")

st.markdown("---")
st.markdown("This app predicts whether a farmer will repay a loan based on various factors. Use the sidebar to input farmer details and explore relationships in the dataset.")
