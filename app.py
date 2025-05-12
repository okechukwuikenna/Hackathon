import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# --- Streamlit UI ---
st.set_page_config(page_title="Farmer Loan Repayment Predictor", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Identify correct target column
possible_target_names = [col for col in df.columns if col.strip().lower() == 'debt']
if possible_target_names:
    target_col = possible_target_names[0]
else:
    st.error("❌ Error: Column 'Debt' not found in the dataset!")
    st.write("Available columns:", list(df.columns))
    st.stop()

# Prepare data
X = df.drop(columns=[target_col])
y = df[target_col]
categorical_cols = X.select_dtypes(include='object').columns.tolist()

encoder = OrdinalEncoder()
X_encoded = X.copy()
X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])

model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# --- UI: Input form ---
st.title("💸 Farmer Loan Repayment Predictor")
st.markdown("Use this tool to predict whether a farmer is likely to repay a loan based on demographics and economic activity.")

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

# --- Custom rules for eligibility ---
def meets_repayment_rules(row_df):
    row = row_df.iloc[0]
    return (
        row.get("Age", 0) >= 25 and
        row.get("Debt", "Yes") == "No" and
        row.get("Voters Card", "No") == "Yes" and
        row.get("BVN", "No") == "Yes" and
        row.get("Tax Invoice", "No") == "Yes" and
        row.get("Tax Clearance Cert", "No") == "Yes" and
        row.get("Invest Freq", "") in ["Always", "Sometimes"] and
        row.get("Avg Income Level", "") in [
            "N215,001 - N315,000 per month", "Above N315,000 per month"
        ] and
        row.get("Own Agri Land", "") != "Do not own" and
        row.get("Own Agri Mech Tool", "") != "Do not own" and
        row.get("Educational Level", "") not in ["No education", "Primary complete"] and
        row.get("Drought Damage", "Yes") == "No" and
        row.get("Pest Infestation", "Yes") == "No"
    )

# --- Generate tips for improving eligibility ---
def get_improvement_tips(row):
    tips = []
    if row.get("Debt", "Yes") == "Yes":
        tips.append("Clear outstanding debt.")
    if row.get("Voters Card", "No") != "Yes":
        tips.append("Obtain a Voters Card.")
    if row.get("BVN", "No") != "Yes":
        tips.append("Get a Bank Verification Number (BVN).")
    if row.get("Tax Invoice", "No") != "Yes":
        tips.append("Ensure you have a Tax Invoice.")
    if row.get("Tax Clearance Cert", "No") != "Yes":
        tips.append("Obtain a Tax Clearance Certificate.")
    if row.get("Invest Freq", "") not in ["Always", "Sometimes"]:
        tips.append("Increase frequency of investing.")
    if row.get("Avg Income Level", "") not in [
        "N215,001 - N315,000 per month", "Above N315,000 per month"]:
        tips.append("Increase monthly income level.")
    if row.get("Own Agri Land", "") == "Do not own":
        tips.append("Consider acquiring agricultural land.")
    if row.get("Own Agri Mech Tool", "") == "Do not own":
        tips.append("Invest in agricultural mechanization tools.")
    if row.get("Educational Level", "") in ["No education", "Primary complete"]:
        tips.append("Consider furthering your education.")
    if row.get("Drought Damage", "Yes") == "Yes":
        tips.append("Use drought-resistant techniques or crops.")
    if row.get("Pest Infestation", "Yes") == "Yes":
        tips.append("Use pest control methods or tools.")
    return tips

# --- Prediction ---
if meets_repayment_rules(input_df):
    prediction = "Yes"
    proba = [0.01, 0.99]
else:
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]

# --- Display results ---
st.subheader("Prediction Result")
if prediction == "Yes":
    st.success(f"✅ This farmer is **likely to repay** the loan. (Confidence: {round(max(proba)*100, 2)}%)")
else:
    st.error(f"⚠️ This farmer is **unlikely to repay** the loan. (Confidence: {round(max(proba)*100, 2)}%)")
    st.markdown("### 💡 Tips to Improve Loan Eligibility")
    for tip in get_improvement_tips(input_df.iloc[0]):
        st.markdown(f"- {tip}")

# --- Download Result as CSV ---
result_data = input_df.copy()
result_data["Prediction"] = prediction
result_data["Confidence (%)"] = round(max(proba) * 100, 2)
if prediction == "No":
    result_data["Tips"] = "; ".join(get_improvement_tips(input_df.iloc[0]))
else:
    result_data["Tips"] = "Eligible - No tips needed."

csv = result_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📅 Download Prediction Result",
    data=csv,
    file_name="loan_prediction_result.csv",
    mime="text/csv"
)

# --- Data Exploration Section ---
st.markdown("---")
st.subheader("📊 Farmer Dataset Overview")

# Dynamic Visuals
st.markdown("### Visualize Any Two Variables")
var_x = st.selectbox("X-axis", df.columns)
var_color = st.selectbox("Group by (Color)", df.columns)

fig = px.histogram(df, x=var_x, color=var_color, barmode='group',
                   title=f"{var_x} grouped by {var_color}")
st.plotly_chart(fig, use_container_width=True)

# Pie Chart
st.markdown("### Distribution of Loan Repayment (Pie Chart)")
pie_fig = px.pie(df, names=target_col, title="Loan Repayment Distribution")
st.plotly_chart(pie_fig, use_container_width=True)

# Feature Importance
st.markdown("### 🔍 Feature Importance: What Influences Loan Repayment Most")
importances = model.feature_importances_
feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

importance_fig = px.bar(feat_df.head(10), x="Importance", y="Feature", orientation="h",
                        title="Top 10 Influential Features")
st.plotly_chart(importance_fig, use_container_width=True)
