import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# --- Streamlit page setup ---
st.set_page_config(page_title="Farmer Loan Repayment Predictor", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- Detect 'Debt' column ---
possible_target_names = [col for col in df.columns if col.strip().lower() == 'debt']
if possible_target_names:
    target_col = possible_target_names[0]
else:
    st.error("❌ Error: Column 'Debt' not found in the dataset!")
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
st.title("💸 Farmer Loan Repayment Predictor")
st.markdown("Use this tool to predict whether a farmer is likely to repay a loan based on demographic and economic data.")

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

# --- Custom rule to override prediction ---
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
        row.get("Avg Income Level", "") in ["N215,001 - N315,000 per month", "Above N315,000 per month"] and
        row.get("Own Agri Land", "") != "Do not own" and
        row.get("Own Agri Mech Tool", "") != "Do not own" and
        row.get("Educational Level", "") not in ["No education", "Primary complete"] and
        row.get("Drought Damage", "Yes") == "No" and
        row.get("Pest Infestation", "Yes") == "No"
    )

# --- Prediction ---
if meets_repayment_rules(input_df):
    prediction = "Yes"
    proba = [0.01, 0.99]
else:
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]

# --- Display prediction ---
st.subheader("🔮 Prediction Result")
confidence = round(max(proba) * 100, 2)

if prediction == "Yes":
    st.success(f"✅ This farmer is **likely to repay** the loan. (Confidence: {confidence}%)")
else:
    st.error(f"⚠️ This farmer is **unlikely to repay** the loan. (Confidence: {confidence}%)")

# --- Download prediction result ---
result_df = input_df.copy()
result_df["Prediction"] = prediction
result_df["Confidence (%)"] = confidence

csv = result_df.to_csv(index=False).encode()
st.download_button("📥 Download Prediction Result", data=csv, file_name="loan_prediction_result.csv", mime="text/csv")

# --- Visualization ---
st.markdown("---")
st.subheader("📊 Compare Variables")

var_x = st.selectbox("Select X-axis Variable", df.columns, index=0)
var_color = st.selectbox("Group by (color)", df.columns, index=df.columns.get_loc(target_col))

fig = px.histogram(
    df,
    x=var_x,
    color=var_color,
    barmode='group',
    title=f"{var_x} grouped by {var_color}",
    color_discrete_map={"Yes": "blue", "No": "red"},
    category_orders={var_x: sorted(df[var_x].dropna().unique(), key=str)}
)
st.plotly_chart(fig, use_container_width=True)

# --- Feature importance ---
st.subheader("🔍 Top Features Influencing Repayment")
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig_imp = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation='h',
    title="Feature Importance",
    color="Importance",
    color_continuous_scale="bluered"
)
st.plotly_chart(fig_imp, use_container_width=True)

# --- Tips Section ---
st.markdown("---")
st.subheader("💡 Tips to Improve Loan Eligibility")

st.markdown("""
Improving your eligibility for agricultural loans is essential for building trust with lenders. Here are some tips:

- 📄 **Submit complete documentation**: Including Voter’s Card, BVN, Tax Invoice, and Tax Clearance Certificate.
- 💼 **Maintain steady income**: Aim for consistent income over ₦215,000/month.
- 🌿 **Invest in your farm**: Frequent reinvestment in tools and land boosts credibility.
- 🏡 **Own agricultural property**: Owning land and mechanized tools shows capacity to scale.
- 📘 **Continue learning**: Completing secondary education or beyond helps improve eligibility.
- 🛠️ **Control risks**: Reduce crop loss due to pests and drought by using modern methods.
""")

# --- Footer ---
st.markdown("---")
st.caption("Developed for farmer empowerment and financial inclusion.")
