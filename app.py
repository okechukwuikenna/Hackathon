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
    st.error("\u274c Error: Column 'Debt' not found in the dataset!")
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
st.title("\ud83d\udcb8 Farmer Loan Repayment Predictor")
st.markdown("Use this tool to predict whether a farmer is likely to repay a loan based on demographic and economic data.")

# --- Sidebar inputs ---
st.sidebar.header("\ud83d\udcdc Enter Farmer Details")

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

# --- Eligibility function ---
def is_eligible(row_df):
    row = row_df.iloc[0]
    return (
        row.get("Age", 0) >= 21 and
        row.get("BVN", "No") == "Yes" and
        row.get("Debt", "Yes") == "No" and
        row.get("Tax Invoice", "No") == "Yes" and
        row.get("Avg Income Level", "") in [
            "N115,001 - N215,000 per month",
            "N215,001 - N315,000 per month",
            "Above N315,000 per month"
        ]
    )

# --- Prediction ---
st.subheader("\ud83d\udd2e Prediction Result")
if not is_eligible(input_df):
    st.warning("\ud83d\udeab This farmer **does not meet the required eligibility conditions** for prediction:")
    st.markdown("""
    - Age 21 or above  
    - BVN is provided  
    - No existing debt  
    - Has Tax Invoice  
    - Income level above \u20a6115,000/month  
    """)
    st.error("\u274c This farmer is **not likely to repay** the loan based on missing mandatory criteria.")
else:
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]
    confidence = round(max(proba) * 100, 2)

    if prediction == "Yes":
        st.success(f"\u2705 This farmer is **likely to repay** the loan. (Confidence: {confidence}%)")
    else:
        st.error(f"\u26a0\ufe0f This farmer is **unlikely to repay** the loan. (Confidence: {confidence}%)")

    result_df = input_df.copy()
    result_df["Prediction"] = prediction
    result_df["Confidence (%)"] = confidence
    csv = result_df.to_csv(index=False).encode()
    st.download_button("\ud83d\udcc5 Download Prediction Result", data=csv, file_name="loan_prediction_result.csv", mime="text/csv")

# --- Visualization ---
st.markdown("---")
st.subheader("\ud83d\udcca Compare Variables")

chart_type = st.selectbox("Choose Chart Type", ["Histogram", "Boxplot", "Scatter"])
var_x = st.selectbox("X-axis Variable", df.columns, index=0)
var_color = st.selectbox("Group by (color)", df.columns, index=df.columns.get_loc(target_col))
var_y = st.selectbox("Y-axis Variable (if applicable)", df.columns, index=1)

if chart_type == "Histogram":
    fig = px.histogram(
        df,
        x=var_x,
        color=var_color,
        barmode='group',
        color_discrete_map={"Yes": "blue", "No": "red"},
        category_orders={var_x: sorted(df[var_x].dropna().unique(), key=str)}
    )
elif chart_type == "Boxplot":
    fig = px.box(df, x=var_x, y=var_y, color=var_color, color_discrete_map={"Yes": "blue", "No": "red"})
elif chart_type == "Scatter":
    fig = px.scatter(df, x=var_x, y=var_y, color=var_color, color_discrete_map={"Yes": "blue", "No": "red"})

st.plotly_chart(fig, use_container_width=True)

# --- Feature importance ---
st.subheader("\ud83d\udd0d Top Features Influencing Repayment")
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
st.subheader("\ud83d\udca1 Tips to Improve Loan Eligibility")

st.markdown("""
Improving your eligibility for agricultural loans is essential for building trust with lenders. Here are some tips:

- \ud83d\udcc4 **Submit complete documentation**: Including Voter’s Card, BVN, Tax Invoice, and Tax Clearance Certificate.
- \ud83d\udcbc **Maintain steady income**: Aim for consistent income over ₦215,000/month.
- \ud83c\udf3f **Invest in your farm**: Frequent reinvestment in tools and land boosts credibility.
- \ud83c\udfe1 **Own agricultural property**: Owning land and mechanized tools shows capacity to scale.
- \ud83d\udcd8 **Continue learning**: Completing secondary education or beyond helps improve eligibility.
- \u2692\ufe0f **Control risks**: Reduce crop loss due to pests and drought by using modern methods.
""")

# --- Footer ---
st.markdown("---")
st.caption("Developed for farmer empowerment and financial inclusion.")
