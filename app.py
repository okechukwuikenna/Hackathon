import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="Farmer Loan Repayment Predictor", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()  # Clean column names
    return df

df = load_data()

# --- Target Column Detection ---
possible_target_names = [col for col in df.columns if col.strip().lower() == 'debt']
if possible_target_names:
    target_col = possible_target_names[0]
else:
    st.error("❌ Error: Column 'Debt' not found in the dataset!")
    st.write("Available columns:", list(df.columns))
    st.stop()

# --- Prepare Features and Encode ---
X = df.drop(columns=[target_col])
y = df[target_col]
categorical_cols = X.select_dtypes(include='object').columns.tolist()
encoder = OrdinalEncoder()
X_encoded = X.copy()
X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# --- Train Model ---
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# --- Streamlit UI ---
st.title("💸 Farmer Loan Repayment Predictor")
st.markdown("Use this tool to predict whether a farmer is likely to repay a loan based on their demographics and economic activity.")

# --- Sidebar: Farmer Input ---
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

# --- Custom Rule Logic ---
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

# --- Prediction ---
if meets_repayment_rules(input_df):
    prediction = "Yes"
    proba = [0.01, 0.99]
else:
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]

# --- Display Prediction ---
st.subheader("🔮 Prediction Result")
confidence = round(max(proba) * 100, 2)
if prediction == "Yes":
    st.success(f"✅ This farmer is **likely to repay** the loan. (Confidence: {confidence}%)")
else:
    st.error(f"⚠️ This farmer is **unlikely to repay** the loan. (Confidence: {confidence}%)")

# --- Download Results ---
result_df = input_df.copy()
result_df["Prediction"] = prediction
result_df["Confidence"] = confidence

csv = result_df.to_csv(index=False).encode()
st.download_button("📥 Download Prediction Result", data=csv, file_name="loan_prediction_result.csv", mime="text/csv")

# --- Dashboard ---
st.markdown("---")
st.subheader("📊 Farmer Dataset Overview")

# --- Visualizations ---
st.markdown("### 📈 Explore Data Patterns")

var_x = st.selectbox("Select X-axis variable", df.columns, index=0)
var_color = st.selectbox("Select color grouping", df.columns, index=1)

fig = px.histogram(
    df, 
    x=var_x, 
    color=var_color, 
    barmode='group',
    title=f"{var_x} grouped by {var_color}",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    category_orders={var_x: sorted(df[var_x].dropna().unique(), key=str)}
)
st.plotly_chart(fig, use_container_width=True)

# --- Target Distribution ---
st.markdown("### 📊 Debt Distribution")
st.bar_chart(df[target_col].value_counts())

# --- Feature Importance ---
st.markdown("### 🔍 Top Features Influencing Repayment")
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
    color_continuous_scale="Agsunset"
)
st.plotly_chart(fig_imp, use_container_width=True)

# --- Tips for Farmers ---
st.markdown("---")
st.subheader("💡 Tips to Improve Loan Eligibility")

st.markdown("""
Improving your eligibility for loan repayment programs can significantly boost your chances of approval. Below are key factors to consider:

- 📄 **Provide complete documentation**: Ensure your **Voter’s Card**, **BVN**, **Tax Invoice**, and **Tax Clearance Certificate** are available.
- 💼 **Stable income**: Try to maintain a regular income level of at least ₦215,000 per month.
- 🌱 **Farm investment**: Regular investment in your farm (tools, seeds, irrigation) shows seriousness and commitment.
- 🏠 **Own agricultural assets**: Land and mechanized tools enhance reliability in farming.
- 🎓 **Education counts**: Farmers with at least secondary education have higher chances.
- 🛡️ **Reduce risk factors**: Minimize drought damage and pest infestation using proper farming techniques and early preparation.
""")

# --- Footer ---
st.markdown("---")
st.markdown("This app predicts whether a farmer will repay a loan based on personal, financial, and agricultural indicators. Use the sidebar to explore different farmer scenarios and visualize trends in the dataset.")
