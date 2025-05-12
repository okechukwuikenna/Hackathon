import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px  # Import Plotly for interactive plots

# --- Streamlit UI ---
st.set_page_config(page_title="Farmer Loan Repayment Predictor", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()  # Strip any extra spaces
    return df

df = load_data()

# Display column names for debugging
st.write("Available columns in dataset:", df.columns.tolist())

# Define target column
target_col = 'Debt'

# Check if 'Debt' exists in the dataset
if target_col not in df.columns:
    st.error(f"The '{target_col}' column is not present in the dataset. Please check the data and ensure it is included.")
    st.stop()  # Stop further execution if the column is not found

# Drop target column and prepare features
X = df.drop(columns=[target_col])
y = df[target_col]

# Get categorical columns from X
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Encode categorical columns
encoder = OrdinalEncoder()
X_encoded = X.copy()
X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# --- Streamlit UI ---
st.title("💸 Farmer Loan Repayment Predictor")
st.markdown("""Use this tool to predict whether a farmer is likely to repay a loan based on their demographics and economic activity.""")

# Sidebar for user input
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

# Encode user input
input_encoded = input_df.copy()
input_encoded[categorical_cols] = encoder.transform(input_df[categorical_cols])

# Custom rules for repayment prediction
def meets_repayment_rules(user_input_row):
    row = user_input_row.iloc[0]  # Single-row dataframe
    return (
        row["Age"] >= 25 and
        row["Debt"] == "No" and  # Ensure Debt column is used here
        row["Voters Card"] == "Yes" and
        row["BVN"] == "Yes" and
        row["Tax Invoice"] == "Yes" and
        row["Tax Clearance Cert"] == "Yes" and
        row["Invest Freq"] in ["Always", "Sometimes"] and
        row["Avg Income Level"] in ["N215,001 - N315,000 per month", "Above N315,000 per month"] and
        row["Own Agri Land"] != "Do not own" and
        row["Own Agri Mech Tool"] != "Do not own" and
        row["Educational Level"] not in ["No education", "Primary complete"] and
        row["Drought Damage"] == "No" and
        row["Pest Infestation"] == "No"
    )

# Check if user input meets repayment rules
if meets_repayment_rules(input_df):
    prediction = "Yes"
    proba = [0.01, 0.99]  # Override to high confidence
else:
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]

# Display prediction result
st.subheader("Prediction Result")
if prediction == "Yes":
    st.success(f"✅ This farmer is **likely to repay** the loan. (Confidence: {round(max(proba)*100, 2)}%)")
else:
    st.error(f"⚠️ This farmer is **unlikely to repay** the loan. (Confidence: {round(max(proba)*100, 2)}%)")

# --- Dashboard section ---
st.markdown("---")
st.subheader("📊 Farmer Dataset Overview")

col1, col2 = st.columns(2)

# Income Level Distribution
with col1:
    st.markdown("**Income Level Distribution**")
    fig1 = px.histogram(df, y='Avg Income Level', color='Avg Income Level', 
                        title="Income Level Distribution", 
                        category_orders={"Avg Income Level": sorted(df['Avg Income Level'].dropna().unique())})
    st.plotly_chart(fig1)

# Ownership of Agricultural Land
with col2:
    st.markdown("**Ownership of Agricultural Land**")
    fig2 = px.histogram(df, x='Own Agri Land', color='Debt',  # Use 'Debt' column here
                        title="Ownership of Agricultural Land vs Loan Repayment",
                        barmode='stack')
    st.plotly_chart(fig2)

# Gender vs Loan Repayment
st.subheader("Gender vs Loan Repayment")
fig3 = px.histogram(df, x='Gender', color='Debt',  # Use 'Debt' column here
                    title="Gender vs Loan Repayment", barmode='stack')
st.plotly_chart(fig3)

# Education Level vs Loan Repayment
st.subheader("Education Level vs Loan Repayment")
fig4 = px.histogram(df, x='Educational Level', color='Debt',  # Use 'Debt' column here
                    title="Education Level vs Loan Repayment", 
                    category_orders={"Educational Level": sorted(df['Educational Level'].dropna().unique())})
st.plotly_chart(fig4)

# Drought Damage vs Loan Repayment
st.subheader("Drought Damage vs Loan Repayment")
fig5 = px.histogram(df, x='Drought Damage', color='Debt',  # Use 'Debt' column here
                    title="Drought Damage vs Loan Repayment", barmode='stack')
st.plotly_chart(fig5)

# Pest Infestation vs Loan Repayment
st.subheader("Pest Infestation vs Loan Repayment")
fig6 = px.histogram(df, x='Pest Infestation', color='Debt',  # Use 'Debt' column here
                    title="Pest Infestation vs Loan Repayment", barmode='stack')
st.plotly_chart(fig6)

# --- End of Dashboard ---
st.markdown("---")
st.markdown("This app predicts whether a farmer will repay a loan based on various factors. Use the sidebar to input farmer details and view predictions and insights.")
