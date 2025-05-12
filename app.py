import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px  # Import Plotly for interactive plots

# --- Streamlit UI ---
# Move set_page_config to the very top of your app
st.set_page_config(page_title="Farmer Loan Repayment Predictor", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
    return df

df = load_data()

# Strip column names
df.columns = df.columns.str.strip()

# Define target
target_col = 'Paying Borrowed'

# Drop target and prepare X, y
X = df.drop(columns=[target_col])
y = df[target_col]

# Get categorical columns from X (after dropping target)
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Encode
encoder = OrdinalEncoder()
X_encoded = X.copy()
X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# --- Streamlit UI ---
st.set_page_config(page_title="Farmer Loan Repayment Predictor", layout="wide")

# Page title and description
st.title("💸 Farmer Loan Repayment Predictor")
st.markdown("""Use this tool to predict whether a farmer is likely to repay a loan based on their demographics and economic activity.""")

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

# Encode input
input_encoded = input_df.copy()
input_encoded[categorical_cols] = encoder.transform(input_df[categorical_cols])

# Prediction
prediction = model.predict(input_encoded)[0]
proba = model.predict_proba(input_encoded)[0]

# Prediction result
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
    fig2 = px.histogram(df, x='Own Agri Land', color='Paying Borrowed', 
                        title="Ownership of Agricultural Land vs Loan Repayment",
                        barmode='stack')
    st.plotly_chart(fig2)

# Gender vs Loan Repayment
st.subheader("Gender vs Loan Repayment")
fig3 = px.histogram(df, x='Gender', color='Paying Borrowed', 
                    title="Gender vs Loan Repayment", barmode='stack')
st.plotly_chart(fig3)

# Education Level vs Loan Repayment
st.subheader("Education Level vs Loan Repayment")
fig4 = px.histogram(df, x='Educational Level', color='Paying Borrowed',
                    title="Education Level vs Loan Repayment", 
                    category_orders={"Educational Level": sorted(df['Educational Level'].dropna().unique())})
st.plotly_chart(fig4)

# Drought Damage vs Loan Repayment
st.subheader("Drought Damage vs Loan Repayment")
fig5 = px.histogram(df, x='Drought Damage', color='Paying Borrowed', 
                    title="Drought Damage vs Loan Repayment", barmode='stack')
st.plotly_chart(fig5)

# Pest Infestation vs Loan Repayment
st.subheader("Pest Infestation vs Loan Repayment")
fig6 = px.histogram(df, x='Pest Infestation', color='Paying Borrowed', 
                    title="Pest Infestation vs Loan Repayment", barmode='stack')
st.plotly_chart(fig6)

# --- End of Dashboard ---
st.markdown("---")
st.markdown("This app predicts whether a farmer will repay a loan based on various factors. Use the sidebar to input farmer details and view predictions and insights.")
