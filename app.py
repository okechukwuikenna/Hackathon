import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace
    return df

df = load_data()

# Preprocessing
df.columns = df.columns.str.strip()
categorical_cols = df.select_dtypes(include='object').columns.tolist()
target_col = 'Paying Borrowed'

X = df.drop(columns=[target_col])
y = df[target_col]

encoder = OrdinalEncoder()
X_encoded = X.copy()

try:
    X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])
except KeyError as e:
    st.error(f"Encoding error: {e}")
    st.stop()

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# --- Streamlit UI ---
st.set_page_config(page_title="Farmer Loan Repayment Predictor", layout="wide")

# Load custom CSS
def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ CSS file not found. Skipping custom styles.")

load_css()

# Page title and description
st.title("💸 Farmer Loan Repayment Predictor")
st.markdown("""
    Use this tool to predict whether a farmer is likely to repay a loan based on their demographics and economic activity.
""")

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
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.countplot(y='Avg Income Level', data=df, order=df['Avg Income Level'].value_counts().index, ax=ax1)
    st.pyplot(fig1)

# Ownership of Agricultural Land
with col2:
    st.markdown("**Ownership of Agricultural Land**")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='Own Agri Land', hue='Paying Borrowed', ax=ax2)
    st.pyplot(fig2)

# Gender vs Loan Repayment
st.subheader("Gender vs Loan Repayment")
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x='Gender', hue='Paying Borrowed', ax=ax3)
st.pyplot(fig3)

# Education Level vs Loan Repayment
st.subheader("Education Level vs Loan Repayment")
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x='Educational Level', hue='Paying Borrowed', ax=ax4)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig4)

# Drought Damage vs Loan Repayment
st.subheader("Drought Damage vs Loan Repayment")
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x='Drought Damage', hue='Paying Borrowed', ax=ax5)
st.pyplot(fig5)

# Pest Infestation vs Loan Repayment
st.subheader("Pest Infestation vs Loan Repayment")
fig6, ax6 = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x='Pest Infestation', hue='Paying Borrowed', ax=ax6)
st.pyplot(fig6)

# End
st.markdown("---")
st.markdown("This app predicts whether a farmer will repay a loan based on various factors. Use the sidebar to input farmer details and view predictions and insights.")
