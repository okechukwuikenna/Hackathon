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

# Correct the error: 'Debt' should replace 'Paying Borrowed'
target_col = 'Debt'  # Changed from 'Paying Borrowed' to 'Debt'

# Check if target column exists in the DataFrame before proceeding
if target_col not in df.columns:
    st.error(f"Error: Column '{target_col}' not found in the dataset!")
    st.stop()

# Drop target and prepare X, y
X = df.drop(columns=[target_col], errors='ignore')  # Use errors='ignore' to avoid KeyError
y = df[target_col]

# Get categorical columns from X (after dropping target)
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Encode categorical features
encoder = OrdinalEncoder()
X_encoded = X.copy()
X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# --- Streamlit UI ---
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

# Function to determine if the farmer meets the repayment criteria
def meets_repayment_rules(user_input_row):
    """Custom rule to classify a farmer as likely to repay."""
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

# Check if the user meets the rules for loan repayment
if meets_repayment_rules(input_df):
    prediction = "Yes"
    proba = [0.01, 0.99]  # Override to high confidence
else:
    # Prediction from model if rules are not met
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]

# Prediction result
st.subheader("Prediction Result")
if prediction == "Yes":
    st.success(f"✅ This farmer is **likely to repay** the loan. (Confidence: {round(max(proba)*100, 2)}%)")
else:
    st.error(f"⚠️ This farmer is **unlikely to repay** the loan. (Confidence: {round(max(proba)*100, 2)}%)")

# --- Generalized Visualization Function ---
def visualize_variable_comparison(df, column_name, target_col='Debt'):
    """
    Visualize comparison of a given column with the target column and add distinct colors.
    """
    # Ensure that the column is a categorical column
    fig = px.histogram(df, x=column_name, color=target_col, 
                        title=f"{column_name} vs {target_col}", 
                        barmode='stack', 
                        category_orders={column_name: sorted(df[column_name].dropna().unique())})
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)

# --- Dashboard section ---
st.markdown("---")
st.subheader("📊 Farmer Dataset Overview")

col1, col2 = st.columns(2)

# Visualizing different variables against 'Debt'
with col1:
    st.markdown("**Gender vs Debt**")
    visualize_variable_comparison(df, 'Gender')

with col2:
    st.markdown("**Drought Damage vs Debt**")
    visualize_variable_comparison(df, 'Drought Damage')

# More visualizations
st.subheader("Other Variable Comparisons")

visualize_variable_comparison(df, 'Pest Infestation')
visualize_variable_comparison(df, 'Educational Level')
visualize_variable_comparison(df, 'Own Agri Land')
visualize_variable_comparison(df, 'Invest Freq')

# --- End of Dashboard ---
st.markdown("---")
st.markdown("This app predicts whether a farmer will repay a loan based on various factors. Use the sidebar to input farmer details and view predictions and insights.")
