import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import random

# --- Streamlit page setup ---
st.set_page_config(page_title="Farmer Loan Repayment Predictor", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
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

# --- Custom compulsory conditions for eligibility ---
def meets_repayment_rules(row_df):
    row = row_df.iloc[0]
    return (
        row.get("Age", 0) >= 21 and
        row.get("Debt", "Yes") == "No" and
        row.get("BVN", "No") == "Yes" and
        row.get("Tax Invoice", "No") == "Yes" and
        row.get("Avg Income Level", "") in [
            "N115,001 - N215,000 per month",
            "N215,001 - N315,000 per month",
            "Above N315,000 per month"
        ]
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
st.subheader("📊 Dynamic Variable Comparison")

# Select variables for visualization
var_x = st.selectbox("Select X-axis Variable", df.columns, key="x")
var_color = st.selectbox("Group by (color)", df.columns, index=df.columns.get_loc(target_col), key="color")
var_facet = st.selectbox("Split by (facet column)", ["None"] + list(df.columns), key="facet")
chart_type = st.selectbox(
    "Choose Chart Type",
    ["Bar", "Column", "Scatter", "Line", "Box", "Violin", "Pie", "Histogram", "Heatmap"]
)

# Generate a random color palette
unique_values = df[var_color].dropna().unique()
color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
random.shuffle(color_palette)
color_map = {val: color_palette[i % len(color_palette)] for i, val in enumerate(unique_values)}

fig = None

# --- Dynamic chart rendering ---
if chart_type == "Bar":
    fig = px.bar(
        df,
        x=var_x,
        color=var_color,
        title=f"Bar Chart: {var_x} grouped by {var_color}",
        barmode='group',
        color_discrete_map=color_map,
        facet_col=var_facet if var_facet != "None" else None
    )

elif chart_type == "Column":
    fig = px.histogram(
        df,
        x=var_color,
        color=var_x,
        title=f"Column Chart: {var_color} vs {var_x}",
        barmode='group',
        color_discrete_map=color_map,
        facet_col=var_facet if var_facet != "None" else None
    )

elif chart_type == "Scatter":
    fig = px.scatter(
        df,
        x=var_x,
        y=var_facet if var_facet != "None" else var_x,  # Use `var_x` as fallback if `var_facet` is None
        color=var_color,
        title=f"Scatter Plot: {var_x} vs {var_facet} colored by {var_color}",
        color_discrete_map=color_map
    )

elif chart_type == "Line":
    fig = px.line(
        df.sort_values(by=var_x),
        x=var_x,
        y=var_facet if var_facet != "None" else var_x,
        color=var_color,
        title=f"Line Chart: {var_facet if var_facet != 'None' else var_x} over {var_x} grouped by {var_color}",
        color_discrete_map=color_map
    )

elif chart_type == "Box":
    fig = px.box(
        df,
        x=var_color,
        y=var_x,
        color=var_color,
        title=f"Box Plot: {var_x} grouped by {var_color}",
        color_discrete_map=color_map
    )

elif chart_type == "Violin":
    fig = px.violin(
        df,
        x=var_color,
        y=var_x,
        color=var_color,
        box=True,
        title=f"Violin Plot: {var_x} grouped by {var_color}",
        color_discrete_map=color_map
    )

elif chart_type == "Pie":
    pie_data = df[var_color].value_counts().reset_index()
    pie_data.columns = [var_color, 'Count']
    fig = px.pie(
        pie_data,
        names=var_color,
        values='Count',
        title=f"Pie Chart of {var_color}",
        color=var_color,
        color_discrete_map=color_map
    )

elif chart_type == "Histogram":
    fig = px.histogram(
        df,
        x=var_x,
        color=var_color,
        facet_col=var_facet if var_facet != "None" else None,
        barmode='group',
        title=f"Histogram: {var_x} grouped by {var_color} and split by {var_facet if var_facet != 'None' else ''}",
        color_discrete_map=color_map,
        category_orders={var_x: sorted(df[var_x].dropna().unique(), key=str)}
    )

elif chart_type == "Heatmap":
    try:
        corr_df = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(
            corr_df,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Heatmap of Numeric Feature Correlation"
        )
    except Exception as e:
        st.warning("⚠️ Heatmap requires only numeric features.")
        st.write(str(e))

# --- Show chart ---
if fig:
    fig.update_layout(height=600)
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
- 💼 **Maintain steady income**: Aim for consistent
