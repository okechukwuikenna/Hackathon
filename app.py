import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import random
import io
from PIL import Image


# --- Streamlit page setup ---
# --- Page Configuration ---
st.set_page_config(page_title="AgriConnect", layout="wide")

import streamlit as st

# --- App Title & Subtitle ---
st.markdown("""
# AgriConnect
### *The bridge between young farmers and Financiers*
""")
# Create placeholders
prediction_placeholder = st.empty()
image_placeholder = st.empty()

image_placeholder.image('https://raw.githubusercontent.com/okechukwuikenna/Hackathon/main/young%20farmers.jpg', 
                        caption="Young Farmers", 
                        use_container_width=True)

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- Detect 'Invest Freq' as the new target variable ---
possible_target_names = [col for col in df.columns if col.strip().lower() == 'invest freq']
if possible_target_names:
    target_col = possible_target_names[0]
else:
    st.error("❌ Error: Column 'Invest Freq' not found in the dataset!")
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
st.markdown("This model uses historical data to estimate the likelihood of a farmer repaying a loan based on demographic and economic features.")

# --- Sidebar inputs ---
st.sidebar.header("Enter Farmer Details To Predict Loan Eligibility")
dark_mode = st.sidebar.checkbox("Enable Dark Mode", value=False, help="Toggle dark/light theme for charts")


# Map to Plotly template
theme_template = "plotly_dark" if dark_mode else "plotly_white"

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

def meets_repayment_rules(row_df):
    row = row_df.iloc[0]
    return (
        row.get("Age", 0) >= 21 and
        row.get("Debt", "Yes") == "No" and
        row.get("BVN", "No") == "Yes" and
        row.get("Tax Invoice", "No") == "Yes" and
        any(income in row.get("Avg Income Level", "") for income in [
            "N115,001", "N215,001", "Above N315,000"
        ])
    )

# --- Prediction ---
predict_button = st.sidebar.button("Predict")

if predict_button:
    income = input_df.iloc[0].get("Avg Income Level", "")
    age = input_df.iloc[0].get("Age", 0)
    bvn = input_df.iloc[0].get("BVN", "No")
    debt = input_df.iloc[0].get("Debt", "Yes")
    tax_invoice = input_df.iloc[0].get("Tax Invoice", "No")

    # Check if the farmer qualifies for the loan based on business rules
    if (
        age >= 21 and
        bvn == "Yes" and
        debt == "No" and
        tax_invoice == "Yes" and
        (
            ("N115,001" in income) or 
            ("N135,001" in income) or 
            ("N155,001" in income) or
            ("N175,001" in income) or
            ("N195,001" in income) or 
            ("N215,001" in income) or 
            ("N235,001" in income) or 
            ("N255,001" in income) or 
            ("N275,001" in income) or 
            ("N295,001" in income) or 
            ("Above N315,000" in income)
        )
    ):
        # If all business rules are met (including income), approve the loan
        prediction = "Yes"
        proba = [0.01, 0.99]
    else:
        # If conditions are not met, use the model's prediction
        prediction = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0]

if predict_button:
    # ... your existing prediction code ...

    with prediction_placeholder.container():
        st.subheader("Prediction Result")
        confidence = round(max(proba) * 100, 2)

        if prediction == "Yes":
            st.success(f"✅ This farmer is **likely to repay** the loan. (Confidence: {confidence}%)")
        else:
            st.error(f"⚠️ This farmer is **unlikely to repay** the loan. (Confidence: {confidence}%)")
            
            st.markdown("### ⚠️ Loan Approval Conditions Not Met:")
            missing_criteria = []

            if age < 21:
                missing_criteria.append("Age must be at least 21 years old.")
            if bvn != "Yes":
                missing_criteria.append("A valid BVN is required.")
            if debt == "Yes":
                missing_criteria.append("You MUST pay your previous debt.")
            if tax_invoice != "Yes":
                missing_criteria.append("A valid Tax Invoice is required.")
            if not (
                ("N115,001" in income) or 
                ("N135,001" in income) or 
                ("N155,001" in income) or
                ("N175,001" in income) or 
                ("N195,001" in income) or 
                ("N215,001" in income) or 
                ("N235,001" in income) or 
                ("N255,001" in income) or 
                ("N275,001" in income) or 
                ("N295,001" in income) or 
                ("Above N315,000" in income)
            ):
                missing_criteria.append(" Monthly Income must be above ₦114,999 ")

            for criterion in missing_criteria:
                st.write(f"- {criterion}")

        st.subheader("Farmer Details (Selected by You)")
        st.write(input_df)

        result_df = input_df.copy()
        result_df["Prediction"] = prediction
        result_df["Confidence (%)"] = confidence
        csv = result_df.to_csv(index=False).encode()
        st.download_button(" ⬇️ Download Prediction Result", data=csv, file_name="loan_prediction_result.csv", mime="text/csv")

    
# --- Visualization ---
st.markdown("---")
st.subheader("Explore feature relationships")

columns_with_none = ["None"] + list(df.columns)

# Selectors with tooltips
var_x = st.selectbox("Select X-axis Variable", columns_with_none, key="x", help="Choose the variable for the X-axis")
var_y = st.selectbox("Select Y-axis Variable", columns_with_none, key="y", help="Choose the Y-axis variable (ignored for Pie/Donut)")
var_color = st.selectbox("Group by (color)", columns_with_none, key="color", help="Optional: Group/Color by a categorical column")
var_facet = st.selectbox(
    "Split by (facet column)",
    ["None", "Gender"],
    key="facet",
    help="Optional: Split chart into subplots"
)

chart_type = st.selectbox(
    "Choose Chart Type",
    ["None", "Bar", "Column", "Scatter", "Line", "Box", "Violin", "Pie", "Donut", "Histogram", "Heatmap"],
    help="Select the chart type you want to render"
)

# Generate color map
if var_color != "None":
    unique_values = df[var_color].dropna().unique()
    color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    random.shuffle(color_palette)
    color_map = {val: color_palette[i % len(color_palette)] for i, val in enumerate(unique_values)}
else:
    color_map = {}

fig = None

# --- Chart Logic ---
if chart_type != "None":
    try:
        if chart_type == "Bar":
            fig = px.bar(
                df,
                x=var_x if var_x != "None" else None,
                y=var_y if var_y != "None" else None,
                color=var_color if var_color != "None" else None,
                facet_col=var_facet if var_facet != "None" else None,
                barmode='group',
                title="Bar Chart",
                color_discrete_map=color_map
            )

        elif chart_type == "Column":
            fig = px.histogram(
                df,
                x=var_x if var_x != "None" else None,
                y=var_y if var_y != "None" else None,
                color=var_color if var_color != "None" else None,
                facet_col=var_facet if var_facet != "None" else None,
                barmode='group',
                title="Column Chart",
                color_discrete_map=color_map
            )

        elif chart_type == "Scatter":
            fig = px.scatter(
                df,
                x=var_x if var_x != "None" else None,
                y=var_y if var_y != "None" else None,
                color=var_color if var_color != "None" else None,
                facet_col=var_facet if var_facet != "None" else None,
                title="Scatter Plot",
                color_discrete_map=color_map
            )

        elif chart_type == "Line":
            fig = px.line(
                df.sort_values(by=var_x if var_x != "None" else df.columns[0]),
                x=var_x if var_x != "None" else None,
                y=var_y if var_y != "None" else None,
                color=var_color if var_color != "None" else None,
                facet_col=var_facet if var_facet != "None" else None,
                title="Line Chart",
                color_discrete_map=color_map
            )

        elif chart_type == "Box":
            fig = px.box(
                df,
                x=var_x if var_x != "None" else None,
                y=var_y if var_y != "None" else None,
                color=var_color if var_color != "None" else None,
                title="Box Plot",
                color_discrete_map=color_map
            )

        elif chart_type == "Violin":
            fig = px.violin(
                df,
                x=var_x if var_x != "None" else None,
                y=var_y if var_y != "None" else None,
                color=var_color if var_color != "None" else None,
                box=True,
                title="Violin Plot",
                color_discrete_map=color_map
            )

        elif chart_type == "Pie":
            pie_col = var_color if var_color != "None" else var_x
            if pie_col != "None":
                pie_data = df[pie_col].value_counts().reset_index()
                pie_data.columns = ['Value', 'Count']
                fig = px.pie(
                    pie_data,
                    names='Value',
                    values='Count',
                    title="Pie Chart",
                    color='Value',
                    color_discrete_map=color_map
                )
            else:
                st.warning("⚠️ Please select at least a Group by or X-axis for the Pie chart.")

        elif chart_type == "Donut":
            donut_col = var_color if var_color != "None" else var_x
            if donut_col != "None":
                donut_data = df[donut_col].value_counts().reset_index()
                donut_data.columns = ['Value', 'Count']
                fig = px.pie(
                    donut_data,
                    names='Value',
                    values='Count',
                    title="Donut Chart",
                    hole=0.5,
                    color='Value',
                    color_discrete_map=color_map
                )
            else:
                st.warning("⚠️ Please select at least a Group by or X-axis for the Donut chart.")

        elif chart_type == "Histogram":
            fig = px.histogram(
                df,
                x=var_x if var_x != "None" else None,
                color=var_color if var_color != "None" else None,
                facet_col=var_facet if var_facet != "None" else None,
                title="Histogram",
                barmode='group',
                color_discrete_map=color_map
            )

        elif chart_type == "Heatmap":
            corr_df = df.select_dtypes(include=[np.number]).corr()
            fig = px.imshow(
                corr_df,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Heatmap of Numeric Feature Correlation"
            )

    except Exception as e:
        st.error("❌ Failed to render the chart. Please check your selections.")
        st.exception(e)

if fig:
    fig.update_layout(template=theme_template, height=600)
    st.plotly_chart(fig, use_container_width=True)

# --- Feature importance ---
st.subheader("Top Features Influencing Repayment")
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
st.subheader("Tips to Improve Loan Eligibility")
st.markdown("""
Improving your eligibility for agricultural loans is essential for building trust with lenders. Here are some tips:

- **Submit complete documentation**: Including Voter’s Card, BVN, Tax Invoice, and Tax Clearance Certificate.
- **Maintain steady income**: Aim for consistent income above ₦115,000/month.
- **Invest in your farm**: Frequent reinvestment in tools and land boosts credibility.
- **Own agricultural property**: Owning land and mechanized tools shows capacity to scale.
- **Continue learning**: Completing secondary education or beyond helps improve eligibility.
- **Control risks**: Reduce crop loss due to pests and drought by using modern methods.
""")

# --- Footer ---
st.markdown("---")
st.caption("Developed for farmer empowerment and financial inclusion.")
