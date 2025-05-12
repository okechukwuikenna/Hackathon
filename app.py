import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Streamlit page setup ---
st.set_page_config(page_title="Farmer Loan Repayment Predictor", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_excel("loan_features_tables.xlsx")
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    return df

df = load_data()

# --- Visualization ---
st.markdown("---")
st.subheader("📊 Compare Three Variables Dynamically")

# Let the user choose the type of chart
chart_type = st.selectbox("Select Chart Type", ["Histogram", "Bar", "Line", "Scatter", "Heatmap"])

# Select X and Y variables
var_x = st.selectbox("Select X-axis Variable", df.columns, key="x")
var_y = st.selectbox("Select Y-axis Variable", df.columns, key="y")

# Optional third variable (Color/Facet), no third variable selected by default
var_color_or_facet = st.selectbox("Optional: Select Color/Facet Variable", ["None"] + list(df.columns), key="color_facet")

# Check if user wants to facet or color by the third variable
facet_args = {}
color_args = {}

if var_color_or_facet != "None":
    # Ask the user if they want to use it for coloring or faceting
    use_as_facet = st.radio("Would you like to use the third variable for:", ["Facet", "Color"])

    if use_as_facet == "Facet":
        facet_args = {"facet_col": var_color_or_facet}
    elif use_as_facet == "Color":
        color_args = {"color": var_color_or_facet}

# Select a color scheme for better aesthetics and randomness
color_seq = px.colors.qualitative.Plotly  # Random colors from Plotly's predefined palettes

# Generate the chart based on selected chart type
if chart_type == "Histogram":
    fig = px.histogram(
        df, x=var_x, y=var_y, barmode="group", color_discrete_sequence=color_seq, **facet_args, **color_args
    )

elif chart_type == "Bar":
    grouped_df = df.groupby([var_x, var_y])[var_y].mean().reset_index()
    fig = px.bar(
        grouped_df, x=var_x, y=var_y, color_discrete_sequence=color_seq, barmode="group", **facet_args, **color_args
    )

elif chart_type == "Line":
    fig = px.line(
        df, x=var_x, y=var_y, markers=True, color_discrete_sequence=color_seq, **facet_args, **color_args
    )

elif chart_type == "Scatter":
    fig = px.scatter(
        df, x=var_x, y=var_y, color_discrete_sequence=color_seq, **facet_args, **color_args
    )

elif chart_type == "Heatmap":
    # Creating a heatmap requires numerical data on both x and y axes.
    # Make sure to choose a column for aggregation or use suitable numeric columns
    pivot_table = df.pivot_table(index=var_y, columns=var_x, values=df.columns[-2], aggfunc="count").fillna(0)
    fig = px.imshow(pivot_table, aspect="auto", color_continuous_scale="bluered")

# Update chart layout and display
fig.update_layout(height=600, title=f"{chart_type} of {var_x} vs {var_y}")
st.plotly_chart(fig, use_container_width=True)
