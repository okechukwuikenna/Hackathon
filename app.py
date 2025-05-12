# --- Eligibility check ---
def is_eligible(row_df):
    row = row_df.iloc[0]
    # Normalize average income string
    income_str = row.get("Avg Income Level", "")
    income_levels = [
        "Below N15,000 per month",
        "N15,001 - N35,000 per month",
        "N35,001 - N55,000 per month",
        "N55,001 - N115,000 per month",
        "N115,001 - N215,000 per month",
        "N215,001 - N315,000 per month",
        "Above N315,000 per month"
    ]
    min_required_index = income_levels.index("N115,001 - N215,000 per month")
    user_index = income_levels.index(income_str) if income_str in income_levels else -1

    return (
        row.get("Age", 0) >= 21 and
        row.get("BVN", "No") == "Yes" and
        row.get("Debt", "Yes") == "No" and
        row.get("Tax Invoice", "No") == "Yes" and
        user_index >= min_required_index
    )

# --- Prediction section with eligibility check ---
st.subheader("🔮 Prediction Result")

if is_eligible(input_df):
    if meets_repayment_rules(input_df):
        prediction = "Yes"
        proba = [0.01, 0.99]
    else:
        prediction = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0]

    confidence = round(max(proba) * 100, 2)
    if prediction == "Yes":
        st.success(f"✅ This farmer is **likely to repay** the loan. (Confidence: {confidence}%)")
    else:
        st.error(f"⚠️ This farmer is **unlikely to repay** the loan. (Confidence: {confidence}%)")

    # Download button
    result_df = input_df.copy()
    result_df["Prediction"] = prediction
    result_df["Confidence (%)"] = confidence
    csv = result_df.to_csv(index=False).encode()
    st.download_button("📥 Download Prediction Result", data=csv, file_name="loan_prediction_result.csv", mime="text/csv")

else:
    st.warning("🚫 The applicant does not meet the **minimum eligibility criteria** to assess loan risk.")
    st.markdown("""
**Required Conditions:**
- Age **21 or older**
- BVN must be **Yes**
- Debt must be **No**
- Tax Invoice must be **Yes**
- Avg Income Level at least **₦115,001/month**
""")
