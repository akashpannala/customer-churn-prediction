import streamlit as st
import joblib
import pandas as pd
import os

#PATHS
MODEL_DIR = 'models'

#LOAD
dt = joblib.load(os.path.join(MODEL_DIR, 'model_dt.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
le_geo = joblib.load(os.path.join(MODEL_DIR, 'le_geo.pkl'))
le_gen = joblib.load(os.path.join(MODEL_DIR, 'le_gen.pkl'))
all_features = joblib.load(os.path.join(MODEL_DIR, 'feature_order.pkl'))  # ← EXACT ORDER

numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

st.title("Bank Customer Churn Predictor")
st.markdown("**Enter details → Instant churn prediction**")

#FORM
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        geography = st.selectbox("Geography", le_geo.classes_)
        gender = st.selectbox("Gender", le_gen.classes_)
        age = st.number_input("Age", 18, 100, 35)
        tenure = st.number_input("Tenure (years)", 0, 10, 5)

    with col2:
        balance = st.number_input("Balance (€)", 0.0, 300000.0, 0.0, step=1000.0)
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        salary = st.number_input("Estimated Salary (€)", 0.0, 200000.0, 50000.0, step=1000.0)

    submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        #ENCODE
        geo_enc = le_geo.transform([geography])[0]
        gen_enc = le_gen.transform([gender])[0]

        #BUILD INPUT IN EXACT ORDER 
        input_values = [
            geo_enc,           # Geography
            gen_enc,           # Gender
            credit_score,      # CreditScore
            age,               # Age
            tenure,            # Tenure
            balance,           # Balance
            num_products,      # NumOfProducts
            has_cr_card,       # HasCrCard
            is_active,         # IsActiveMember
            salary             # EstimatedSalary
        ]

        #Create DataFrame with EXACT column names and order
        input_df = pd.DataFrame([input_values], columns=all_features)

        #SCALE
        input_scaled = input_df.copy()
        input_scaled[numerical_features] = scaler.transform(input_scaled[numerical_features])

        #PREDICT
        prob = dt.predict_proba(input_scaled)[0][1]
        pred = dt.predict(input_scaled)[0]

        #RESULTS
        st.markdown("---")
        if pred == 1:
            st.error("**HIGH CHURN RISK**")
            st.warning(f"**Churn Probability: {prob:.2%}**")
        else:
            st.success("**LOW CHURN RISK**")
            st.info(f"**Churn Probability: {prob:.2%}**")

        # INPUT VIEW
        with st.expander("View Input Data"):
            st.write("**Raw Input:**", input_df)
            st.write("**Scaled Input:**", input_scaled)
            st.write("**Feature Order Used:**", all_features)