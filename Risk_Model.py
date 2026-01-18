import streamlit as st
import pandas as pd
import joblib

st.title("Health Risk Prediction App")

# ✅ Load ONLY the .pkl file
model = joblib.load("Risk_model1.pk1")

age = st.number_input("Age", min_value=0)
length_of_stay = st.number_input("Length Of Stay (Days)", min_value=0)
treatment_cost = st.number_input("Treatment Cost", min_value=0.0)
abnormal_lab_count = st.number_input("Abnormal Lab Count", min_value=0)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[age, length_of_stay, treatment_cost, abnormal_lab_count]],
        columns=["Age", "LengthOfStay", "TreatmentCost", "AbnormalLabCount"]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.success("High Risk" if prediction == 1 else "Low Risk")
    st.info(f"Risk Probability: {probability:.2f}")
