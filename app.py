import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load models and artifacts
stacking_model = joblib.load("best_model.pkl")
feature_names = joblib.load("feature_names.pkl")
scaler = joblib.load("scaler.pkl")
perm_importances = joblib.load("permutation_importance.pkl")  # pandas.Series

st.set_page_config(page_title="Heart Disease Risk", layout="centered")
st.title("💓 Heart Disease Risk Predictor")
st.markdown("Enter your medical data to estimate heart disease risk.")

# Form input
user_input = {}
with st.form("heart_form"):
    with st.expander("🧾 Patient Health Details", expanded=True):
        user_input["oldpeak"] = st.number_input("Oldpeak (ST Depression)", min_value=0.0, step=0.1,
                                                help="ST depression induced by exercise relative to rest.")
        user_input["sex"] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
                                         help="Biological sex (0 = Female, 1 = Male).")
        user_input["exercise angina"] = st.selectbox("Exercise Induced Angina", [0, 1],
                                                     format_func=lambda x: "No" if x == 0 else "Yes",
                                                     help="Chest pain triggered by physical activity.")
        user_input["fasting blood sugar"] = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                                                         format_func=lambda x: "No" if x == 0 else "Yes",
                                                         help="Blood sugar after fasting > 120 mg/dL.")
        user_input["chest pain type"] = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                                                     format_func=lambda x: {
                                                         1: "Typical angina", 2: "Atypical angina",
                                                         3: "Non-anginal pain", 4: "Asymptomatic"}[x],
                                                     help="Type of chest pain experienced.")
        user_input["cholesterol"] = st.slider("Cholesterol (mg/dL)", 100, 600, 200,
                                              help="Total serum cholesterol in mg/dL.")
        user_input["max heart rate"] = st.slider("Max Heart Rate", 60, 220, 150,
                                                 help="Maximum heart rate achieved during test.")
        user_input["resting bp s"] = st.slider("Resting BP (mmHg)", 80, 200, 120,
                                               help="Resting systolic blood pressure.")
        user_input["age"] = st.slider("Age", 20, 100, 50, help="Patient's age in years.")
        user_input["resting ecg"] = st.selectbox("Resting ECG", [0, 1, 2],
                                                 format_func=lambda x: {
                                                     0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x],
                                                 help="Electrocardiographic results at rest.")
        user_input["ST slope"] = st.selectbox("ST Slope", [1, 2, 3],
                                              format_func=lambda x: {
                                                  1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
                                              help="Slope of peak exercise ST segment.")

    submitted = st.form_submit_button("🧠 Predict Risk")

if submitted:
    input_data = [user_input[feat] for feat in feature_names]

    if len(input_data) != len(feature_names):
        st.error("Feature count mismatch.")
        st.stop()

    input_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = stacking_model.predict(input_scaled)[0]
    proba = stacking_model.predict_proba(input_scaled)[0][1]

    # Show prediction result
    st.header("📊 Prediction Result")
    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
        st.markdown(f"**Heart Disease Probability:** {proba:.4f}")
        st.markdown("🔁 *Please consult a cardiologist for further evaluation.*")
    else:
        st.success("✅ No Heart Disease Detected")
        st.markdown(f"**Heart Disease Probability:** {proba:.4f}")
        st.markdown("🎉 *Keep up your heart-healthy habits!*")

    # Show top feature tip
    top_feature = perm_importances.sort_values(ascending=False).index[0]
    tip_map = {
        "oldpeak": "Reducing ST depression during exercise may improve cardiovascular outcomes.",
        "sex": "While sex is non-modifiable, knowing your risk helps guide prevention.",
        "fasting blood sugar": "Maintaining healthy blood sugar can reduce heart risks.",
        "exercise angina": "Reducing exertion-related chest pain is crucial — follow up is advised.",
        "cholesterol": "Managing cholesterol is a key step in reducing heart disease risk.",
        "max heart rate": "Improving cardiovascular fitness can positively impact max heart rate.",
        "resting bp s": "Controlling high blood pressure helps prevent heart complications.",
        "age": "Healthy lifestyle choices at any age delay heart disease onset.",
        "resting ecg": "Abnormal ECGs should be discussed with a cardiologist.",
        "ST slope": "A downsloping ST segment during exercise can indicate heart stress."
    }
    st.info(f"💡 **Health Tip:** {tip_map.get(top_feature, 'Consult a doctor for personalized advice.')}")

    # Show permutation importance bar chart
    st.subheader("📌 Feature Impact (Permutation Importance)")
    plt.figure()
    perm_importances.sort_values(ascending=True).tail(10).plot(kind='barh', color='steelblue')
    plt.xlabel("Mean Decrease in F1 Score")
    plt.title("Top 10 Important Features (RandomForestClassifier)")
    st.pyplot(plt.gcf())
