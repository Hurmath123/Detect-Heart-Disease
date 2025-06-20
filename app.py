import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set Streamlit page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# ✅ Set background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("heart.webp")  # Make sure heart.webp is in the same folder

# Load saved models and data
model = joblib.load("best_model.pkl")
feature_names = joblib.load("feature_names.pkl")
scaler = joblib.load("scaler.pkl")
perm_importances = joblib.load("permutation_importance.pkl")

# Title
st.title("💓 Heart Disease Risk Estimator")

# Sidebar with evaluation
st.sidebar.subheader("📈 Model Metrics")
st.sidebar.write("Model: **Random Forest (Tuned)**")
st.sidebar.write("Top 3 Features:")
top_3 = perm_importances.sort_values(ascending=False).head(3)
for feat, score in top_3.items():
    st.sidebar.write(f"• {feat}: {score:.4f}")
# Optional: To style the sidebar, use custom CSS (Streamlit does not support sidebar color directly)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #c74a69;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input form with tooltips
user_input = {}
with st.form("heart_form"):
    user_input["age"] = st.slider("Age", 20, 100, 50, help="Enter patient's age in years.")
    user_input["sex"] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="0 = Female, 1 = Male.")
    user_input["chest pain type"] = st.selectbox("Chest Pain Type", [1, 2, 3, 4], format_func=lambda x: {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}[x], help="Type of chest pain experienced during activity or rest.")
    user_input["resting bp s"] = st.slider("Resting BP (mmHg)", 80, 200, 120, help="Resting systolic blood pressure (in mmHg).")
    user_input["cholesterol"] = st.slider("Cholesterol (mg/dL)", 100, 600, 200, help="Serum cholesterol level in mg/dL.")
    user_input["fasting blood sugar"] = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Is fasting blood sugar > 120 mg/dL? (1 = Yes, 0 = No)")
    user_input["resting ecg"] = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}[x], help="Result of resting electrocardiographic test.")
    user_input["max heart rate"] = st.slider("Max Heart Rate Achieved", 60, 220, 150, help="Maximum heart rate achieved during exercise.")
    user_input["exercise angina"] = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Was angina induced by exercise? (1 = Yes, 0 = No)")
    user_input["oldpeak"] = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, step=0.1, help="ST depression induced by exercise relative to rest.")
    user_input["ST slope"] = st.selectbox("ST Slope", [1, 2, 3], format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x], help="Slope of the ST segment during peak exercise.")

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    try:
        input_values = [user_input[feat] for feat in feature_names]
        X_input = scaler.transform(np.array(input_values).reshape(1, -1))
                prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        if prediction == 1:
            st.error("⚠️ Heart Disease Detected")
        st.markdown(f"**Probability of Heart Disease:** `{probabilities[1] * 100:.2f}%`")
        else:
            st.success("✅ No Heart Disease Detected")
        st.markdown(f"**Probability of No Heart Disease:** `{probabilities[0] * 100:.2f}%`")

        # Show most influential feature
        top_feat = perm_importances.sort_values(ascending=False).index[0]
        st.info(f"💡 Most Influential Feature: **{top_feat}**")

        # Permutation importance plot
        st.subheader("🔍 Feature Importance (Top 10)")
        top10 = perm_importances.sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots()
        top10.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title("Top 10 Important Features")
        ax.set_xlabel("Mean Decrease in Score")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction error: {e}")
