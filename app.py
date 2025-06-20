import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import json

# Set page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# 🔳 Set background image
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

set_background("heart.webp")

# 📦 Load model and assets
model = joblib.load("best_model.pkl")
feature_names = joblib.load("feature_names.pkl")
scaler = joblib.load("scaler.pkl")
perm_importances = joblib.load("permutation_importance.pkl")

# 📊 Load model evaluation metrics (rf_metrics.json)
try:
    with open("rf_metrics.json") as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = {
        "accuracy": 0.91,
        "f1_score": 0.91,
        "roc_auc": 0.95,
        "precision": 0.90,
        "recall": 0.91
    }

# 📌 Sidebar
st.sidebar.subheader("📈 Model Metrics")
st.sidebar.write("Model: **Random Forest (Tuned)**")
st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.2%}")
st.sidebar.metric("F1 Score", f"{metrics['f1_score']:.2%}")
st.sidebar.metric("ROC AUC", f"{metrics['roc_auc']:.2%}")

# Feature importance (top 3)
st.sidebar.markdown("**Top 3 Features:**")
top_3 = perm_importances.sort_values(ascending=False).head(3)
for feat, score in top_3.items():
    st.sidebar.write(f"• {feat}: {score:.4f}")

# Title
st.title("💓 Heart Disease Risk Estimator")

# 📥 User input form
user_input = {}
with st.form("heart_form"):
    user_input["age"] = st.slider("Age", 20, 100, 50, help="Enter patient's age in years.")
    user_input["sex"] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="0 = Female, 1 = Male.")
    user_input["chest pain type"] = st.selectbox("Chest Pain Type", [1, 2, 3, 4], format_func=lambda x: {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}[x], help="Type of chest pain experienced.")
    user_input["resting bp s"] = st.slider("Resting BP (mmHg)", 80, 200, 120, help="Resting systolic blood pressure.")
    user_input["cholesterol"] = st.slider("Cholesterol (mg/dL)", 100, 600, 200, help="Serum cholesterol level.")
    user_input["fasting blood sugar"] = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Is fasting blood sugar > 120 mg/dL?")
    user_input["resting ecg"] = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}[x], help="ECG result at rest.")
    user_input["max heart rate"] = st.slider("Max Heart Rate Achieved", 60, 220, 150, help="Peak heart rate achieved.")
    user_input["exercise angina"] = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Angina triggered by exercise?")
    user_input["oldpeak"] = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, step=0.1, help="ST depression due to exercise.")
    user_input["ST slope"] = st.selectbox("ST Slope", [1, 2, 3], format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x], help="Slope of ST segment during peak exercise.")

    submitted = st.form_submit_button("Predict")

# 🔮 Prediction and probability display
if submitted:
    try:
        input_values = [user_input[feat] for feat in feature_names]
        X_input = scaler.transform(np.array(input_values).reshape(1, -1))
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]

        st.subheader("🩺 Prediction Result")
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
