import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model, scaler, and features
model = joblib.load("hdLogisticModel.pkl")
scaler = joblib.load("hdScaler.pkl")
features = joblib.load("hdFeatures.pkl")

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a clean, simple design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .input-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
    }
    .high-risk {
        background: linear-gradient(135deg, #FFE6E6, #FFD1D1);
        border: 2px solid #FF4B4B;
    }
    .low-risk {
        background: linear-gradient(135deg, #E6F7E6, #D1F2D1);
        border: 2px solid #4CAF50;
    }
    .feature-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown(
    '<h1 class="main-header">❤️ Heart Disease Risk Checker</h1>', unsafe_allow_html=True
)
st.markdown('<p class="sub-header">Quick assessment based on key health indicators</p>', unsafe_allow_html=True)

# Simple input form
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Basic demographics - most important features based on correlation analysis
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "**Age**",
            min_value=18,
            max_value=100,
            value=50,
            help="Your current age in years",
        )
        sex = st.selectbox(
            "**Sex**", ["Male", "Female"], help="Biological sex (Men have higher risk)"
        )

    with col2:
        chest_pain = st.selectbox(
            "**Chest Pain Experience**",
            [
                "No chest pain/Asymptomatic",
                "Typical chest pain during exercise",
                "Atypical chest pain",
                "Non-heart related chest pain",
            ],
            help="Type of chest discomfort you experience",
        )

        max_hr = st.number_input(
            "**Maximum Heart Rate** (if known)",
            min_value=60,
            max_value=220,
            value=150,
            help="Highest heart rate during exercise. If unsure, use 220 minus your age",
        )

    # Additional risk factors
    st.markdown("**Additional Health Information:**")
    col3, col4 = st.columns(2)

    with col3:
        exercise_angina = st.selectbox(
            "**Exercise causes chest discomfort?**",
            ["No", "Yes"],
            help="Do you experience chest pain during physical activity?",
        )

        bp_category = st.selectbox(
            "**Blood Pressure Category**",
            [
                "Normal (< 120/80)",
                "Elevated (120-129/80)",
                "High (130+/80+)",
                "Don't know",
            ],
            help="Your general blood pressure category",
        )

    with col4:
        cholesterol_level = st.selectbox(
            "**Cholesterol Level**",
            ["Normal (< 200)", "Borderline (200-239)", "High (240+)", "Don't know"],
            help="Your cholesterol level if known from recent tests",
        )

        diabetes_risk = st.selectbox(
            "**Diabetes/High Blood Sugar**",
            ["No", "Yes", "Pre-diabetic"],
            help="Do you have diabetes or high fasting blood sugar?",
        )

    st.markdown("</div>", unsafe_allow_html=True)

# Prediction button
predict_button = st.button(
    "🔍 Check Heart Disease Risk", use_container_width=True, type="primary"
)

if predict_button:
    # Convert simple inputs to model features
    # Map chest pain types
    chest_pain_mapping = {
        "No chest pain/Asymptomatic": "ASY",
        "Typical chest pain during exercise": "TA",
        "Atypical chest pain": "ATA",
        "Non-heart related chest pain": "NAP",
    }
    chest_pain_type = chest_pain_mapping[chest_pain]

    # Estimate blood pressure (use typical values for categories)
    bp_mapping = {
        "Normal (< 120/80)": 110,
        "Elevated (120-129/80)": 125,
        "High (130+/80+)": 145,
        "Don't know": 120,  # Use average
    }
    resting_bp = bp_mapping[bp_category]

    # Estimate cholesterol
    chol_mapping = {
        "Normal (< 200)": 180,
        "Borderline (200-239)": 220,
        "High (240+)": 260,
        "Don't know": 200,  # Use average
    }
    cholesterol = chol_mapping[cholesterol_level]

    # Convert diabetes risk to fasting BS
    fasting_bs = 1 if diabetes_risk in ["Yes", "Pre-diabetic"] else 0

    # Set default values for less critical features based on typical values
    resting_ecg = "Normal"  # Most common
    st_slope = "Flat"  # Most common in dataset
    oldpeak = 1.0  # Average value

    # Prepare input data matching the exact features from the model
    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0,
        "ChestPainType_ATA": 1 if chest_pain_type == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain_type == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain_type == "TA" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0,
    }

    # Create DataFrame with the exact feature order from training
    input_df = pd.DataFrame([input_data])

    # Ensure all features are present and in correct order
    input_df = input_df.reindex(columns=features, fill_value=0)    # Apply the same scaling as in training
    input_df_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_df_scaled)[0]
    prediction_proba = model.predict_proba(input_df_scaled)[0]

    # Debug information (you can remove this later)
    with st.expander("🔧 Debug Info (Model Input)"):
        st.write("**Input to model:**")
        st.dataframe(input_df)
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Probabilities:** No Disease: {prediction_proba[0]:.3f}, Disease: {prediction_proba[1]:.3f}")

    # Display results with simplified messaging
    if prediction == 1:
        confidence = prediction_proba[1] * 100
        st.markdown(
            f"""
        <div class="prediction-box high-risk">
            <h2>⚠️ Higher Risk Detected</h2>
            <h3>Risk Level: {confidence:.0f}%</h3>
            <p><strong>Recommendation:</strong> Consider consulting with a healthcare provider for a comprehensive cardiac evaluation.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )        # Show risk factors for high risk cases
        risk_factors = []
        if age > 55:
            risk_factors.append(f"Age ({age} years)")
        if sex == "Male":
            risk_factors.append("Male gender")
        if bp_category == "High (130+/80+)":
            risk_factors.append("High blood pressure")
        if cholesterol_level == "High (240+)":
            risk_factors.append("High cholesterol")
        if diabetes_risk in ["Yes", "Pre-diabetic"]:
            risk_factors.append("Diabetes/High blood sugar")
        if exercise_angina == "Yes":
            risk_factors.append("Exercise-induced chest pain")
        if chest_pain_type in ["TA", "ATA"]:
            risk_factors.append("Chest pain symptoms")

        if risk_factors:
            st.warning("**Contributing factors:** " + ", ".join(risk_factors))

    else:
        confidence = prediction_proba[0] * 100
        st.markdown(
            f"""
        <div class="prediction-box low-risk">
            <h2>✅ Lower Risk Profile</h2>
            <h3>Risk Level: {confidence:.0f}% likely no disease</h3>
            <p><strong>Recommendation:</strong> Continue maintaining a healthy lifestyle. Regular check-ups are still important for preventive care.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Information section
st.markdown("---")
with st.expander("ℹ️ About This Assessment"):
    st.markdown(
        """
    **How it works:**
    - Uses machine learning trained on 918 patient records
    - Focuses on the most predictive health indicators
    - Provides estimates when specific values aren't known
    
    **Key factors analyzed:**
    - Age and biological sex
    - Chest pain patterns
    - Exercise capacity and symptoms  
    - Blood pressure and cholesterol levels
    - Diabetes status
    
    **Important:** This is a screening tool only and should never replace professional medical advice. 
    Always consult healthcare providers for proper diagnosis and treatment.
    """
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.9rem;'>Heart Disease Risk Assessment | For Educational Purposes</div>",
    unsafe_allow_html=True,
)
