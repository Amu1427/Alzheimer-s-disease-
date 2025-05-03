import streamlit as st
import pandas as pd
import numpy as np
import pickle


with open('alzheimer_model.pkl', 'rb') as f:
    model = pickle.load(f)


st.title("🧠 Alzheimer's Disease Prediction")


feature_names = model.named_steps['preprocessor'].get_feature_names_out()


with st.form("prediction_form"):
    st.header("Patient Details")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 50, 100, 65)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.slider("Education Level (years)", 0, 20, 12)
        bmi = st.slider("BMI", 15.0, 40.0, 25.0)
        
    with col2:
        activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
        diabetes = st.radio("Diabetes", ["No", "Yes"])
    

    hypertension = st.radio("Hypertension", ["No", "Yes"])
    cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High"])
    family_history = st.radio("Family History of Alzheimer's", ["No", "Yes"])
    cognitive_score = st.slider("Cognitive Test Score", 30, 100, 75)
    depression = st.selectbox("Depression Level", ["Low", "Medium", "High"])
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        
        input_dict = {
            'Country': 'USA',  
            'Age': age,
            'Gender': gender,
            'Education Level': education,
            'BMI': bmi,
            'Physical Activity Level': activity,
            'Smoking Status': smoking,
            'Alcohol Consumption': alcohol,
            'Diabetes': diabetes,
            'Hypertension': hypertension,
            'Cholesterol Level': cholesterol,
            'Family History of Alzheimer’s': family_history,
            'Cognitive Test Score': cognitive_score,
            'Depression Level': depression,
            'Sleep Quality': 'Good',  
            'Dietary Habits': 'Healthy',
            'Air Pollution Exposure': 'Medium',
            'Employment Status': 'Retired',
            'Marital Status': 'Married',
            'Genetic Risk Factor (APOE-ε4 allele)': 'No',
            'Social Engagement Level': 'Medium',
            'Income Level': 'Medium',
            'Stress Levels': 'Medium',
            'Urban vs Rural Living': 'Urban'
        }
        
        
        input_df = pd.DataFrame([input_dict], columns=model.feature_names_in_)
        
        try:
            
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            
            
            st.success(f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
            st.metric("Probability of Alzheimer's", f"{proba[1]*100:.1f}%")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Input data used:")
            st.write(input_df)