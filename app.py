import streamlit as st
import joblib
import numpy as np
import pandas as pd  # Import pandas for handling data

# Load your saved model (update path if needed)
model = joblib.load('Datasetsmodel2.joblib')

st.title('NObeyesdad Prediction App')
st.write("This app predicts obesity levels (NObeyesdad) based on several lifestyle features.")

# ----------------------------
# Input widgets for all features
# ----------------------------

FCVC = st.slider('Frequency of Vegetable Consumption (FCVC)', 1, 3, 2)
FAF = st.slider('Physical Activity Frequency (hours per week)', 0, 7, 2)
CALC = st.selectbox('Alcohol Consumption (CALC)', options=[0, 1, 2, 3], format_func=lambda x: ['No', 'Sometimes', 'Frequently', 'Always'][x])
MTRANS = st.selectbox('Transportation Used (MTRANS)', options=[0, 1, 2, 3, 4], format_func=lambda x: ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'][x])

# ----------------------------
# Define class labels
# ----------------------------
NObeyesdad_classes = [
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Insufficient_Weight',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

# ----------------------------
# Predict button
# ----------------------------
if st.button('Predict'):
    # Create DataFrame with all features in correct order
    input_data = pd.DataFrame([[ FCVC, FAF,  CALC, MTRANS]],
                              columns=[ 'FCVC', 'FAF',  'CALC', 'MTRANS'])

    # Convert to numpy array
    input_features = input_data.to_numpy()

    # Get prediction
    prediction = model.predict(input_features)

    # Map prediction to label
    predicted_class = NObeyesdad_classes[prediction[0]]

    st.success(f'Predicted NObeyesdad category: **{predicted_class}**')
