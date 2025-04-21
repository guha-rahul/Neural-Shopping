import streamlit as st
import joblib
import pandas as pd
import numpy as np
from neural_network import NeuralNetwork

# Load saved components
model = joblib.load('trained_nn_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title("🛍️ Online Shopper Purchase Prediction")

# Input fields
st.header("Enter User Session Data")
input_data = {
    'Administrative': st.number_input("Administrative", min_value=0),
    'Administrative_Duration': st.number_input("Administrative Duration", min_value=0.0),
    'Informational': st.number_input("Informational", min_value=0),
    'Informational_Duration': st.number_input("Informational Duration", min_value=0.0),
    'ProductRelated': st.number_input("Product Related", min_value=0),
    'ProductRelated_Duration': st.number_input("Product Related Duration", min_value=0.0),
    'BounceRates': st.slider("Bounce Rates", 0.0, 1.0, 0.1),
    'ExitRates': st.slider("Exit Rates", 0.0, 1.0, 0.1),
    'PageValues': st.number_input("Page Values", min_value=0.0),
    'SpecialDay': st.slider("Special Day", 0.0, 1.0, 0.0),
    'Month': st.selectbox("Month", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
    'OperatingSystems': st.selectbox("Operating Systems", list(range(1, 9))),
    'Browser': st.selectbox("Browser", list(range(1, 14))),
    'Region': st.selectbox("Region", list(range(1, 10))),
    'TrafficType': st.selectbox("Traffic Type", list(range(1, 21))),
    'VisitorType': st.selectbox("Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other']),
    'Weekend': st.radio("Weekend", [0, 1])
}

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=['Month', 'VisitorType'], drop_first=True)

    # Align input with model features
    missing_cols = set(feature_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[feature_names]

    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)[0]

    st.success(f"Prediction: {'✅ Will Purchase' if prediction else '❌ Will Not Purchase'}")
