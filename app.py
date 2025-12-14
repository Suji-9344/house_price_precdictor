import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🏠 House Price Predictor")

st.write("Enter the details of the house to predict its price:")

# User input fields
area = st.number_input('Area (in sq ft)', min_value=100, max_value=10000, value=1000)
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)

# When the user clicks Predict
if st.button('Predict'):
    # Prepare input as 2D array
    input_features = np.array([[area, bedrooms, bathrooms]])
    
    try:
        # Make prediction
        prediction = model.predict(input_features)
        st.success(f"Predicted House Price: ₹{prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"Error: {e}")

