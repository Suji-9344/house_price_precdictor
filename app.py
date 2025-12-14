import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("trained_house_pricing_LR_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("House Price Prediction App")
st.write("Predict house price based on input features")

# Input fields for features
# (Adjust feature names and number according to your model)
# Example assumes 3 features: "Size (sqft)", "Bedrooms", "Age of House"
size = st.number_input("Size (sqft)", min_value=100, max_value=10000, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
age = st.number_input("Age of House (years)", min_value=0, max_value=100, value=10)

# Create input array
input_features = np.array([[size, bedrooms, age]])

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict(input_features)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
