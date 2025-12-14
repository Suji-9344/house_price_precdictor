import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('trained_decision_tree_pickle.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Decision Tree Prediction App")
st.write("Enter the input values to get the prediction.")

# Example: assuming your model expects 3 features (change according to your model)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

# Prediction
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)
    st.success(f"The predicted value is: {prediction[0]}")
