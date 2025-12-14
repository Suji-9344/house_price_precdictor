import streamlit as st
import pickle
import numpy as np

# Load the trained model
try:
    with open('house_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found! Please make sure 'house_price_model.pkl' is in the same folder as app.py")
    st.stop()  # Stop the app if model is missing
