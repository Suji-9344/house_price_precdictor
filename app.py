import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------
# Create Dataset (inside code)
# ---------------------------
data = {
    "Area": [800, 1000, 1200, 1500, 1800, 2000, 2300, 2600],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 5, 5],
    "Bathrooms": [1, 2, 2, 3, 3, 4, 4, 5],
    "Price": [3000000, 4000000, 5000000, 6500000, 8000000, 9000000, 11000000, 13000000]
}

df = pd.DataFrame(data)

# ---------------------------
# Train Model
# ---------------------------
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🏠 House Price Predictor")
st.write("This app trains the model automatically and predicts house price.")

area = st.number_input("Area (sq ft)", min_value=500, max_value=5000, value=1200)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted House Price: ₹{prediction[0]:,.2f}")

