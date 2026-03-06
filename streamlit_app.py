import streamlit as st
import requests

Api_url = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Insurance Premium Predictor")

st.title("💰 Insurance Premium Prediction")

# ----------------------------
# User Inputs
# ----------------------------

age = st.text_input("Age", placeholder="Enter Age")
sex = st.selectbox("Sex", ["Select Sex", "male", "female"])
bmi = st.text_input("BMI", placeholder="Enter BMI")
children = st.text_input("Number of Children", placeholder="Enter Children")
smoker = st.selectbox("Are you Smoker", ["Select Option", "yes", "no"])
region = st.selectbox("Region", ["Select Region", "northeast", "northwest", "southeast", "southwest"])

# ----------------------------
# Predict Button
# ----------------------------

if st.button("Predict Premium"):

    input_data = {
       "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region
    }

    try:
        response = requests.post(Api_url, json=input_data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Premium: ₹ {result['predicted_premium']:.2f}")
        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(e)