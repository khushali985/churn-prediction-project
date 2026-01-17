# Gender-> 1Female 0Male
# Churn -> 1yes 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# order of the x will be this 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st # build web ui
import joblib # load trained ML objects
import numpy as np # array manipulations

# loading trained scaler and model
scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")
st.divider()
st.write("please enter the values and hit the predict button for getting a prediction.")
st.divider()

age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Enter tenure", min_value=0, max_value=130, value=10)
monthlycharges = st.number_input("Enter monthly charges", min_value=30, max_value=150, value=150)
gender = st.selectbox("enter your gender", ["Male", "Female"])

st.divider()
predict_button = st.button("Predict!")

if predict_button:
    gender_selected = 1 if gender == "Female" else 0
    
    # preparing input array in correct order
    x = [ age, gender_selected, tenure, monthlycharges]
    x1 = np.array(x)
    
    # scaling input
    x_array = scaler.transform([x1])
    
    # getting prediction
    prediction = model.predict(x_array)[0]
    
    # displaying prediction
    predicted = "Churn" if prediction == 1 else "Not Churn"
    st.write(f"The customer is predicted to be: {predicted}")
else:
    st.write("Please enter the values and use predict button.")
    
