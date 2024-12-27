import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
with open(r"C:\\Users\\Ranjan kumar pradhan\\.vscode\\Diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)
scaler = StandardScaler()


# Define the Streamlit app
st.title("Diabetes Prediction App")

# Input fields for user data
st.header("Enter the following details:")
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0, step=1)

# Prediction button
if st.button("Predict Diabetes"):
    # Collect input data into a DataFrame
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree_function],
        "Age": [age]
    })

    # Scale the input data
    input_data_scaled = scaler.fit_transform(input_data)

    # Predict using the loaded model
    prediction = model.predict(input_data_scaled)

    # Display the result
    if prediction[0] == 1:
        st.success("The model predicts that the person has diabetes.")
    else:
        st.success("The model predicts that the person does not have diabetes.")
