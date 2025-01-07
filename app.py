import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('notebooks/logistic_regression_model.pkl')

# Streamlit user interface
st.title('Thyroid Cancer Condition Prediction')

# User inputs
age = st.number_input('Age', min_value=1, max_value=100, value=25)
sex = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
on_thyroxine = st.selectbox('Are you on Thyroxine?', options=[0, 1])
query_on_thyroxine = st.selectbox('Is there a query about Thyroxine?', options=[0, 1])
on_antithyroid_medication = st.selectbox('Are you on Antithyroid medication?', options=[0, 1])
sick = st.selectbox('Are you sick?', options=[0, 1])
pregnant = st.selectbox('Are you pregnant?', options=[0, 1])
thyroid_surgery = st.selectbox('Have you had thyroid surgery?', options=[0, 1])
I131_treatment = st.selectbox('Have you received I131 treatment?', options=[0, 1])
query_hypothyroid = st.selectbox('Is there a query about Hypothyroidism?', options=[0, 1])

# Define the required columns
required_columns = ['age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 
                    'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid']

# Input data
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'on thyroxine': [on_thyroxine],
    'query on thyroxine': [query_on_thyroxine],
    'on antithyroid medication': [on_antithyroid_medication],
    'sick': [sick],
    'pregnant': [pregnant],
    'thyroid surgery': [thyroid_surgery],
    'I131 treatment': [I131_treatment],
    'query hypothyroid': [query_hypothyroid],
})

# Filter the columns to match the model's required input
input_data = input_data[required_columns]

# Convert all data to numeric
input_data = input_data.apply(pd.to_numeric, errors='coerce')

# Prediction using the model
if st.button('Predict'):
    prediction = model.predict(input_data)

    # Display the result
    if prediction == 1:
        st.write("Prediction: You have thyroid cancer")
    else:
        st.write("Prediction: You are  thyroid cancer")
