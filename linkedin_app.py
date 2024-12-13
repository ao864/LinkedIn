import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib  # For loading the pre-trained model

# Load the pre-trained model (assuming you've saved your model)
model = joblib.load('log1.pkl')  # Replace 'model.pkl' with the actual filename

# Title of the Streamlit app
st.title('LinkedIn Usage Prediction')

# Collect user inputs
st.header('Enter the following details:')

# Create input fields for the model features
income = st.number_input('Income (e.g., 8)', min_value=1, max_value=9, step=1)
education = st.number_input('Education Level (e.g., 7)', min_value=1, max_value=9, step=1)
parent = st.selectbox('Are you a parent?', ('No', 'Yes'))
marital_status = st.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
gender = st.selectbox('Gender', ('Male', 'Female', '2 or More Genders', 'Genderfluid', 'Unsure', ''))
age = st.number_input('Age (e.g., 42)', min_value=18, max_value=100, step=1)

# Convert the inputs into the correct format for the model
parent = 1 if parent == 'Yes' else 0
marital_status = 1 if marital_status == 'Married' else 0
gender = 1 if gender == 'Female' else 0

# Feature vector to feed into the model
features = np.array([[income, education, parent, marital_status, gender, age]])

# Predict the probability and class
probability = model.predict_proba(features)[:, 1]
prediction = model.predict(features)

# Display the results
if st.button('Predict'):
    st.write(f'Probability of LinkedIn usage: {round(probability[0] * 100,2)}%')
    if prediction[0] == 1:
        st.write('The person is classified as a LinkedIn user.')
    else:
        st.write('The person is classified as not a LinkedIn user.')
