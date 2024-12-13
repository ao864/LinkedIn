import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib  # For loading the pre-trained model
import plotly.graph_objects as go

# Load the pre-trained model (assuming you've saved your model)
model = joblib.load('log1.pkl') 

# Title of the Streamlit app
st.title('LinkedIn Usage Prediction')

# Collect user inputs
st.header('Enter the following details:')

# Define a function to map the income input to its corresponding numerical value
def map_income_to_value(income_category):
    if income_category == "Less than $10,000":
        return 1
    elif income_category == "10 to under $20,000":
        return 2
    elif income_category == "20 to under $30,000":
        return 3
    elif income_category == "30 to under $40,000":
        return 4
    elif income_category == "40 to under $50,000":
        return 5
    elif income_category == "50 to under $75,000":
        return 6
    elif income_category == "75 to under $100,000":
        return 7
    elif income_category == "100 to under $150,000":
        return 8
    elif income_category == "$150,000 or more":
        return 9
    else:
        return None  # If no valid category is selected

# Define a function to map the education level input to its corresponding numerical value
def map_education_to_value(education_level):
    if education_level == "Less than high school (Grades 1-8 or no formal schooling)":
        return 1
    elif education_level == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
        return 2
    elif education_level == "High school graduate (Grade 12 with diploma or GED certificate)":
        return 3
    elif education_level == "Some college, no degree (includes some community college)":
        return 4
    elif education_level == "Two-year associate degree from a college or university":
        return 5
    elif education_level == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
        return 6
    elif education_level == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
        return 7
    elif education_level == "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)":
        return 8
    else:
        return None  # If no valid category is selected

# Create input fields for the model features
income_category = st.selectbox("Select your income category", [
    "Less than $10,000",
    "10 to under $20,000",
    "20 to under $30,000",
    "30 to under $40,000",
    "40 to under $50,000",
    "50 to under $75,000",
    "75 to under $100,000",
    "100 to under $150,000",
    "$150,000 or more"
])

# Call the function to map the selected category to a value
income = map_income_to_value(income_category)

education_category = st.selectbox("Select your education category", [
    "Less than high school (Grades 1-8 or no formal schooling)",
    "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
    "High school graduate (Grade 12 with diploma or GED certificate)",
    "Some college, no degree (includes some community college)",
    "Two-year associate degree from a college or university",
    "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
    "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
    "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"
])

# Call the function to map the selected category to a value
education = map_education_to_value(education_category)

parent = st.selectbox('Are you a parent?', ('No', 'Yes'))
marital_status = st.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
gender = st.selectbox('Gender', ('Male', 'Female', '2 or More Genders', 'Genderfluid', 'Unsure', ''))
age = st.number_input('Age (e.g., 42)', min_value=18, max_value=100, step=1)

# Convert the inputs into the correct format for the model
parent = 1 if parent == 'Yes' else 0
marital_status = 1 if marital_status == 'Married' else 0
gender = 1 if gender == 'Female' else 0

# Convert the inputs into the correct format for the model
parent = 1 if parent == 'Yes' else 0
marital_status = 1 if marital_status == 'Married' else 0
gender = 1 if gender == 'Female' else 0

# Convert the inputs into the correct format for the model
parent = 1 if parent == 'Yes' else 0
marital_status = 1 if marital_status == 'Married' else 0
gender = 1 if gender == 'Female' else 0

# Feature vector to feed into the model
features = np.array([[income, education, parent, marital_status, gender, age]])

# Predict the probability and class
probability = model.predict_proba(features)[:, 1]
prediction = model.predict(features)

def predictor(inputs):
    prediction = model.predict(inputs)
    probability = model.predict_proba(inputs)[0][1] 

    if st.button('Predict'):
        st.write(f'Probability of LinkedIn usage: {probability * 100}%')
    if prediction[0] == 1:
        st.write('The person is classified as a LinkedIn user.')
    else:
        st.write('The person is classified as not a LinkedIn user.')

    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value= probability,
    title={'text': f"LinkedIn User Prediction: {prediction}"},
    gauge={
        "axis": {"range": [0, 100]},  
        "steps": [
            {"range": [0, 33], "color": "red"},
            {"range": [34, 66], "color": "gray"},
            {"range": [67, 100], "color": "lightgreen"}
        ],
        "bar": {"color": "yellow"}
    }
))
    
    return probability, prediction[0], fig

predictor(features)

