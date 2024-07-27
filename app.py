import streamlit as st
import joblib
import numpy as np
import sklearn

print("Scikit-learn version:", sklearn.__version__)

# Load the trained model (make sure the path to your model is correct)
model = joblib.load('model_compressed_high.pkl')

# Title of the app
st.title("Credit Score Prediction App")

# Function to get user inputs
def get_user_input():
    col1, col2 = st.columns(2)

    with col1:
        Annual_Income = st.number_input('Annual Income', min_value=0.0, step=0.1)
        Delay_from_due_date = st.number_input('Delay from due date', min_value=0.0, step=0.1)
        Credit_Mix = st.selectbox('Credit Mix', ['Bad', 'Standard', 'Good'])
        Credit_Utilization_Ratio = st.number_input('Credit Utilization Ratio', min_value=0.0, step=0.1)
        Total_EMI_per_month = st.number_input('Total EMI per month', min_value=0.0, step=0.1)

       
    with col2:
        Interest_Rate = st.number_input('Interest Rate', min_value=0.0, step=0.1)
        Changed_Credit_Limit = st.number_input('Changed Credit Limit', min_value=0.0, step=0.1)
        Outstanding_Debt = st.number_input('Outstanding Debt', min_value=0.0, step=0.1)
        Credit_History_Age = st.number_input('Credit History Age', min_value=0.0, step=0.1)
        Monthly_Balance = st.number_input('Monthly Balance', min_value=0.0, step=0.1)
        
    # Encode the Credit_Mix feature
    credit_mix_mapping = {'Bad': 0, 'Standard': 1, 'Good': 2}
    Credit_Mix_encoded = credit_mix_mapping[Credit_Mix]

    # Create an input array for the model
    input_data = np.array([Outstanding_Debt, Credit_Mix_encoded, Credit_History_Age, Interest_Rate, 
                           Delay_from_due_date, Changed_Credit_Limit, Monthly_Balance, 
                           Credit_Utilization_Ratio, Annual_Income, Total_EMI_per_month]).reshape(1, -1)
    
    return input_data

# Get user input
user_input = get_user_input()

# Button to make predictions
if st.button("Predict Credit Score"):
    # Make prediction
    prediction = model.predict(user_input)
    
    # Display the result
    st.subheader(f'The predicted credit score is: {prediction[0]}')
