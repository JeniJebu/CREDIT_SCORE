import streamlit as st
import joblib
import numpy as np
import sklearn


# Load the trained model (make sure the path to your model is correct)
model = joblib.load('dt_model_compressed_high.pkl')

# Title of the app
st.title("Credit Score Prediction App")

# Function to get user inputs
def get_user_input():
    col1, col2 = st.columns(2)

    with col1:
        Annual_Income = st.number_input('Annual Income', min_value=0.0, step=0.1)
        Num_of_Loan = st.number_input('Number of Loans', min_value=0, step=1)
        Num_of_Delayed_Payment = st.number_input('Number of Delayed Payments', min_value=0, step=1)
        Num_Credit_Inquiries = st.number_input('Number of Credit Inquiries', min_value=0, step=1)
        Credit_Utilization_Ratio = st.number_input('Credit Utilization Ratio', min_value=0.0, step=0.1)
        Total_EMI_per_month = st.number_input('Total EMI per month', min_value=0.0, step=0.1)

       
    with col2:
        Interest_Rate = st.number_input('Interest Rate', min_value=0.0, step=0.1)
        Delay_from_due_date = st.number_input('Delay from due date', min_value=0.0, step=0.1)
        Changed_Credit_Limit = st.number_input('Changed Credit Limit', min_value=0.0, step=0.1)
        Outstanding_Debt = st.number_input('Outstanding Debt', min_value=0.0, step=0.1)
        Credit_History_Age = st.number_input('Credit History Age', min_value=0.0, step=0.1)
        Monthly_Balance = st.number_input('Monthly Balance', min_value=0.0, step=0.1)
        

    # Create an input array for the model
    input_data = np.array([Annual_Income, Interest_Rate, Num_of_Loan, Delay_from_due_date, 
                           Num_of_Delayed_Payment, Changed_Credit_Limit,Num_Credit_Inquiries,Outstanding_Debt,
                           Credit_Utilization_Ratio, Credit_History_Age,Total_EMI_per_month,Monthly_Balance]).reshape(1, -1)
    
    return input_data

# Get user input
user_input = get_user_input()

#Center the Predict button using CSS
st.markdown("""
    <style>
    .stButton>button {
        display: block;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)


# Button to make predictions
if st.button("Predict Credit Score"):
    # Make prediction
    prediction = model.predict(user_input)
    
    # Display the result
    st.subheader(f'The predicted credit score is: {prediction[0]}')
