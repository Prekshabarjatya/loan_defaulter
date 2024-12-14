import streamlit as st
import pickle
import os

# Function to load the trained model
def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        st.error(f"Error: {model_path} not found!")
        return None

# Function to run the Streamlit app
def run():
    # Title of the app
    st.title("Bank Loan Prediction using Machine Learning")

    # Input fields for user data
    account_no = st.text_input('Account number')
    fn = st.text_input('Full Name')

    gen_display = ('Female', 'Male')
    gen_options = list(range(len(gen_display)))
    gen = st.selectbox("Gender", gen_options, format_func=lambda x: gen_display[x])

    mar_display = ('No', 'Yes')
    mar_options = list(range(len(mar_display)))
    mar = st.selectbox("Marital Status", mar_options, format_func=lambda x: mar_display[x])

    dep_display = ('No', 'One', 'Two', 'More than Two')
    dep_options = list(range(len(dep_display)))
    dep = st.selectbox("Dependents", dep_options, format_func=lambda x: dep_display[x])

    edu_display = ('Not Graduate', 'Graduate')
    edu_options = list(range(len(edu_display)))
    edu = st.selectbox("Education", edu_options, format_func=lambda x: edu_display[x])

    emp_display = ('Job', 'Business')
    emp_options = list(range(len(emp_display)))
    emp = st.selectbox("Employment Status", emp_options, format_func=lambda x: emp_display[x])

    prop_display = ('Rural', 'Semi-Urban', 'Urban')
    prop_options = list(range(len(prop_display)))
    prop = st.selectbox("Property Area", prop_options, format_func=lambda x: prop_display[x])

    cred_display = ('Between 300 to 500', 'Above 500')
    cred_options = list(range(len(cred_display)))
    cred = st.selectbox("Credit Score", cred_options, format_func=lambda x: cred_display[x])

    mon_income = st.number_input("Applicant's Monthly Income($)", value=0)
    co_mon_income = st.number_input("Co-Applicant's Monthly Income($)", value=0)
    loan_amt = st.number_input("Loan Amount", value=0)

    dur_display = ['2 Month', '6 Month', '8 Month', '1 Year', '16 Month']
    dur_options = range(len(dur_display))
    dur = st.selectbox("Loan Duration", dur_options, format_func=lambda x: dur_display[x])

    # Load the model
    model_path = './ML_Model.pkl'  # Ensure this is the correct path to the model file
    model = load_model(model_path)

    # Check if model loaded successfully
    if model:
        # Handle loan duration input
        duration = 0
        if dur == 0:
            duration = 60
        elif dur == 1:
            duration = 180
        elif dur == 2:
            duration = 240
        elif dur == 3:
            duration = 360
        elif dur == 4:
            duration = 480

        # Prepare features for prediction
        features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop]]

        # Prediction button
        if st.button("Submit"):
            prediction = model.predict(features)
            result = int("".join(map(str, prediction)))

            # Display result based on prediction
            if result == 0:
                st.error(
                    f"Hello: {fn} || Account number: {account_no} || "
                    "According to our calculations, you will not get the loan from the bank."
                )
            else:
                st.success(
                    f"Hello: {fn} || Account number: {account_no} || "
                    "Congratulations!! You will get the loan from the bank."
                )

# Run the Streamlit app
if __name__ == "__main__":
    run()
