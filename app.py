import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the pretrained model (Ensure you have a model saved in .pkl format)
# Pretrained model should be trained on similar dataset structure
@st.cache
def load_model():
    model = joblib.load('random_forest_model.pkl')  # Load your saved model here
    return model

# Load the data (use file uploader if the file is not found)
@st.cache
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        # Load data from the uploaded Excel file
        data = pd.read_excel(uploaded_file)
    else:
        # Handle the case where no file is uploaded
        data = None
    return data

# Function to match columns and extract relevant features
def match_columns(data, model):
    # Get the feature names used by the pretrained model
    feature_names = model.feature_names_in_

    # Let user match the dataset columns with model's expected columns
    matched_columns = {}
    
    for feature in feature_names:
        if feature in data.columns:
            matched_columns[feature] = feature
        else:
            matched_columns[feature] = None
    
    return matched_columns

# Function to predict loan default based on user input
def predict_loan_default(user_input, model, feature_names):
    # Ensure user input matches model's expected input features
    user_data = [user_input]
    prediction = model.predict(user_data)
    return prediction[0]

# Build the main app function
def app():
    st.title("LendingClub Loan Default Prediction")

    # Upload Excel file
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    # Load and preprocess the data
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.subheader("Data Preview")
        st.write(data.head())
        
        # Load the pretrained model
        model = load_model()

        # Match columns between the uploaded data and model's expected columns
        matched_columns = match_columns(data, model)
        st.subheader("Column Mapping")
        st.write(matched_columns)

        # Let the user input their data for prediction
        st.subheader("Enter Your Loan Details for Prediction")

        user_input = []
        
        # Collect user input for each feature
        for feature in matched_columns:
            if matched_columns[feature]:
                value = st.number_input(f"Enter {feature}", value=0)
                user_input.append(value)
            else:
                user_input.append(0)  # Default to 0 if the column is not matched

        # Button to trigger prediction
        if st.button("Predict Default"):
            prediction = predict_loan_default(user_input, model, matched_columns)
            if prediction == 1:
                st.write("Loan is likely to default (Charged-off).")
            else:
                st.write("Loan is likely to be fully paid.")

    else:
        st.write("Please upload an Excel file to proceed.")

# Run the app
if __name__ == "__main__":
    app()
