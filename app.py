import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

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

# Preprocess the data (handle missing values, etc.)
def preprocess_data(data):
    # Handle missing values (simple example)
    data = data.dropna()
    return data

# Train the model
def train_model(data):
    # Feature selection and target variable
    # Assuming the column indicating loan default is 'default'
    X = data.drop(columns=["default"])  # Remove the target column from features
    y = data["default"]  # Target variable (1 for default, 0 for no default)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.write(classification_report(y_test, y_pred))
    
    return model

# Visualize the data
def plot_data(data):
    st.subheader("Loan Default Status Distribution")
    sns.countplot(x="default", data=data)
    st.pyplot()

    # Additional visualizations can be added here

# Build the main app function
def app():
    st.title("LendingClub Loan Default Risk Prediction")
    
    # File uploader for Excel file
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    
    # Load and preprocess the data
    data = load_data(uploaded_file)
    if data is not None:
        data = preprocess_data(data)
        
        # Display basic statistics and data exploration
        st.subheader("Data Preview")
        st.write(data.head())
        
        # Show data visualizations
        plot_data(data)
        
        # Train the machine learning model
        model = train_model(data)
        
        # User input for prediction
        st.subheader("Loan Application Prediction")
        
        applicant_income = st.number_input("Applicant Income", min_value=1000, max_value=1000000, step=1000)
        loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=1000000, step=1000)
        # Additional features can be added here
        
        # Create a feature vector for the user input
        user_input = [[applicant_income, loan_amount]]
        
        # Predict loan default status
        if st.button("Predict"):
            prediction = model.predict(user_input)
            if prediction == 1:
                st.write("Loan is likely to default (Charged-off).")
            else:
                st.write("Loan is likely to be fully paid.")
    else:
        st.write("Please upload an Excel file to proceed.")
    
# Run the app
if __name__ == "__main__":
    app()
