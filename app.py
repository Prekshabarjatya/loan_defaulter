import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to load and preprocess the data
def load_and_preprocess_data(file):
    # Read the Excel file
    data = pd.read_excel(file, engine='openpyxl')
    
    # Display the first few rows of the data
    st.write("Preview of Dataset:")
    st.write(data.head())
    
    # Columns to drop (change these as per the dataset)
    drop_columns = ["loan_status", "id", "member_id"]
    
    # Only drop columns that exist in the dataset
    data = data.drop(columns=[col for col in drop_columns if col in data.columns], errors='ignore')
    
    # Check if 'loan_status' is in the dataset and set it as target
    if 'loan_status' in data.columns:
        X = data.drop(columns=["loan_status"])  # drop 'loan_status' if it exists
        y = data["loan_status"].apply(lambda x: 1 if x == "Default" else 0)  # binary target (Default = 1, Not Default = 0)
    else:
        st.error("Target column 'loan_status' is missing from the dataset.")
        return None, None
    
    # Handle missing values by filling with median (you can customize this approach)
    X.fillna(X.median(), inplace=True)
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Function to train the RandomForest model
def train_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

# Function to predict and display the results
def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for "Default" class
    return y_pred, y_prob

# File uploader for Excel (.xlsx)
uploaded_file = st.file_uploader("Upload your LCDataDictionary.xlsx file here", type=["xlsx"])

if uploaded_file is not None:
    # Load and preprocess the data
    X, y = load_and_preprocess_data(uploaded_file)
    
    if X is not None and y is not None:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the Random Forest model
        model = train_model(X_train, y_train)
        
        # Predict on the test set and get the confidence (probabilities)
        y_pred, y_prob = make_predictions(model, X_test)
        
        # Calculate accuracy on the test set
        accuracy = accuracy_score(y_test, y_pred)
        
        # Display model results
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
        
        # Display confidence for each test sample
        st.write("Prediction Results (Test Set):")
        results = pd.DataFrame({
            'Predicted Default': y_pred,
            'Confidence (Probability)': y_prob
        })
        
        st.write(results.head())  # Preview the results

        # Option to test a new input (prediction from user)
        st.write("Make a prediction for a new borrower:")
        
        # Example: Input fields for the user to provide data for a new borrower
        loan_amnt = st.number_input("Loan Amount", min_value=0)
        annual_inc = st.number_input("Annual Income", min_value=0)
        dti = st.number_input("Debt-to-Income Ratio", min_value=0.0)
        term = st.selectbox("Loan Term", options=["36 months", "60 months"])
        int_rate = st.number_input("Interest Rate", min_value=0.0)
        
        # Additional fields can be added here based on the dataset
        
        # If the user has inputted data, make predictions
        if st.button("Predict Borrower Default"):
            # Prepare the input data for prediction
            new_data = np.array([[loan_amnt, annual_inc, dti, int_rate]])  # Modify based on your features
            new_data_scaled = StandardScaler().fit_transform(new_data)  # Standardize the input data
            
            # Predict using the trained model
            prediction = model.predict(new_data_scaled)
            prediction_prob = model.predict_proba(new_data_scaled)[:, 1]
            
            # Display prediction and confidence
            if prediction[0] == 1:
                st.write(f"The borrower is predicted to default with {prediction_prob[0] * 100:.2f}% confidence.")
            else:
                st.write(f"The borrower is predicted NOT to default with {(1 - prediction_prob[0]) * 100:.2f}% confidence.")

else:
    st.info("Please upload the LCDataDictionary.xlsx file to proceed.")
