import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Function to load and preprocess the data
def load_and_preprocess_data(file):
    # Read the Excel file
    data = pd.read_excel(file, engine='openpyxl')
    
    # Display the first few rows of the data for inspection
    st.write("Preview of Dataset:")
    st.write(data.head())
    
    # Display the columns to verify the correct column names
    st.write("Columns in the dataset:")
    st.write(data.columns)
    
    # Normalize column names (convert to lowercase and remove extra spaces)
    data.columns = data.columns.str.strip().str.lower()
    
    # Check if 'LoanStatNew' exists in the columns (assuming this is your target column)
    if 'loanstatnew' in data.columns:
        X = data.drop(columns=["loanstatnew", "id", "member_id"])  # Drop 'LoanStatNew', 'id', and 'member_id'
        y = data["loanstatnew"].apply(lambda x: 1 if x == "Default" else 0)  # Convert LoanStatNew to binary (1 = Default, 0 = Not Default)
    else:
        st.error("Target column 'LoanStatNew' is missing from the dataset.")
        return None, None
    
    # Handle missing values by filling with the median
    X.fillna(X.median(), inplace=True)
    
    # Standardize the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Function to train and evaluate the Random Forest model
def train_random_forest(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and display the classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    st.write("Classification Report:")
    st.text(report)
    
    return model

# Function to predict and show the confidence
def predict_with_confidence(model, X):
    # Predict probabilities for each class
    proba = model.predict_proba(X)
    # Confidence is the probability of the predicted class
    confidence = proba.max(axis=1)
    prediction = model.predict(X)
    
    return prediction, confidence

# Streamlit App
st.title("Loan Default Prediction App")
st.write("Upload your dataset (XLSX format) to predict whether a borrower will default on their loan repayment.")

# File uploader for dataset
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    X, y = load_and_preprocess_data(uploaded_file)
    
    if X is not None and y is not None:
        # Train Random Forest model
        model = train_random_forest(X, y)
        
        # Take new data input for prediction (you can modify this to get data from user input in the future)
        st.write("Now you can enter data for prediction.")
        
        # Example input form (you can modify this as per your input structure)
        num_columns = X.shape[1]  # number of features in the dataset
        input_data = []
        for i in range(num_columns):
            input_data.append(st.number_input(f"Enter value for feature {X.columns[i]}", value=0))
        
        input_data = [input_data]  # Convert to 2D array
        
        # Standardize the input data
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_data)
        
        # Get predictions and confidence
        prediction, confidence = predict_with_confidence(model, input_scaled)
        
        # Display results
        if prediction == 1:
            st.write("Prediction: Default")
        else:
            st.write("Prediction: Not Default")
        
        st.write(f"Confidence: {confidence[0] * 100:.2f}%")
        
        # Option to download model
        if st.button("Download Model"):
            joblib.dump(model, 'loan_default_model.pkl')
            st.write("Model saved as 'loan_default_model.pkl'")
