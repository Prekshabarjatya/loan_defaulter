import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data (assumed to be in CSV format)
@st.cache
def load_data():
    data = pd.read_csv("loan_data.csv")
    return data

# Preprocess the data (handle missing values, etc.)
def preprocess_data(data):
    # Handle missing values (simple example)
    data = data.dropna()
    return data

# Train the model
def train_model(data):
    # Feature selection and target variable
    X = data.drop(columns=["loan_status"])
    y = data["loan_status"].apply(lambda x: 1 if x == "Charged-off" else 0)
    
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
    sns.countplot(x="loan_status", data=data)
    st.pyplot()

    # Additional visualizations can be added here

# Build the main app function
def app():
    st.title("LendingClub Loan Default Risk Prediction")
    
    # Load and preprocess the data
    data = load_data()
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
    
# Run the app
if __name__ == "__main__":
    app()
