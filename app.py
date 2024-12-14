import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

# Function to preprocess the data
def preprocess_data(data):
    # Handle missing values
    numeric_cols = data.select_dtypes(include=["number"]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    categorical_cols = data.select_dtypes(include=["object"]).columns
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    return data, label_encoders

# Function to train and save a model
def train_model(data, model_path):
    # Select relevant features
    features = [
        "loan_amnt", "annual_inc", "dti", "term", "int_rate",
        "installment", "emp_length", "home_ownership", "purpose"
    ]
    target = "loan_status"

    # Ensure the target column is binary
    data[target] = data[target].apply(lambda x: 1 if x == "Fully Paid" else 0)

    # Split the data into training and test sets
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    st.success("Model trained and saved successfully!")

    return model

# Streamlit app
def run_app():
    st.title("Loan Status Prediction")

    # Load dataset
    data_file = st.file_uploader("Upload your dataset (Excel format)", type=["xlsx"])
    if data_file:
        data = pd.read_excel(data_file)

        # Preprocess the data
        data, label_encoders = preprocess_data(data)

        # Train or load the model
        model_path = Path("loan_model.pkl")
        if not model_path.exists():
            st.write("Training a new model...")
            model = train_model(data, model_path)
        else:
            st.write("Loading the existing model...")
            with open(model_path, "rb") as file:
                model = pickle.load(file)

        st.write("Model is ready for predictions!")

        # Input features for prediction
        loan_amnt = st.number_input("Loan Amount ($)", value=5000)
        annual_inc = st.number_input("Annual Income ($)", value=60000)
        dti = st.number_input("Debt-to-Income Ratio (%)", value=15.0)
        term = st.selectbox("Loan Term", ["36 months", "60 months"])
        int_rate = st.number_input("Interest Rate (%)", value=10.5)
        installment = st.number_input("Monthly Installment ($)", value=150)
        emp_length = st.selectbox("Employment Length", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", 
                                                        "6 years", "7 years", "8 years", "9 years", "10+ years"])
        home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage", "Other"])
        purpose = st.selectbox("Purpose of Loan", ["credit_card", "debt_consolidation", "home_improvement", "major_purchase", 
                                                   "medical", "moving", "vacation", "wedding", "other"])

        # Preprocess the input
        input_data = pd.DataFrame({
            "loan_amnt": [loan_amnt],
            "annual_inc": [annual_inc],
            "dti": [dti],
            "term": [term],
            "int_rate": [int_rate],
            "installment": [installment],
            "emp_length": [emp_length],
            "home_ownership": [home_ownership],
            "purpose": [purpose]
        })

        # Encode the input data
        for col in input_data.columns:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])

        # Make prediction
        if st.button("Predict Loan Status"):
            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.success("Loan Status: Fully Paid")
            else:
                st.error("Loan Status: Charged Off")

# Run the app
if __name__ == "__main__":
    run_app()
