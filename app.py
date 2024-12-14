import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Function to load data
def load_data():
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        return data
    else:
        st.warning("Please upload an Excel file.")
        return None

# Preprocessing the data
def preprocess_data(data):
    # Drop columns that are not useful for modeling
    drop_columns = [
        "id", "member_id", "emp_title", "url", "desc", "zip_code", "title", "next_pymnt_d"
    ]
    data = data.drop(columns=drop_columns, errors='ignore')

    # Handle missing values (simple imputation with median)
    data = data.fillna(data.median(numeric_only=True))

    # Convert categorical columns to dummy variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    return data

# Train the Random Forest model
def train_model(data):
    # Split into features and target
    X = data.drop(columns=["loan_status"])
    y = (data["loan_status"] == "Charged Off").astype(int)  # Binary classification: Default vs Non-default

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Model Accuracy:", accuracy)
    st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    st.write("Classification Report:", classification_report(y_test, y_pred))

    return model, X.columns

# Streamlit app
st.title("Loan Default Prediction App")
st.write("This app predicts whether a loan will default using a Random Forest model.")

# Load and preprocess data
st.header("Data Loading and Preprocessing")
data = load_data()
if data is not None:
    st.write("Dataset Loaded:", data.head())
    processed_data = preprocess_data(data)
    st.write("Processed Data:", processed_data.head())

    # Train model
    st.header("Model Training")
    model, feature_names = train_model(processed_data)

    # Save the model
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)

    st.write("Model training complete and saved as random_forest_model.pkl.")

    # Prediction
    st.header("Make Predictions")

    def user_input_features():
        input_data = {}
        for feature in feature_names:
            input_data[feature] = st.sidebar.number_input(feature, value=0.0)
        return pd.DataFrame([input_data])

    input_df = user_input_features()
    st.write("User Input Features:", input_df)

    if st.button("Predict"):
        loaded_model = pickle.load(open("random_forest_model.pkl", "rb"))
        prediction = loaded_model.predict(input_df)
        prediction_proba = loaded_model.predict_proba(input_df)

        st.write("Prediction:", "Default" if prediction[0] == 1 else "Non-default")
        st.write("Prediction Probability:", prediction_proba)
