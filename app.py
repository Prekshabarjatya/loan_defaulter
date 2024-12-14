import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Function to load the dataset
def load_data(file_path):
    try:
        data = pd.read_excel(file_path)  # Use Excel file
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Function to preprocess data
def preprocess_data(data):
    # Handling missing values
    data = data.fillna(data.mean())
    # Ensure 'loan_status' column exists
    if 'loan_status' not in data.columns:
        st.error("Error: 'loan_status' column is missing in the dataset!")
        return None
    return data

# Function to train and save the model
def train_and_save_model(data, model_path):
    X = data.drop(columns=['loan_status'])
    y = data['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    st.success("Model trained and saved successfully!")
    return model

# Function to load or create the model
def load_or_train_model(model_path, data):
    if not model_path.exists():
        st.warning("Model file not found. Training a new model...")
        return train_and_save_model(data, model_path)
    else:
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

# Main app function
def run():
    st.title("Bank Loan Prediction")

    # File upload for data
    uploaded_file = st.file_uploader("Upload Dataset (Excel format)", type=["xlsx"])
    model_path = "ML_Model.pkl"  # Path to save/load model

    if uploaded_file:
        # Load and preprocess the data
        data = load_data(uploaded_file)
        if data is not None:
            data = preprocess_data(data)

            # Train or load the model
            model = load_or_train_model(model_path, data)

            if model:
                # Input fields for prediction
                gen = st.selectbox("Gender", options=["Female", "Male"])
                mar = st.selectbox("Marital Status", options=["No", "Yes"])
                dep = st.selectbox("Dependents", options=["No", "One", "Two", "More than Two"])
                edu = st.selectbox("Education", options=["Not Graduate", "Graduate"])
                emp = st.selectbox("Employment Status", options=["Job", "Business"])
                mon_income = st.number_input("Applicant's Monthly Income($)", value=0)
                co_mon_income = st.number_input("Co-Applicant's Monthly Income($)", value=0)
                loan_amt = st.number_input("Loan Amount", value=0)
                dur = st.selectbox("Loan Duration", options=["2 Month", "6 Month", "8 Month", "1 Year", "16 Month"])
                cred = st.selectbox("Credit Score", options=["Between 300 to 500", "Above 500"])
                prop = st.selectbox("Property Area", options=["Rural", "Semi-Urban", "Urban"])

                # Map categorical inputs to numeric values (example mapping, adjust as needed)
                duration_mapping = {"2 Month": 60, "6 Month": 180, "8 Month": 240, "1 Year": 360, "16 Month": 480}
                feature_values = [
                    int(gen == "Male"), int(mar == "Yes"), dep, int(edu == "Graduate"),
                    int(emp == "Business"), mon_income, co_mon_income, loan_amt,
                    duration_mapping[dur], int(cred == "Above 500"), prop
                ]

                # Predict and display results
                if st.button("Submit"):
                    prediction = model.predict([feature_values])
                    if prediction[0] == 0:
                        st.error("Loan application is likely to be rejected.")
                    else:
                        st.success("Loan application is likely to be approved.")

# Run the app
if __name__ == "__main__":
    run()
