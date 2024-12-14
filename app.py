import streamlit as st
import pandas as pd
import pickle

# Load the trained model
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_data(data):
    # Drop unnecessary columns
    columns_to_drop = ['id', 'member_id', 'url', 'zip_code', 'addr_state', 'desc']
    data = data.drop(columns=columns_to_drop, errors='ignore')

    # Fill missing values
    data = data.fillna(data.mean(numeric_only=True))

    # Encode categorical columns
    categorical_columns = ['term', 'grade', 'sub_grade', 'emp_title', 'home_ownership', 
                           'verification_status', 'purpose', 'title', 'application_type']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    return data

def run():
    st.title("Bank Loan Default Prediction")

    # Load model
    model_path = "ML_Model.pkl"
    model = load_model(model_path)

    # Upload dataset
    uploaded_file = st.file_uploader("Upload your dataset (.xlsx)", type=["xlsx"])

    if uploaded_file:
        data = pd.read_excel(uploaded_file)

        # Preprocess dataset
        processed_data = preprocess_data(data)

        # Ensure model and data have the same columns
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in processed_data.columns:
                processed_data[col] = 0  # Add missing columns with default values
        processed_data = processed_data[model_features]

        # Make predictions
        predictions = model.predict(processed_data)
        data['Prediction'] = predictions

        st.write("Predictions completed! Here are the results:")
        st.write(data.head())

        # Download the results
        st.download_button(
            label="Download Predictions",
            data=data.to_csv(index=False),
            file_name="loan_predictions.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    run()
