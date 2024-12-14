import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Title of the app
st.title("Loan Defaulter Prediction App")

# Function to load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data(file_path="LCDataDictionary.xlsx"):
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)

        # Drop unnecessary columns
        X = df.drop(columns=["loan_status", "id", "member_id"], errors="ignore")  # Features
        y = df["loan_status"] if "loan_status" in df.columns else None  # Target

        if y is None:
            st.warning("The 'loan_status' column is missing in the dataset.")
        return X, y
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        return None, None

# Load the dataset
X, y = load_and_preprocess_data()

if X is not None and y is not None:
    # Train/Test Split
    st.subheader("Train the Model")
    test_size = st.slider("Select Test Data Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predictions on test data
    y_pred = model.predict(X_test)

    # Display Model Performance
    st.subheader("Model Performance")
    st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Predict defaulters based on user input
    st.subheader("Predict Loan Default")
    st.write("Enter details to predict if a loan will default:")

    # Create input fields for user-provided data
    input_data = {}
    for col in X.columns:
        input_data[col] = st.text_input(f"Enter {col}:", "")

    # When the user clicks "Predict"
    if st.button("Predict"):
        try:
            # Convert input data to a DataFrame
            input_df = pd.DataFrame([input_data])
            input_df = input_df.apply(pd.to_numeric, errors='coerce')  # Handle numerical input

            # Handle missing/invalid values
            if input_df.isnull().values.any():
                st.error("Please fill in all fields with valid numerical values.")
            else:
                # Scale the input data
                input_scaled = scaler.transform(input_df)

                # Make prediction
                prediction = model.predict(input_scaled)
                st.success(f"The loan is predicted to {'Default' if prediction[0] else 'Not Default'}.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.warning("The dataset couldn't be loaded or 'loan_status' column is missing. Please check the dataset.")

# Footer
st.sidebar.markdown("Developed by Preksha Barjatya")
