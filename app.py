import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

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
    
    # Check if 'loan_status' exists in the columns
    if 'loan_status' in data.columns:
        X = data.drop(columns=["loan_status", "id", "member_id"])  # Drop 'loan_status', 'id', and 'member_id'
        y = data["loan_status"].apply(lambda x: 1 if x == "Default" else 0)  # Convert loan status to binary (1 = Default, 0 = Not Default)
    else:
        st.error("Target column 'loan_status' is missing from the dataset.")
        return None, None
    
    # Handle missing values by filling with the median
    X.fillna(X.median(), inplace=True)
    
    # Standardize the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Streamlit UI setup
st.title("Loan Default Prediction App")

# File upload feature
uploaded_file = st.file_uploader("Upload Your Excel File", type=["xlsx"])

if uploaded_file is not None:
    X, y = load_and_preprocess_data(uploaded_file)

    if X is not None and y is not None:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of default (class 1)

        # Display results
        st.write(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Display a prediction example
        st.write("Prediction Example (First Record in Test Set):")
        example_index = 0
        example_pred = y_pred[example_index]
        example_pred_prob = y_pred_prob[example_index]
        st.write(f"Predicted: {'Default' if example_pred == 1 else 'No Default'}")
        st.write(f"Confidence: {example_pred_prob * 100:.2f}%")
        
        # Prediction button
        st.subheader("Make a Prediction")
        input_data = []
        for column in X.columns:
            input_data.append(st.number_input(f"Enter value for {column}", value=0.0))

        # Make prediction on new data
        if st.button("Predict"):
            prediction = model.predict([input_data])
            confidence = model.predict_proba([input_data])[0][1]  # Get probability of default
            st.write(f"Prediction: {'Default' if prediction[0] == 1 else 'No Default'}")
            st.write(f"Confidence: {confidence * 100:.2f}%")
