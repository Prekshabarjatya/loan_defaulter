import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Title of the app
st.title("Loan Defaulter Prediction App")

# Function to load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data(file_path="LCDataDictionary.xlsx"):
    try:
        # Load the dataset
        df = pd.read_excel(file_path)

        # Display dataset preview
        st.write("Dataset Preview:")
        st.write(df.head())

        # Check if 'loan_status' exists
        if 'loan_status' not in df.columns:
            st.error("The 'loan_status' column is missing in the dataset.")
            return None, None
        
        # Prepare features (X) and target (y)
        X = df.drop(columns=["loan_status", "id", "member_id"], errors="ignore")
        y = df["loan_status"]

        return X, y
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        return None, None

# Load the dataset
X, y = load_and_preprocess_data()

if X is not None and y is not None:
    # Handle missing values
    X = X.fillna(0)  # Replace missing values with 0

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(X.describe())

    # Encode target variable
    y = y.astype('category').cat.codes  # Convert to numeric categories

    # Train-Test Split
    st.subheader("Model Training")
    test_size = st.slider("Select Test Data Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Display metrics
    st.subheader("Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

else:
    st.warning("The dataset couldn't be loaded or 'loan_status' column is missing. Please check the dataset.")

# Footer
st.sidebar.markdown("Developed by Preksha Barjatya")
