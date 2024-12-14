import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Title of the app
st.title("Loan Defaulter Prediction App")

# Function to load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data(file_path="data/LCDataDictionary.xlsx"):
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        # Drop unnecessary columns (update according to your file)
        X = df.drop(columns=["loan_status", "id", "member_id"], errors="ignore")  # Features
        y = df["loan_status"] if "loan_status" in df.columns else None  # Target

        if y is None:
            st.warning("The 'loan_status' column is missing in the dataset.")
        return X, y
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        return None, None

# Load the dataset from the specified path
X, y = load_and_preprocess_data()

if X is not None:
    # Display the dataset
    st.write("Preview of the dataset:")
    st.write(X.head())

    # Display dataset statistics
    st.write("Summary statistics of the dataset:")
    st.write(X.describe())

    # Data Visualization
    st.subheader("Data Visualization")

    # Distribution of a numerical column
    if "annual_inc" in X.columns:
        st.write("Distribution of Annual Income:")
        plt.figure(figsize=(10, 5))
        sns.histplot(X["annual_inc"], kde=True, bins=10)
        st.pyplot(plt.gcf())

    # Loan Status Distribution
    if y is not None:
        st.write("Loan Status Distribution:")
        plt.figure(figsize=(8, 4))
        sns.countplot(x=y)
        st.pyplot(plt.gcf())

    # Correlation heatmap
    st.write("Correlation Heatmap:")
    plt.figure(figsize=(12, 6))
    corr = X.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt.gcf())

    # Train a simple RandomForest Classifier model
    if y is not None:
        st.subheader("Train a Loan Default Prediction Model")

        # Splitting the data
        st.write("Splitting the data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model Training
        st.write("Training the RandomForest Classifier...")
        model = RandomForestClassifier()
        model.fit(X_train_scaled, y_train)

        # Model Prediction
        y_pred = model.predict(X_test_scaled)

        # Model Evaluation
        st.write("### Model Evaluation:")
        st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Save the trained model
        joblib.dump(model, "loan_default_model.pkl")
        st.success("Model training complete and saved as 'loan_default_model.pkl'")

else:
    st.warning("The dataset couldn't be loaded. Please ensure the file exists and is correctly formatted.")

# Footer
st.sidebar.markdown("Developed by Team 33")
