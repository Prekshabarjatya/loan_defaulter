import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title("Loan Defaulter Prediction App")

# Function to load and preprocess the data
def load_and_preprocess_data(uploaded_file):
    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Display the dataset's columns to debug
        st.write("Columns in dataset:", data.columns.tolist())

        # Drop unnecessary columns (update column names if necessary)
        X = data.drop(columns=["LoanStatNew", "id", "member_id"], errors="ignore")
        
        # Ensure the target column exists
        if "loan_status" not in data.columns:
            st.error("Target column 'loan_status' is missing from the dataset.")
            st.stop()
        
        y = data["loan_status"]
        return X, y
    else:
        st.error("No file uploaded.")
        st.stop()

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# Load and preprocess the data
if uploaded_file is not None:
    X, y = load_and_preprocess_data(uploaded_file)

    # Display the first few rows of the dataset
    st.write("Preview of the dataset:")
    st.write(X.head())
    
    # Display dataset statistics
    st.write("Summary statistics of the dataset:")
    st.write(X.describe())
    
    # Data Visualization
    st.subheader("Data Visualization")
    
    # Distribution of annual income
    if "annual_inc" in X.columns:
        st.write("Distribution of Annual Income:")
        plt.figure(figsize=(10, 5))
        sns.histplot(X["annual_inc"], kde=True, bins=30)
        st.pyplot(plt.gcf())  # Display the plot
    else:
        st.warning("'annual_inc' column is missing from the dataset.")
    
    # Loan Status Distribution
    st.write("Loan Status Distribution:")
    plt.figure(figsize=(8, 4))
    sns.countplot(x=y)
    st.pyplot(plt.gcf())
    
    # Correlation heatmap
    st.write("Correlation Heatmap:")
    plt.figure(figsize=(12, 6))
    corr = X.select_dtypes(include=["float64", "int64"]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt.gcf())
    
    # Additional Analysis
    st.subheader("Additional Analysis")
    st.write("Perform further data analysis here...")
else:
    st.info("Please upload a CSV file to start.")

# Footer
st.sidebar.markdown("Developed by [Your Name]")

