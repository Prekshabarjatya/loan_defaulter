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

# Function to load and preprocess the dataset from a fixed Excel file
@st.cache
def load_and_preprocess_data(file_path="data/LoanData.xlsx"):
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        # Drop unnecessary columns if they exist
        X = df.drop(columns=["loan_status", "id", "member_id"], errors="ignore")  # Features
        y = df["loan_status"]  # Target
        return X, y
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        return None, None

# Load the dataset from a fixed path
X, y = load_and_preprocess_data()

if X is not None and y is not None:
    # Display the dataset
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
        sns.histplot(X["annual_inc"], kde=True, bins=10)
        st.pyplot(plt.gcf())

    # Loan Status Distribution
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

# Footer
st.sidebar.markdown("Developed by Team 33")

