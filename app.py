import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

if X is not None:
    # Display the dataset
    st.subheader("Dataset Preview")
    st.write(X.head())

    # Display dataset statistics
    st.subheader("Summary Statistics")
    st.write(X.describe())

    # Data Visualization
    st.subheader("Data Visualization")

    # Distribution of a numerical column
    if "annual_inc" in X.columns:
        st.write("Distribution of Annual Income:")
        plt.figure(figsize=(10, 5))
        sns.histplot(X["annual_inc"].dropna(), kde=True, bins=10)
        st.pyplot(plt.gcf())

    # Loan Status Distribution
    if y is not None:
        st.write("Loan Status Distribution:")
        plt.figure(figsize=(8, 4))
        sns.countplot(x=y)
        st.pyplot(plt.gcf())

    # Correlation heatmap
    st.write("Correlation Heatmap:")
    numeric_X = X.select_dtypes(include=["number"])  # Select only numeric columns
    numeric_X = numeric_X.fillna(0)  # Fill missing values with 0
    plt.figure(figsize=(12, 6))
    corr = numeric_X.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt.gcf())

    # Train a basic Random Forest Classifier
    st.subheader("Model Training")
    if y is not None:
        test_size = st.slider("Select Test Data Size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(numeric_X, y, test_size=test_size, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the Random Forest Classifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Display Metrics
        st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

else:
    st.warning("The dataset couldn't be loaded. Please ensure the file exists and is correctly formatted.")

# Footer
st.sidebar.markdown("Developed by Preksha Barjatya")
