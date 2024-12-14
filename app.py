import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data():
    try:
        # Load your dataset (replace with your actual dataset path)
        data = pd.read_csv("your_dataset.csv")
        return data
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        raise
    except Exception as e:
        print(f"Error while loading data: {e}")
        raise

def preprocess_data(data):
    # Handle missing values and other preprocessing tasks
    # Example: Fill missing values with the mean (this should be adapted to your dataset)
    data = data.fillna(data.mean())
    # You can add more preprocessing steps here (like encoding categorical columns)
    return data

def train_model(data):
    # Check if 'loan_status' exists in the columns before dropping it
    if "loan_status" in data.columns:
        X = data.drop(columns=["loan_status"])
        y = data["loan_status"]
    else:
        raise KeyError("The 'loan_status' column is missing from the dataset.")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Optionally, save the model
    joblib.dump(model, "loan_defaulter_model.pkl")
    
    return model, X.columns.tolist()

# Main flow
if __name__ == "__main__":
    try:
        data = load_data()
        processed_data = preprocess_data(data)
        model, feature_names = train_model(processed_data)
    except KeyError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
