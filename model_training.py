import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(df, target_column):
    """Preprocess data: Handle categorical variables and split features/target."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    
    # Drop missing values
    df = df.dropna().copy()  # Copy prevents SettingWithCopyWarning
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Apply Label Encoding to categorical features
    label_encoders = {}
    for col in categorical_cols:
        if col != target_column:  # Don't encode target column if it's categorical
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Store encoders for later use
    
    # Extract features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y  # Only return two values, fixing the unpacking error

def split_and_scale_data(X, y):
    """Split data into train-test sets and apply scaling."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train_scaled, y_train):
    """Train a RandomForestRegressor model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model

def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Model Evaluation:\nMean Squared Error: {mse:.4f}\nR2 Score: {r2:.4f}")

def save_model(model, scaler, model_filename="PriceModel.pkl", scaler_filename="PriceScaler.pkl"):
    """Save trained model and scaler to files."""
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_filename, "wb") as f:
        pickle.dump(scaler, f)
    print("Model and scaler saved successfully!")

def main():
    """Main function to run the training pipeline."""
    file_path = "fake_job_postings.csv"  # Update this if needed
    target_column = "fraudulent"  # Replace with actual target column name
    
    df = load_data(file_path)
    if df is not None:
        try:
            X, y = preprocess_data(df, target_column)
            X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
            model = train_model(X_train_scaled, y_train)
            evaluate_model(model, X_test_scaled, y_test)
            save_model(model, scaler)
            
            

        except Exception as e:
            print(f"Error in processing: {e}")


if __name__ == "__main__":
    main()
