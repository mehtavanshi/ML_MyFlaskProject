import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the target column
TARGET_COLUMN = "fraudulent"
DROP_COLUMNS = ["job_id", TARGET_COLUMN]  # Columns to drop before training


def load_data(file_path):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully!")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def preprocess_data(df):
    """Preprocess data: Handle categorical variables and split features/target."""
    df = df.dropna().copy()  # Drop rows with missing values

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Apply Label Encoding to categorical features
    label_encoders = {}
    for col in categorical_cols:
        if col not in DROP_COLUMNS:  # Avoid encoding target
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Store encoders for app usage

    # Extract features and target
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMN]

    return X, y, label_encoders


def split_and_scale_data(X, y):
    """Split data into train-test sets and apply scaling."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()  # Return feature order


def train_model(X_train_scaled, y_train):
    """Train a RandomForestClassifier model with tuned parameters."""
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    return model


def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model performance using accuracy and classification report."""
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"📊 Model Evaluation:\n✅ Accuracy: {accuracy:.4f}\n")
    print("📜 Classification Report:\n", report)
    


def save_model(model, scaler, label_encoders, feature_order):
    """Save trained model, scaler, encoders, and feature order to files."""
    with open("JobFraudModel.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("JobFraudScaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("JobFraudEncoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    with open("JobFraudFeatures.pkl", "wb") as f:
        pickle.dump(feature_order, f)

    print("✅ Model, scaler, encoders, and feature order saved successfully!")


def main():
    """Main function to run the training pipeline."""
    file_path = "fake_job_postings.csv"  # Update the dataset path if needed

    df = load_data(file_path)
    if df is not None:
        try:
            X, y, label_encoders = preprocess_data(df)
            X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_order = split_and_scale_data(X, y)
            model = train_model(X_train_scaled, y_train)
            evaluate_model(model, X_test_scaled, y_test)
            save_model(model, scaler, label_encoders, feature_order)
        except Exception as e:
            print(f"❌ Error in processing: {e}")


if __name__ == "__main__":
    main()
