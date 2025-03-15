import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("fake_job_postings.csv")  # Change this to your actual dataset file

# 🔵 Step 1: Identify Categorical Columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# 🔵 Step 2: Encode Categorical Features
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))  # Convert all object columns to numeric

# 🔵 Step 3: Prepare Features and Target Variable
X = df.drop(columns=["fraudulent", "job_id"])  # Drop target & unnecessary columns
y = df["fraudulent"]

# 🔵 Step 4: Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\n🔄 Class distribution after SMOTE:\n", y_resampled.value_counts())

# 🔵 Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 🔵 Step 6: Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔵 Step 7: Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 🔵 Step 8: Make Predictions
y_pred = model.predict(X_test_scaled)

# 🔵 Step 9: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.2f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 🔵 Step 10: Save the Trained Model
joblib.dump(model, "fake_job_detector.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model training complete! Model saved as 'fake_job_detector.pkl'")
