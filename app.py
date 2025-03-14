import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load the trained model
with open("priceModel.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# ✅ Define the exact features your model was trained on
expected_features = [
    "telecommuting", "has_company_logo", "has_questions", "employment_type",
    "required_experience", "required_education", "industry", "function"
]

# ✅ List of all dataset columns
all_columns = [
    "job_id", "title", "location", "department", "salary_range", "company_profile",
    "description", "requirements", "benefits", "telecommuting", "has_company_logo",
    "has_questions", "employment_type", "required_experience", "required_education",
    "industry", "function", "fraudulent"
]

# ✅ Remove non-essential columns
columns_to_remove = [
    "job_id", "title", "location", "department", "salary_range", "company_profile",
    "description", "requirements", "benefits", "fraudulent"
]

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1️⃣ Get form input data
        input_data = request.form.to_dict()
        print("Received Data:", input_data)  # Debugging output

        # 2️⃣ Convert input data to a DataFrame
        df = pd.DataFrame([input_data])

        # 3️⃣ Convert numerical values properly
        df = df.apply(pd.to_numeric, errors='coerce')

        # 4️⃣ Remove unwanted columns
        df = df.drop(columns=columns_to_remove, errors='ignore')

        # 5️⃣ Ensure only required columns remain
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0  # Default value for missing columns

        # 6️⃣ Ensure column order matches training
        df = df[expected_features]

        # 7️⃣ Debug processed data
        print("Processed DataFrame:\n", df)

        # 8️⃣ Make prediction
        prediction = model.predict(df)

        return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
