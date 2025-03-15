import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load trained model, scaler, encoders, and feature order
with open("JobFraudModel.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("JobFraudScaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("JobFraudEncoders.pkl", "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)
with open("JobFraudFeatures.pkl", "rb") as feature_file:
    expected_features = pickle.load(feature_file)  # Ensure correct order

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 📝 1️⃣ Get input data
        input_data = request.form.to_dict()

        # 🏷️ 2️⃣ Apply label encoding from saved encoders
        for col, encoder in label_encoders.items():
            if col in input_data:
                value = str(input_data[col])  # Ensure string format
                input_data[col] = (
                    encoder.transform([value])[0] if value in encoder.classes_ else 0
                )

        # 📝 3️⃣ Convert input data to DataFrame
        df = pd.DataFrame([input_data])

        # 🔢 4️⃣ Convert numerical values
        df = df.apply(pd.to_numeric, errors='coerce')

        # ✅ 5️⃣ Ensure all required columns exist
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0  # Fill missing columns with default values

        # 🔄 6️⃣ Ensure correct feature order
        df = df[expected_features]

        # 🔬 7️⃣ Scale input features
        df_scaled = scaler.transform(df)

        # 🔮 8️⃣ Make prediction
        prediction = model.predict(df_scaled)
        result = "Fake Job Posting 🚨" if prediction[0] == 1 else "Real Job Posting ✅"

        return render_template('index.html', prediction_text=f'Result: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
