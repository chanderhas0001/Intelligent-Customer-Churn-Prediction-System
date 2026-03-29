
from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_COLUMNS = [
    'account_length','area_code','international_plan','voice_mail_plan',
    'number_vmail_messages','total_day_minutes','total_day_calls',
    'total_eve_minutes','total_eve_calls','total_night_minutes',
    'total_night_calls','total_intl_minutes','total_intl_calls',
    'number_customer_service_calls'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_data = [data[col] for col in FEATURE_COLUMNS]
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "churn_probability": float(prob)
    })

if __name__ == '__main__':
    app.run()
