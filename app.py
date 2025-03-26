from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load the trained model and encoders
model = joblib.load("fraud_detection_model.pkl")
encoder_transaction_type = joblib.load("encoder_transaction_type.pkl")
encoder_location = joblib.load("encoder_location.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Get JSON data from request
        data = request.get_json()

        # ✅ Ensure all required fields are present
        if not all(k in data for k in ["TransactionType", "Location", "Amount", "Time"]):
            return jsonify({"error": "Missing required fields"}), 400

        # ✅ Extract features safely with default values if needed
        transaction_type = data.get("TransactionType")
        location = data.get("Location")
        amount = data.get("Amount")
        time = data.get("Time")

        # ✅ Check if numerical fields are valid
        if amount is None or time is None:
            return jsonify({"error": "Amount and Time must be provided"}), 400

        try:
            amount = float(amount)
            time = float(time)
        except ValueError:
            return jsonify({"error": "Amount and Time must be valid numbers"}), 400

        # ✅ Convert categorical values using label encoders
        if transaction_type not in encoder_transaction_type.classes_:
            return jsonify({"error": f"Invalid TransactionType: '{transaction_type}'. Valid values are: {list(encoder_transaction_type.classes_)}"}), 400

        if location not in encoder_location.classes_:
            return jsonify({"error": f"Invalid Location: '{location}'. Valid values are: {list(encoder_location.classes_)}"}), 400

        transaction_type_encoded = encoder_transaction_type.transform([transaction_type])[0]
        location_encoded = encoder_location.transform([location])[0]

        # ✅ Prepare input array for model prediction
        features = np.array([[transaction_type_encoded, location_encoded, amount, time]])

        # ✅ Predict fraud or not
        prediction = model.predict(features)[0]
        result = "Fraud" if prediction == 1 else "Not Fraud"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
