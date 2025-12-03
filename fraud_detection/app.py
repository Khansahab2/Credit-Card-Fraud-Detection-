from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoder
model = joblib.load("fraud_detection_model.pkl")
encoder = joblib.load("label_encoder.jb")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect form data
            cc_num = float(request.form["cc_num"])
            merchant = request.form["merchant"]
            category = request.form["category"]
            amt = float(request.form["amt"])
            gender = request.form["gender"]
            job = request.form["job"]
            unix_time = float(request.form["unix_time"])
            merch_lat = float(request.form["merch_lat"])
            hour = int(request.form["hour"])
            day = int(request.form["day"])
            month = int(request.form["month"])

            # Encode categorical features
            merchant_enc = encoder.transform([merchant])[0] if merchant in encoder.classes_ else 0
            category_enc = encoder.transform([category])[0] if category in encoder.classes_ else 0
            gender_enc = encoder.transform([gender])[0] if gender in encoder.classes_ else 0
            job_enc = encoder.transform([job])[0] if job in encoder.classes_ else 0

            # Prepare input
            input_data = np.array([[cc_num, merchant_enc, category_enc, amt, gender_enc, job_enc,
                                    unix_time, merch_lat, hour, day, month]])

            # Prediction
            prediction = model.predict(input_data)[0]

            result = "⚠️ Fraudulent Transaction Detected!" if prediction == 1 else "✅ Legit Transaction"
            return render_template("result.html", result=result)

        except Exception as e:
            return render_template("result.html", result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
