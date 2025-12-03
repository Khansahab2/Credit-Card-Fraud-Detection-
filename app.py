from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import traceback
import os

app = Flask(__name__)

# Load model and encoders
MODEL_PATH = 'fraud_detection_model.pkl'
ENCODER_PATH = 'label_encoder.jb'

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("Model or encoder not found. Run train_model.py first.")

print("Loading model and encoders...")
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)
print("Model loaded.")

# Features expected by the model
MODEL_FEATURES = ['merchant', 'category', 'amt', 'gender', 'job', 'hour', 'day', 'month']

@app.route('/')
def index():
    # Pass categories and jobs for dropdowns if needed
    # We can extract them from encoders
    try:
        categories = list(encoders['category'].classes_)
        jobs = list(encoders['job'].classes_)
        # Limit for display if too many
        return render_template('index.html', categories=categories[:100], jobs=jobs[:100])
    except Exception as e:
        print(f"Error loading categories: {e}")
        return render_template('index.html', categories=[], jobs=[])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        input_data = {}
        
        # Handle Date/Time
        # Support both full timestamp or separate fields
        if 'trans_date_trans_time' in data and data['trans_date_trans_time']:
            try:
                dt = pd.to_datetime(data['trans_date_trans_time'])
                input_data['hour'] = dt.hour
                input_data['day'] = dt.day
                input_data['month'] = dt.month
            except:
                return jsonify({'error': 'Invalid date format'}), 400
        elif all(k in data for k in ['hour', 'day', 'month']):
            try:
                input_data['hour'] = int(data['hour'])
                input_data['day'] = int(data['day'])
                input_data['month'] = int(data['month'])
            except:
                return jsonify({'error': 'Invalid hour/day/month'}), 400
        else:
            # Fallback to current time if nothing provided? 
            # Or return error. Let's return error to be safe.
            return jsonify({'error': 'Missing date/time information'}), 400

        # Handle other features
        for col in ['merchant', 'category', 'amt', 'gender', 'job']:
            if col not in data:
                 return jsonify({'error': f'Missing {col}'}), 400
            input_data[col] = data[col]

        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocessing
        # Categorical Encoding
        for col in ['merchant', 'category', 'gender', 'job']:
            le = encoders[col]
            val = str(df.iloc[0][col])
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                # Handle unknown labels
                # Assign to the first class (usually 0) or handle gracefully
                print(f"Warning: Unknown label '{val}' for feature '{col}'. Assigning default.")
                df[col] = 0 

        # Ensure correct order
        X = df[MODEL_FEATURES]
        
        # Predict
        prob = model.predict_proba(X)[0][1]
        is_fraud = prob > 0.5
        
        return jsonify({
            'fraud': bool(is_fraud),
            'probability': round(float(prob), 4),
            'message': 'FRAUD DETECTED! 🚨' if is_fraud else 'Transaction is SAFE ✅'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)