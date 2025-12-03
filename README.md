# Credit Card Fraud Detection App

Demo Link: https://credit-card-fraud-detection-is6e.onrender.com/

This is a Flask-based web application that uses a Machine Learning model (Random Forest) to detect fraudulent credit card transactions.

## Features
- **Real-time Prediction**: Predicts if a transaction is fraudulent based on transaction details.
- **Machine Learning**: Uses a Random Forest Classifier trained on a large dataset.
- **Web Interface**: Simple and user-friendly web interface for testing.
- **API Endpoint**: REST API endpoint for programmatic access.

## Tech Stack
- **Backend**: Python, Flask
- **ML Libraries**: scikit-learn, pandas, numpy, joblib
- **Frontend**: HTML, CSS, JavaScript

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Khansahab2/Credit-Card-Fraud-Detection-.git
    cd Credit-Card-Fraud-Detection-
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    python app.py
    ```
    The app will be available at `http://localhost:5000`.

## Usage

### Web Interface
Visit `http://localhost:5000` in your browser. Fill in the transaction details and click "Predict".

### API Endpoint
**URL**: `/predict`
**Method**: `POST`
**Content-Type**: `application/json`

**Request Body**:
```json
{
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "misc_net",
    "amt": 4.97,
    "gender": "F",
    "job": "Psychologist, counselling",
    "trans_date_trans_time": "2019-01-01 00:00:18"
}
```

**Response**:
```json
{
    "fraud": false,
    "message": "Transaction is SAFE ✅",
    "probability": 0.0
}
```

## Training the Model
To retrain the model, run:
```bash
python train_model.py
```
This will generate `fraud_detection_model.pkl` and `label_encoder.jb`.

## Deployment
This app is ready for deployment on platforms like Render.
1.  Push code to GitHub.
2.  Connect repository to Render Web Service.
3.  Build Command: `pip install -r requirements.txt`
4.  Start Command: `gunicorn app:app`
