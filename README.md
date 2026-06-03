# Credit Card Fraud Detection

> Real-time AI-powered system for detecting credit card transaction fraud using Machine Learning

**Live Demo**: [https://credit-card-fraud-detection-is6e.onrender.com/](https://credit-card-fraud-detection-is6e.onrender.com/)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Khansahab2/Credit-Card-Fraud-Detection-)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/Khansahab2/Credit-Card-Fraud-Detection-)

This repository contains a complete, end-to-end Flask-based web application that leverages a Random Forest classifier to identify fraudulent credit card transactions in real-time. By analyzing various characteristics of a transaction (such as merchant, category, amount, gender, job, and transaction time), the system provides instant risk classification and detailed probability metrics.

---

## Features

- **Real-Time Prediction**: Instantly evaluates transaction parameters and predicts fraud probability.
- **Machine Learning Core**: Employs a Random Forest Classifier trained on a large dataset of credit card transactions for high precision.
- **Interactive Dashboard**: A clean and responsive web interface designed with Bootstrap 5, featuring dynamic dropdown menus populated directly from the trained label encoders.
- **RESTful API**: Out-of-the-box support for programmatic access via a JSON POST endpoint (`/predict`).
- **Flexible Training Pipeline**: Includes a standalone training script (`train_model.py`) to retrain and update models on new datasets.
- **Serialization Ready**: Model and label encoders are serialized using `joblib` for rapid loading and minimal inference latency.

---

## Project Structure

```text
├── fraud_detection/
│   └── dataset.csv              # Training dataset (excluded from git if too large)
├── static/
│   ├── script.js                # Frontend AJAX request handling & UI animations
│   └── style.css                # Custom CSS styling for the interface
├── templates/
│   └── index.html               # Main dashboard HTML template (Bootstrap 5)
├── Procfile                     # Process file for Heroku/Render deployment
├── README.md                    # Project documentation
├── app.py                       # Main Flask web application and API service
├── train_model.py               # ML training and preprocessing script
├── requirements.txt             # Python dependencies
├── fraud_detection_model.pkl    # Serialized Random Forest model artifact
├── fraud_model.joblib           # Serialized model backup
└── label_encoder.jb             # Serialized category/merchant/job/gender encoders
```

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- `pip` (Python package manager)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Khansahab2/Credit-Card-Fraud-Detection-.git
   cd Credit-Card-Fraud-Detection-
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Model Files Exist**:
   If the pre-trained `fraud_detection_model.pkl` and `label_encoder.jb` files are already present in the root directory, you can skip to running the app. Otherwise, train the model to generate them:
   ```bash
   python train_model.py
   ```

### Running the Application

Start the Flask dev server:
```bash
python app.py
```

The application will be running locally at `http://localhost:5000`.

---

## Usage Examples

### Web Interface
Navigate to `http://localhost:5000` in your web browser. Fill in the transaction details (such as Category, Amount, Merchant, Job, and DateTime fields) and click **Detect Fraud**. The interface will display a real-time notification indicating whether the transaction is safe or fraudulent along with a percentage progress bar.

### API Integration
You can integrate the prediction engine programmatically by sending JSON requests to the `/predict` endpoint.

#### API Endpoint Specifications
- **URL**: `/predict`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`

#### Request Schema
| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `merchant` | String | Name of the merchant | `"fraud_Rippin, Kub and Mann"` |
| `category` | String | Transaction category | `"misc_net"` |
| `amt` | Float | Transaction amount in USD | `4.97` |
| `gender` | String | Gender of the cardholder (`M`/`F`) | `"F"` |
| `job` | String | Job/Occupation of the cardholder | `"Psychologist, counselling"` |
| `trans_date_trans_time` | String | Date and time of transaction (format: `YYYY-MM-DD HH:MM:SS`) | `"2019-01-01 00:00:18"` |
| `hour` | Integer | Alternative: Hour of the transaction (0-23) | `0` |
| `day` | Integer | Alternative: Day of the month (1-31) | `1` |
| `month` | Integer | Alternative: Month of the year (1-12) | `1` |

> [!NOTE]
> You must provide either `trans_date_trans_time` OR the individual integer fields `hour`, `day`, and `month`.

#### Curl Request Example
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "merchant": "fraud_Rippin, Kub and Mann",
       "category": "misc_net",
       "amt": 4.97,
       "gender": "F",
       "job": "Psychologist, counselling",
       "trans_date_trans_time": "2019-01-01 00:00:18"
     }'
```

#### JSON Response Examples
- **Safe Transaction**:
  ```json
  {
      "fraud": false,
      "probability": 0.0,
      "message": "Transaction is SAFE ✅"
  }
  ```

- **Fraudulent Transaction**:
  ```json
  {
      "fraud": true,
      "probability": 0.86,
      "message": "FRAUD DETECTED! 🚨"
  }
  ```

---

## Retraining the Model

If you have fresh transaction logs or want to update the classification logic, follow these steps:

1. Place your new dataset CSV file inside `fraud_detection/dataset.csv`.
2. Run the training script:
   ```bash
   python train_model.py
   ```

The script will automatically perform feature engineering (extracting `hour`, `day`, and `month` from the timestamp), encode categorical values with `LabelEncoder`, split the data, train a `RandomForestClassifier` (using `n_jobs=-1` for multi-core parallel processing), and output the updated serialization files:
- `fraud_detection_model.pkl`
- `label_encoder.jb`

---

## Deployment

The application is structured to be production-ready and can be deployed to cloud hosting platforms like **Render** or **Heroku**.

### Render Configuration
1. Connect your GitHub repository to Render.
2. Create a new **Web Service**.
3. Choose the Python environment and configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
