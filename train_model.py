import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train():
    # Load data
    data_path = 'd:/Fraud_detection/fraud_detection/dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    # Read only necessary columns to save memory if needed, but dataset is 350MB, should be fine.
    df = pd.read_csv(data_path)

    # Feature Engineering
    print("Preprocessing...")
    # Convert date
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month

    # Select features
    # Keeping it consistent with what we can easily get from the app
    features = ['merchant', 'category', 'amt', 'gender', 'job', 'hour', 'day', 'month']
    target = 'is_fraud'

    # Handle categorical encoding
    encoders = {}
    for col in ['merchant', 'category', 'gender', 'job']:
        le = LabelEncoder()
        # Convert to string to ensure consistency
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} classes")

    X = df[features]
    y = df[target]

    # Train model
    print("Training model (Random Forest)...")
    # Using a smaller subset for speed if needed, but let's try full first. 
    # n_jobs=-1 uses all cores.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    print(f"Test Accuracy: {score:.4f}")

    # Save
    print("Saving artifacts...")
    joblib.dump(model, 'fraud_detection_model.pkl')
    joblib.dump(encoders, 'label_encoder.jb')
    print("Done. Model and encoders saved.")

if __name__ == "__main__":
    train()
