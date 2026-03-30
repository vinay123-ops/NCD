import os
import kagglehub
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def main():
    print("--- [1/5] Downloading Dataset ---")
    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
    csv_file_path = os.path.join(path, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

    print("--- [2/5] Processing Data ---")
    df = pd.read_csv(csv_file_path)
    
    # Define Targets and Features
    target_columns = ['Diabetes_binary', 'HighBP', 'HeartDiseaseorAttack']
    X = df.drop(target_columns, axis=1)
    y = df[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling - IMPORTANT: We save this scaler for the API!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("--- [3/5] Building Neural Network ---")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='sigmoid') # 3 outputs for 3 diseases
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("--- [4/5] Training (20 Epochs) ---")
    model.fit(X_train_scaled, y_train, epochs=20, batch_size=128, validation_split=0.2, verbose=1)

    print("--- [5/5] Saving Assets ---")
    # Save the Model
    model.save('ncd_multilabel_risk_model.h5')
    # Save the Scaler (The API needs this!)
    joblib.dump(scaler, 'scaler.pkl')
    
    print("\nSUCCESS: 'ncd_multilabel_risk_model.h5' and 'scaler.pkl' are ready.")

if __name__ == "__main__":
    main()