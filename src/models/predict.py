import joblib
import sys
import numpy as np
import pandas as pd
from src.features.build_features import clean_data, split_and_scale

def load_model(filepath: str):
    return joblib.load(filepath)

def predict_single_instance(model, scaler, input_data):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_data)[0][1]
    else:
        probability = None

    return prediction[0], probability

if __name__ == "__main__":
    sys.path.insert(0, "../../")
    
    # load and process data
    df = pd.read_csv("../../data/raw/cardio_train.csv", sep=";")
    df_cleaned = clean_data(df)
    x_train, x_test, y_train, y_test, scaler = split_and_scale(df_cleaned)
    
    try:
        model = load_model("../../models/logistic_regression_model.pkl")    
        print("Making predictions on test set...\n")
        test_predictions = model.predict(x_test)
        print(f"Predictions shape: {test_predictions.shape}")
        print(f"Unique predictions: {np.unique(test_predictions)}")
        print(f"Average prediction confidence: {model.predict_proba(x_test)[:, 1].mean():.4f}")
        sample_input = x_test[:1]
        pred, prob = predict_single_instance(model, scaler, x_test[:1])
        print(f"\nSample prediction: {pred}, Confidence: {prob:.4f}")
        
    except FileNotFoundError:
        print("Model not found. Train a model first using train.py")
