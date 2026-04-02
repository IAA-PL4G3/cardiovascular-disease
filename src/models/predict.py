import joblib
import sys
import numpy as np
import pandas as pd

def load_model(filepath: str):
    return joblib.load(filepath)

if __name__ == "__main__":
    sys.path.insert(0, "../../")
    from src.features.build_features import clean_data, split_and_scale
    
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
        
    except FileNotFoundError:
        print("Model not found. Train a model first using train.py")
