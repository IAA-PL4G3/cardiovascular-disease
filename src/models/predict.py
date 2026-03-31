import joblib


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
