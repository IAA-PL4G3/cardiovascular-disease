import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from scipy.special import expit
from recommendations import generate_recommendations
from explainability import generate_all_explanations, print_all_reports

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", module="shap")

# load previously saved models
imputer = joblib.load('../../models/knn_imputer.pkl')
scaler = joblib.load('../../models/scaler_with_feature_engineering.pkl')
models = {}
models['Logistic Regression'] = joblib.load('../../models/logistic_regression_with_feature_engineering.pkl')
models['Decision Tree'] = joblib.load('../../models/decision_tree_with_feature_engineering.pkl')
models['Linear SVM'] = joblib.load('../../models/linear_svm_with_feature_engineering.pkl')
models['Random Forest'] = joblib.load('../../models/random_forest_with_feature_engineering.pkl')
models['XGBoost'] = joblib.load('../../models/xgboost_with_feature_engineering.pkl')
models['LightGBM'] = joblib.load('../../models/lightgbm_with_feature_engineering.pkl')

def process_and_predict(form_data):
    """
    Process user input and predict the probability of cardiovascular disease.

    Expected form_data dictionary keys:
    age_years, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
    """
    # mandatory fields
    mandatory_fields = ['age_years', 'gender', 'height', 'weight', 'smoke', 'alco', 'active']
    for field in mandatory_fields:
        if form_data.get(field) is None or form_data.get(field) == "":
            raise ValueError(f"Field '{field}' is mandatory!")

    # calculate BMI
    weight = form_data['weight']
    height_m = form_data['height'] / 100
    bmi = weight / (height_m ** 2)

    # handle optional fields (replace 0 or empty with np.nan)
    def handle_optional(value):
        if value in [0, "0", None, "", "Unknown"]:
            return np.nan
        return float(value)

    ap_hi = handle_optional(form_data.get('ap_hi'))
    ap_lo = handle_optional(form_data.get('ap_lo'))
    cholesterol = handle_optional(form_data.get('cholesterol'))
    gluc = handle_optional(form_data.get('gluc'))

    # create a DataFrame matching the feature-engineered training structure
    user_df = pd.DataFrame([{
        'gender': form_data['gender'],
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': form_data['smoke'],
        'alco': form_data['alco'],
        'active': form_data['active'],
        'bmi': bmi,
        'age_years': form_data['age_years']
    }])

    # apply KNN Imputer to fill missing values
    imputed_data_array = imputer.transform(user_df)
    imputed_data_array[0, 3] = round(imputed_data_array[0, 3]) # cholesterol
    imputed_data_array[0, 4] = round(imputed_data_array[0, 4]) # gluc

    # scale the data
    scaled_data = scaler.transform(imputed_data_array)

    # make the prediction
    probs = {}
    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled_data)[0][1]
        else:
            score = model.decision_function(scaled_data)[0]
            probability = expit(score)
        probs[model_name] = probability

    probability = np.mean(list(probs.values()))
    return probs, probability, imputed_data_array[0]

if __name__ == "__main__":
    # ask user for input
    form_data = {}
    form_data['age_years'] = int(input("Enter age in years: "))
    form_data['gender'] = int(input("Enter gender (1 for male, 2 for female): "))
    form_data['height'] = float(input("Enter height in cm: "))
    form_data['weight'] = float(input("Enter weight in kg: "))
    form_data['ap_hi'] = input("Enter systolic blood pressure (or leave blank if unknown): ")
    form_data['ap_lo'] = input("Enter diastolic blood pressure (or leave blank if unknown): ")
    form_data['cholesterol'] = input("Enter cholesterol level (1: normal, 2: above normal, 3: well above normal, or leave blank if unknown): ")
    form_data['gluc'] = input("Enter glucose level (1: normal, 2: above normal, 3: well above normal, or leave blank if unknown): ")
    form_data['smoke'] = int(input("Do you smoke? (0 for no, 1 for yes): "))
    form_data['alco'] = int(input("Do you consume alcohol? (0 for no, 1 for yes): "))
    form_data['active'] = int(input("Are you physically active? (0 for no, 1 for yes): "))

    # process and predict
    try:
        probs, probability, imputed_data_array = process_and_predict(form_data)
        if form_data['ap_hi'] in ["", "0", None]:
            print(f"[*] Estimated Systolic BP (ap_hi): {imputed_data_array[1]:.1f}")
        if form_data['ap_lo'] in ["", "0", None]:
            print(f"[*] Estimated Diastolic BP (ap_lo): {imputed_data_array[2]:.1f}")
        if form_data['cholesterol'] in ["", "0", None]:
            print(f"[*] Estimated Cholesterol Level: {round(imputed_data_array[3])} (1: normal, 2: above normal, 3: well above normal)")
        if form_data['gluc'] in ["", "0", None]:
            print(f"[*] Estimated Glucose Level: {round(imputed_data_array[4])} (1: normal, 2: above normal, 3: well above normal)")
        print(f"Predicted probability of cardiovascular disease: {probability:.4f} ({probability * 100:.1f}%)")
        print(f"Individual model probabilities")
        for model_name, prob in probs.items():
            print(f"{model_name:<20}: {prob:.4f} ({prob * 100:.1f}%)")

        user_input_suggestions = input(f"Want to check suggestions to reduce your risk? (Y/n): ")
        if user_input_suggestions.lower() != 'n':
            suggestions = generate_recommendations(form_data, probability, process_and_predict)
            if not suggestions:
                print("No suggestions available")
            else:
                for i, rec in enumerate(suggestions, 1):
                    action = rec['action']
                    reduction = rec['reduction'] * 100
                    new_prob = rec['new_prob'] * 100
                    print(f"{i}. {action}")
                    print(f"   -> This would reduce your risk by {reduction:.1f}% (New Probability: {new_prob:.1f}%)")

        user_input_explain = input(f"Want to see how each model weighed your data to reach its conclusion? (Y/n): ")
        if user_input_explain.lower() != 'n':
            scaled_data = scaler.transform([imputed_data_array])
            all_contributions = generate_all_explanations(models, scaled_data)
            print_all_reports(all_contributions)
    except ValueError as e:
        print(f"Error: {e}")
