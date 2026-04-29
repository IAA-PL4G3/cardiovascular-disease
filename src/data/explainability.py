import shap
import numpy as np

def generate_all_explanations(models_dict, scaled_user_data):
    """
    Generates explainability reports for all models.
    Uses SHAP for Tree models and Coefficient multiplication for Linear models.
    """
    feature_names = [
        'Gender', 'Systolic BP (ap_hi)', 'Diastolic BP (ap_lo)', 
        'Cholesterol', 'Glucose', 'Smoking', 'Alcohol', 
        'Physical Activity', 'BMI', 'Age'
    ]
    
    all_contributions = {}
    
    for model_name, model in models_dict.items():
        try:
            # tree models - SHAP
            if model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'Decision Tree']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(scaled_user_data)
                if isinstance(shap_values, list):
                    risk_shap = shap_values[1][0]
                else:
                    shap_array = np.array(shap_values)
                    if len(shap_array.shape) == 3: # 3D Array: (n_samples, n_features, n_classes)
                        risk_shap = shap_array[0, :, 1]
                    elif len(shap_array.shape) == 2: # 2D Array: (n_samples, n_features)
                        risk_shap = shap_array[0]
                    else:
                        risk_shap = shap_array
                contributions = list(zip(feature_names, risk_shap))
                
            # linear models - Coefficient * Scaled Value
            elif model_name in ['Logistic Regression', 'Linear SVM']:
                coefs = model.coef_[0]
                impacts = coefs * scaled_user_data[0]
                contributions = list(zip(feature_names, impacts))
                
            else:
                continue
            
            # sort by absolute value
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            all_contributions[model_name] = contributions
            
        except Exception as e:
            print(f"Could not explain {model_name}: {e}")
            
    return all_contributions
def print_all_reports(all_contributions):
    print("Explainability Reports")
    print("How did each model weigh your personal data to reach its conclusion?\n")
    
    for model_name, contributions in all_contributions.items():
        print(f"{model_name.upper()}:")
        for feature, impact in contributions:
            impact_pct = impact * 100
            if impact > 0:
                sign = "+"
                color_icon = "Increased Risk:"
            else:
                sign = "" 
                color_icon = "Decreased Risk:"
            if abs(impact_pct) > 0.01:
                print(f"  {color_icon:<22} {feature:<20} -> {sign}{impact_pct:.2f}%")
        print()