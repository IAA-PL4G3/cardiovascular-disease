import copy

def generate_recommendations(form_data, original_prob, predict_fn):
    """
    Simulates lifestyle changes to see how much the cardiovascular 
    disease probability drops.
    """
    recommendations = []
    
    # helper function to test a specific change
    def test_change(new_data, action_text):
        _, new_prob, _ = predict_fn(new_data)
        reduction = original_prob - new_prob
        if reduction > 0.01: 
            recommendations.append({
                "action": action_text,
                "new_prob": new_prob,
                "reduction": reduction
            })

    # stop smoking
    if form_data.get('smoke') == 1:
        data_copy = copy.deepcopy(form_data)
        data_copy['smoke'] = 0
        test_change(data_copy, "Stop smoking")

    # start physical activity
    if form_data.get('active') == 0:
        data_copy = copy.deepcopy(form_data)
        data_copy['active'] = 1
        test_change(data_copy, "Start physical activity (exercise regularly)")

    # stop consuming alcohol
    if form_data.get('alco') == 1:
        data_copy = copy.deepcopy(form_data)
        data_copy['alco'] = 0
        test_change(data_copy, "Stop consuming alcohol")

    # lose weight (Target BMI of 24.9)
    height_m = form_data['height'] / 100
    current_weight = form_data['weight']
    current_bmi = current_weight / (height_m ** 2)
    
    if current_bmi > 25.0:
        target_weight = 24.9 * (height_m ** 2)
        weight_to_lose = current_weight - target_weight
        data_copy = copy.deepcopy(form_data)
        data_copy['weight'] = target_weight
        test_change(data_copy, f"Lose {weight_to_lose:.1f} kg (Reach a healthy BMI of 24.9)")

    # lower blood pressure
    ap_hi = form_data.get('ap_hi')
    if ap_hi not in ["", "0", None] and float(ap_hi) > 120:
        data_copy = copy.deepcopy(form_data)
        data_copy['ap_hi'] = 120
        ap_lo = form_data.get('ap_lo')
        if ap_lo not in ["", "0", None] and float(ap_lo) > 80:
            data_copy['ap_lo'] = 80
        test_change(data_copy, "Lower your blood pressure to optimal levels (120/80 mmHg)")

    # sort recommendations by the biggest impact (highest reduction first)
    recommendations.sort(key=lambda x: x['reduction'], reverse=True)
    return recommendations