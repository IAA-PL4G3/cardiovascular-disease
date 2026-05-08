import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


def roc_auc_evaluation(model, X_test, y_test, model_name="Model"):
    """
    Returns ROC values and AUC score
    """
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise ValueError("Model does not have predict_proba or decision_function method")
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    auc_score = roc_auc_score(y_test, y_score)
    print(f"{model_name} AUC Score: {auc_score:.4f}")
    return fpr, tpr, auc_score


def plot_roc_curve(fpr, tpr, auc_score, model_name="Model"):
    """
    Plot ROC curve for one model
    """
    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc_score:.3f})"
    )

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models on one ROC plot

    models_dict format:
    {
        "Logistic Regression": model1,
        "Random Forest": model2
    }
    """
    plt.figure(figsize=(8, 6))

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            raise ValueError("Model does not have predict_proba or decision_function method")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_score:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.show()