from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(x_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model


def train_naive_bayes(x_train, y_train):
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model


def train_decision_tree(x_train, y_train, max_depth=7):
    # max_depth added to prevent severe overfitting
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)
    return model


def train_linear_svm(x_train, y_train):
    model = LinearSVC(max_iter=10000, random_state=42)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    return {
        "accuracy": acc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
    }
