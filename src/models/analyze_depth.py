# Analyze depth in decision tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def analyze_decision_tree_depth(X, y, max_depth_range=range(1, 21)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    train_accuracies = []
    test_accuracies = []
    
    for depth in max_depth_range:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    # print best accuracies and corresponding depths
    best_train_acc = max(train_accuracies)
    best_test_acc = max(test_accuracies)
    best_train_depth = max_depth_range[train_accuracies.index(best_train_acc)]
    best_test_depth = max_depth_range[test_accuracies.index(best_test_acc)]
    print(f"Best Train Accuracy: {best_train_acc:.4f} at depth {best_train_depth}")
    print(f"Best Test Accuracy: {best_test_acc:.4f} at depth {best_test_depth}")
    plt.figure(figsize=(10, 6))
    plt.plot(max_depth_range, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(max_depth_range, test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Max Depth of Decision Tree')
    plt.ylabel('Accuracy')
    plt.title('Effect of Max Depth on Decision Tree Performance')
    plt.xticks(max_depth_range)
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # RESULT:
    # Best Train Accuracy: 0.8818 at depth 20
    # Best Test Accuracy: 0.7325 at depth 5
    df = pd.read_csv("../../data/processed/cardio_train_cleaned.csv", sep=",")
    y = df['cardio']
    X = df.drop(columns=['cardio'])
    analyze_decision_tree_depth(X, y)

if __name__ == "__main__":
    main()