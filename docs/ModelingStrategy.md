# Modeling Strategy & Technical Decision Analysis: Cardiovascular Disease Prognosis

## Group Members & GitHub Link
- **GitHub Repository:** [Cardiovascular Disease Prognosis](https://github.com/IAA-PL4G3/cardiovascular-disease/)
- **[Filipe Viseu - 119192](https://github.com/FilipeNV1)**
- **[Duarte Branco - 119253](https://github.com/duartebranco)**
- **[Samuel Vinhas - 119405](https://github.com/samuelvinhas)**


## 1. Data Processing Methodology

### 1.1 Data Loading and Exploration
- **Source:** Kaggle - Cardiovascular Disease Dataset (S. Ulianova)
- **Volume:** 70,000 clinical records with 11 features
- **Features:** Demographics (age, gender), anthropometric data (height, weight), clinical measurements (blood pressure, cholesterol, glucose), and behavioral factors (smoking, physical activity)

### 1.2 Data Cleaning
Our cleaning process addressed some data quality issues:

**Blood Pressure Validation:**
- Systolic pressure (ap_hi): filtered to range [40, 370] mmHg
  - Removes physiologically impossible values (e.g., 14000 due to input errors)
- Diastolic pressure (ap_lo): filtered to range [0, 250] mmHg
  - Removes negative and unrealistic values

### 1.3 Feature Engineering
Two modeling approaches were implemented:

**Baseline Approach:** Uses original features as-is (age in days, height in cm, weight in kg)

**Feature-Engineered Approach:** 
- **BMI Calculation:** weight (kg) / (height (cm) / 100)²
  - Rationale: Combines height and weight into a clinically meaningful metric already used in medical practice
  - Simplifies the feature space and reduces multicollinearity between height and weight
- **Age Standardization:** Converts age from days to years (age_years = age / 365.25)
  - Improves interpretability and aligns with medical documentation standards

### 1.4 Data Splitting and Scaling
- **Train-Test Split:** 80/20 with stratification
  - Stratification ensures balanced class distribution in both sets
  - Prevents representation bias in model evaluation
- **Standardization:** StandardScaler normalization
  - Applied to training data first, then test data using training statistics
  - Prevents data leakage and ensures consistent scaling

---

## 2. Models Explored and Justification

### 2.1 Model Selection Strategy
Four diverse algorithms were selected to explore different modeling paradigms:

| Model | Type | Rationale | Key Hyperparameters |
|-------|------|-----------|-------------------|
| **Logistic Regression** | Linear | Interpretable baseline; provides probability outputs naturally; works well with scaled features | max_iter=1000 |
| **Naive Bayes** | Probabilistic | Computationally efficient; provides confidence scores; good for binary classification | Default (Gaussian) |
| **Decision Tree** | Tree-based | Captures non-linear relationships; feature importance insights; interpretable | max_depth=7 |
| **Linear SVM** | Kernel-based | Strong at finding decision boundaries; robust to outliers | max_iter=10000 |

### 2.2 Decision Tree Regularization
- **Applied:** max_depth=7 constraint
- **Reason:** Prevents severe overfitting while maintaining predictive power
- **Trade-off:** Balances model complexity and generalization

### 2.3 Performance Comparison

**Baseline Models (Original Features):**
```
Logistic Regression:  Accuracy: 0.7303, Precision: 0.7544, Recall: 0.6748, F1: 0.7124
Naive Bayes:          Accuracy: 0.7117, Precision: 0.7612, Recall: 0.6084, F1: 0.6763
Decision Tree:        Accuracy: 0.7317, Precision: 0.7387, Recall: 0.7083, F1: 0.7232
Linear SVM:           Accuracy: 0.7295, Precision: 0.7577, Recall: 0.6666, F1: 0.7092
```

**Feature-Engineered Models (BMI + Age Years):**
```
Logistic Regression:  Accuracy: 0.7296, Precision: 0.7550, Recall: 0.6717, F1: 0.7109
Naive Bayes:          Accuracy: 0.7118, Precision: 0.7635, Recall: 0.6052, F1: 0.6752
Decision Tree:        Accuracy: 0.7297, Precision: 0.7307, Recall: 0.7187, F1: 0.7246
Linear SVM:           Accuracy: 0.7288, Precision: 0.7588, Recall: 0.6628, F1: 0.7075
```

![Baseline vs. Feature-Engineered Performance](../output/plots/02_baseline_vs_engineered_comparison.png)

**Observation:** Feature engineering shows minimal performance differences but Decision Tree notably improves recall (0.7083 → 0.7187), suggesting better positive case detection with engineered features.

### 2.4 Threshold Optimization Analysis
Given the medical context where false negatives (missed disease cases) are more costly than false positives, we analyzed three decision thresholds:

| Threshold | Model | Recall (Sensitivity) | Precision | F1-Score | Intent |
|-----------|-------|-----|----------|----------|--------|
| **0.4** | Logistic Regression | 0.8047 | 0.6764 | 0.7350 | Maximize disease detection (high recall) |
| **0.5** | Logistic Regression | 0.6748 | 0.7544 | 0.7124 | Balanced performance |
| **0.6** | Logistic Regression | 0.5601 | 0.8043 | 0.6604 | Maximize precision (minimize false alarms) |

**Threshold 0.4:** Achieves 80.5% recall on Logistic Regression, detecting 8 out of 10 disease cases while maintaining reasonable precision (67.6%).

### 2.5 Learning Curves and Model Complexity
- **Learning Curves:** We generated learning curves for all four models to check bias-variance trade-offs and understand how each model behaves as training data increases. The shaded areas represent a +/− 1 standard deviation

#### 2.5.1 Decision Tree Learning Curves
![Decision Tree Learning Curve](../output/plots/learning_curves/decision_tree_learning_curve.png)
- Moderate overfitting: Gap between training accuracy (~0.737) and validation accuracy (~0.730)
- Convergence: Both curves are gradually converging as training size increases, with the training accuracy steadily decreasing from ~0.759 (5k samples) toward the validation plateau which indicates that the model is learning patterns rather than memorizing noise
- Final Observation: Small overfitting problem. The gap could potentially be reduced further by tightening max_depth or applying pruning, at the cost of some recall.

#### 2.5.2 Linear SVM Learning Curves
![Linear SVM Learning Curve](../output/plots/learning_curves/linear_svm_learning_curve.png)
- Saturatio: The model already reaches near-peak performance with as few as 5,000 samples, suggesting the linear decision boundary is well-defined by a small subset of data and that adding more samples does not help.
- Convergence: Training and validation curves are overlapping throughout, with both hovering around 0.725. This indicates that the model generalises as well as it trains
- Final Observation: High-bias (underfitting) problem. The relationship between features and cardiovascular disease is not fully captured by a linear boundary. A kernel SVM or non-linear model may extract more signal.
---

#### 2.5.3 Logistic Regression Learning Curves
![Logistic Regression Learning Curve](../output/plots/learning_curves/logistic_regression_learning_curve.png)
- Convergence: Like the Linear SVM, training and validation curves converge quickly and closely track each other around 0.726–0.727, reflecting very low overfitting.
- Gains: The curves flatten early (~10,000 samples), indicating the model has extracted most of the learnable signal available to a linear classifier
- Final Observation: High-bias (underfitting) problem.

#### 2.5.4 Naive Bayes Learning Curves
![Naive Bayes Learning Curve](../output/plots/learning_curves/naive_bayes_learning_curve.png)
- Convergence: Unusually, training accuracy starts above validation and then dips below it around 15,000–22,000 samples before both converge near 0.709. Features are correlated, causing the model to be "overconfident" on small training sets and then increasingly miscalibrated as it sees more data.
- Final Observation: This model is structurally limited for this task - it doesn't give the "confidence" that we would need to output probabilities to readers. The independence assumption is violated, leading to poor performance and unreliable probability estimates.

## 3. Preliminary Ethical Considerations

### 3.1 Data Bias and Fairness
**Gender Imbalance:**
- Dataset contains 65% male and 35% female patients
- **Risk:** Model may perform differently across genders due to training data skew
- **Mitigation Strategy:** 
  - Monitor performance metrics separately by gender during evaluation
  - Consider stratified analysis or reweighting during future model training
  - Flag this limitation to end-users clearly

### 3.2 Clinical Safety & False Negatives
**Medical Context Impact:**
- A false negative (incorrectly classifying disease as healthy) is substantially more dangerous than a false positive in cardiovascular diagnosis
- **Decision:** Prioritize recall during model selection and threshold tuning
- **Implementation:** 0.4 threshold chosen to maximize disease detection
- **User Disclaimer:** System should be explicitly framed as a preliminary screening tool, not diagnostic confirmation

---

## 4. Adjustments Made Since Deliverable 1 (Proposal)

### 4.1 Data Processing Enhancement
**Initial Proposal:** Removed data quality issues (impossible blood pressure values)  
**Implemented:**
- Systematic blood pressure validation rules
- Documented cleaning thresholds (ap_hi: 40-370, ap_lo: 0-250)
- Preserved data integrity while removing only clearly erroneous records

### 4.2 Threshold Analysis Development
**Initial Proposal:** Noted priority on recall over precision (false negatives more dangerous)
**Implemented:**
- Threshold analysis (0.4, 0.5, 0.6)
- Quantified recall-precision trade-offs
- Data-driven recommendation for 0.4 threshold (80.5% recall)

### 4.3 Feature Engineering Evaluation
**New Development (Not in Proposal):**
- Conducted systematic baseline vs. feature-engineered comparison
- BMI and age standardization implemented
- Found minimal overall impact but identified Decision Tree performance improvement
- Decision: Keep feature engineering for interpretability and small recall boost

### 4.4 Model Diversity Strategy
**Expansion from Proposal:**
- Current implementation adds algorithm diversity (4 models for now)
- Enables robust recommendations and cross-model validation

### 4.5 Model Selection
**Initial Proposal:** Mentioned need for adaptive models with missing feature handling  
**Current Status:** 
- We are starting to check how we can do this with kNN and Random Forest, but we have not yet implemented it. We are currently focused on the four models mentioned above to establish a strong baseline and understand the impact of feature engineering and threshold tuning before expanding to more complex models.

---