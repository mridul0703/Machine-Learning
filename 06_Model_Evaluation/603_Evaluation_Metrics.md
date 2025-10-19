# 603. Evaluation Metrics in Machine Learning

Evaluation metrics are crucial to measure how well a machine learning model performs.  
They help us **quantify accuracy, reliability, and robustness** of predictions — allowing us to compare models and fine-tune them effectively.

---

## 🧩 1. Introduction

A machine learning model should not only fit the training data but also **generalize** to unseen data.  
Evaluation metrics provide the tools to **assess performance** for both regression and classification tasks.

| Task Type | Example Algorithms | Common Metrics |
|------------|--------------------|----------------|
| **Regression** | Linear Regression, Random Forest Regressor | MSE, RMSE, MAE, R² |
| **Classification** | Logistic Regression, SVM, Decision Tree | Accuracy, Precision, Recall, F1-score, AUC |

---

## 🔹 2. Regression Metrics

Regression models predict **continuous** values (like house prices, temperature, etc.).

### 🔸 2.1 Mean Squared Error (MSE)

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

- Measures average squared difference between actual and predicted values.  
- Penalizes large errors heavily.

✅ **Lower MSE = Better fit**

---

### 🔸 2.2 Root Mean Squared Error (RMSE)

\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

- Square root of MSE — interpretable in same units as target variable.
- Sensitive to outliers.

✅ **Useful when large errors are undesirable**

---

### 🔸 2.3 Mean Absolute Error (MAE)

\[
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

- Average of absolute differences.
- Robust against outliers.

✅ **Useful for real-world interpretability (e.g., “average error = ₹5000”)**

---

### 🔸 2.4 R² Score (Coefficient of Determination)

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

- Measures how much variance in the target is explained by the model.
- Range: 0 → 1 (sometimes negative if model performs poorly)

✅ **Higher R² = Better explanatory power**

---

| Metric | Range | Interpretation |
|---------|--------|----------------|
| MSE | [0, ∞) | 0 = perfect fit |
| RMSE | [0, ∞) | Lower = better |
| MAE | [0, ∞) | Lower = better |
| R² | (-∞, 1] | Closer to 1 = better model |

---

## 🔹 3. Classification Metrics

Classification models predict **discrete classes** (e.g., spam/not spam, disease/healthy).

---

### 🔸 3.1 Confusion Matrix

|               | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

Helps visualize model performance across categories.

---

### 🔸 3.2 Accuracy

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

✅ **Good when classes are balanced.**

⚠️ **Not reliable for imbalanced datasets.**

---

### 🔸 3.3 Precision

\[
Precision = \frac{TP}{TP + FP}
\]

- Of all predicted positives, how many are actually positive?
- **High Precision** → few false positives.

✅ Used in applications like spam detection (minimize false alarms).

---

### 🔸 3.4 Recall (Sensitivity)

\[
Recall = \frac{TP}{TP + FN}
\]

- Of all actual positives, how many are correctly predicted?
- **High Recall** → fewer false negatives.

✅ Used in applications like medical diagnosis (don’t miss true cases).

---

### 🔸 3.5 F1-Score

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

- Harmonic mean of Precision and Recall.
- Balances between false positives & false negatives.

✅ **Useful when both Precision and Recall matter.**

---

### 🔸 3.6 ROC Curve & AUC

- **ROC (Receiver Operating Characteristic)** → plots True Positive Rate vs False Positive Rate.
- **AUC (Area Under Curve)** → measures the area under ROC curve.

| Metric | Meaning |
|---------|----------|
| AUC = 1 | Perfect model |
| AUC = 0.5 | Random model |
| AUC < 0.5 | Worse than random |

✅ **Higher AUC = Better discrimination between classes**

---

### 🔸 3.7 Log Loss (Cross-Entropy)

\[
LogLoss = -\frac{1}{n} \sum [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
\]

- Penalizes incorrect confident predictions more.
- Used for probabilistic classifiers like Logistic Regression.

---

## 🔹 4. Regression vs. Classification Metrics Summary

| Type | Metric | Ideal Range | Best When |
|------|--------|--------------|------------|
| Regression | MSE / RMSE / MAE / R² | ↓ / ↑ | Continuous prediction problems |
| Classification | Accuracy, Precision, Recall, F1, AUC | ↑ | Discrete label problems |

---

## 🔹 5. Advanced Evaluation Concepts

### 5.1 Weighted Metrics
Used when classes are imbalanced.  
Weighted average gives more importance to dominant classes.

### 5.2 Macro & Micro Averaging
- **Macro:** Average metric for each class (treat all classes equally)  
- **Micro:** Compute metrics globally (weighted by class size)

### 5.3 Cross-Validation Scores
Metrics can be averaged across folds to assess model consistency.

---

## 🔹 6. Model Comparison

When comparing models:
- Always use **the same metric** across models.
- Consider **context**: For rare diseases, recall is more critical than accuracy.
- Use **visual tools**: ROC, Precision-Recall curve, residual plots.

---

## 🧾 7. Summary

| Concept | Description |
|----------|--------------|
| **Regression Metrics** | MSE, RMSE, MAE, R² |
| **Classification Metrics** | Accuracy, Precision, Recall, F1, AUC |
| **Confusion Matrix** | Helps visualize model performance |
| **Goal** | Choose metrics aligned with business or research objectives |

---

## 🧩 8. Exercises

1. Compute MSE, MAE, and R² for a regression model on sample data.  
2. Create a confusion matrix and derive Precision, Recall, F1-score.  
3. Plot ROC and Precision–Recall curves for a binary classifier.  
4. Discuss when to prefer F1-score over accuracy.  
5. Compare AUC for two models and interpret results.

---

✅ **Next Topic:**  
📘 604. Data Preprocessing and Feature Engineering
