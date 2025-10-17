# 501. Logistic Regression — Classification Basics

**Logistic Regression** is one of the most fundamental and widely used algorithms in Machine Learning for **classification tasks**.  
Despite its name, it is a **classification** algorithm, not regression.  
It estimates **probabilities** using the **logistic (sigmoid) function**, making it suitable for predicting **binary outcomes** (0 or 1).

---

## 🧩 1. Introduction

- **Goal:** Predict a **categorical outcome** (e.g., Yes/No, Spam/Not Spam, Disease/No Disease).  
- **Type:** Supervised Learning → Classification.  
- **Output:** Probability value between 0 and 1.

**Examples:**
- Will a customer churn? (Yes/No)  
- Is an email spam?  
- Will a patient develop diabetes?

---

## 🔹 2. Why Not Linear Regression?

If we use **Linear Regression** for classification:
- Predictions may go beyond the [0,1] range.
- Model is not robust for categorical outcomes.
- Errors are not normally distributed.

Hence, we use **Logistic Regression**, which models **probabilities** directly through the **sigmoid function**.

---

## 🔹 3. The Logistic (Sigmoid) Function

The **sigmoid function** maps any real value into the range (0, 1):

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where  
```math
z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n
```

### 🧠 Intuition:
- When $( z \)$ → ∞, $( \sigma(z) \)$ → 1  
- When $( z \)$ → -∞, $( \sigma(z) \)$ → 0  
- So, it models **probabilities** smoothly between 0 and 1.

---

## 🔹 4. The Logistic Regression Model

The logistic regression equation is:

$$
P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}
$$

We can rearrange this as:

$$
\text{logit}(P) = \log\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n
$$

Here:
- $( P \)$ = probability that the instance belongs to class 1.  
- $( \frac{P}{1-P} \)$ = **odds** of being in class 1.  
- **logit(P)** = natural log of odds → gives a **linear relationship**.

---

## 🔹 5. Decision Boundary

To classify:
```math
\hat{y} =
\begin{cases}
1, & \text{if } P(y=1|X) \geq 0.5 \\
0, & \text{otherwise}
\end{cases}
```

You can change the **threshold (0.5)** to make the model more sensitive or specific depending on use case.

---

## 🔹 6. Model Training — Maximum Likelihood Estimation (MLE)

Unlike Linear Regression (which uses least squares),  
**Logistic Regression uses Maximum Likelihood Estimation** to find coefficients $(\(\beta\))$ that maximize the probability of observing the data.

Objective:
```math
\text{maximize } L(\beta) = \prod_{i=1}^{n} P(y_i|x_i)
```

Or in log form:
```math
\text{maximize } \log L(\beta) = \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]
```

---

## 🔹 7. Evaluation Metrics for Classification

| Metric | Formula | Interpretation |
|--------|----------|----------------|
| **Accuracy** | $( \frac{TP + TN}{TP + TN + FP + FN} \)$ | Overall correctness |
| **Precision** | $( \frac{TP}{TP + FP} \)$ | How many predicted positives are real |
| **Recall (Sensitivity)** | $( \frac{TP}{TP + FN} \)$ | How many actual positives are detected |
| **F1 Score** | $( 2 \times \frac{Precision \times Recall}{Precision + Recall} \)$ | Harmonic mean of Precision & Recall |
| **ROC-AUC** | Area under the ROC curve | Measures separability between classes |

---

## 🔹 8. Assumptions of Logistic Regression

1. **Linearity in log-odds:** The logit (log-odds) is a linear combination of inputs.  
2. **Independence:** Observations are independent.  
3. **No multicollinearity:** Predictors should not be highly correlated.  
4. **Large sample size:** Ensures stability of estimates.

---

## 🔹 9. Regularization in Logistic Regression

To avoid overfitting:
- **L1 (Lasso)** regularization → feature selection.
- **L2 (Ridge)** regularization → coefficient shrinkage.

Scikit-learn implementation:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', solver='liblinear')
```
---

---

## 🔹 10. Interpretation of Coefficients

Each coefficient $( \beta_i \)$:

- Represents the change in **log-odds** for a unit increase in $( x_i \)$.
- $( e^{\beta_i} \)$ gives the **odds ratio** — how much the odds of the event increase or decrease.

**Example:**

If $( \beta_1 = 0.7 \)$, then $( e^{0.7} \approx 2.01 \)$.  
→ Each unit increase in $( x_1 \)$ **doubles the odds of success**.

---

## 🔹 11. Limitations

- Assumes **linearity in log-odds** space.  
- Not suitable for **non-linear decision boundaries**.  
- Can **underperform** with highly correlated or irrelevant features.  
- Requires **feature scaling** for optimal convergence.

---

## 🧠 12. Practical Applications

| Application | Description |
|--------------|-------------|
| **Medical diagnosis** | Predicting presence/absence of disease |
| **Credit scoring** | Estimating probability of loan default |
| **Spam filtering** | Classifying emails as spam or not spam |
| **Customer churn** | Predicting whether a customer will leave |
| **Marketing** | Estimating likelihood of conversion or purchase |

---

## 🧩 13. Exercises

1. Train a **Logistic Regression** model on a binary classification dataset (e.g., Titanic, Breast Cancer).  
2. Visualize the **sigmoid curve** and explain the **0.5 threshold**.  
3. Compute **Precision**, **Recall**, and **F1 Score** for your model.  
4. Experiment with **regularization** (`penalty='l1'` vs `'l2'`).  
5. Change the **classification threshold** and observe changes in precision/recall.  
6. Interpret model coefficients in terms of **odds ratios**.

---

✅ **Next Topic:**  
📘 **502. k-Nearest Neighbors (k-NN)**
