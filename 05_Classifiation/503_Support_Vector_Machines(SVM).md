# 503. Support Vector Machines (SVM)

Support Vector Machines (SVM) are powerful **supervised learning algorithms** used for both **classification** and **regression**.  
They are particularly effective in **high-dimensional spaces** and when the number of features exceeds the number of samples.

---

## 🧩 1. Intuition Behind SVM

The idea of SVM is to find the **optimal hyperplane** that separates data points of different classes with the **maximum margin**.

### Example (Binary Classification):

For two classes (red and blue):
- There can be many possible lines (in 2D) or planes (in higher dimensions) that separate them.
- SVM chooses the one that **maximizes the distance** between the nearest points of each class — called the **support vectors**.

### Intuitive Goal:
> “Find the widest possible street (margin) that separates the classes.”

---

## 🔹 2. Key Terminologies

| Term | Description |
|------|--------------|
| **Hyperplane** | Decision boundary that separates classes |
| **Support Vectors** | Data points closest to the hyperplane; critical for defining margin |
| **Margin** | Distance between hyperplane and nearest data points (support vectors) |
| **Optimal Hyperplane** | The one with the **maximum margin** between classes |

---

## 🔹 3. Mathematical Formulation

For binary classification with labels $( y_i \in \{-1, +1\} \)$ :

We want to find a hyperplane:
```math
w^T x + b = 0
```

### Objective:
Maximize margin → minimize $( ||w||^2 \)$

### Subject to constraints:
$$
y_i (w^T x_i + b) \geq 1, \quad \forall i
$$

This is a **convex optimization** problem — guaranteeing a unique global minimum.

---

## 🔹 4. Soft Margin SVM

Real-world data is rarely perfectly separable.  
SVM introduces **slack variables** $( \xi_i \)$ to allow some misclassifications.

$$
\min_{w,b} \frac{1}{2} ||w||^2 + C \sum_i \xi_i
$$
$$
\text{subject to } y_i (w^T x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0
$$

Here, **C** is a **regularization parameter**:
- Large C → Less tolerance to errors (hard margin)
- Small C → More tolerance (soft margin)

---

## 🔹 5. Kernel Trick (Non-linear SVM)

When data isn’t linearly separable, SVM uses the **kernel trick** to project data into a **higher-dimensional space** where a linear separator can exist.

$$
K(x_i, x_j) = \phi(x_i)^T \phi(x_j)
$$

Common Kernels:

| Kernel | Formula | Usage |
|---------|----------|--------|
| **Linear** | $( K(x, y) = x^T y \)$ | When data is linearly separable |
| **Polynomial** | $( K(x, y) = (x^T y + c)^d \)$ | For curved boundaries |
| **RBF (Gaussian)** | $( K(x, y) = \exp(-\gamma \|\|x - y\|\|^2) \)$ | Most popular for non-linear problems |
| **Sigmoid** | $( K(x, y) = \tanh(\alpha x^T y + c) \)$ | Similar to neural networks |

---

## 🔹 6. Decision Function

After training, predictions are made using:

$$
f(x) = \text{sign}(w^T x + b)
$$

For kernelized versions:

$$
f(x) = \text{sign}\left(\sum_i \alpha_i y_i K(x_i, x) + b\right)
$$

where $( \alpha_i \)$ are **Lagrange multipliers** (non-zero only for support vectors).

---

## 🔹 7. Hyperparameters

| Parameter | Meaning | Effect |
|------------|----------|--------|
| **C** | Regularization strength | High C → less regularization, may overfit |
| **Kernel** | Type of transformation | Choose `linear`, `poly`, `rbf`, `sigmoid` |
| **gamma (γ)** | Kernel coefficient for RBF/poly | Controls influence of single training example |
| **degree** | Degree of polynomial kernel | Used only with `poly` kernel |

---

## 🔹 8. SVM for Regression (SVR)

Support Vector Regression (SVR) uses a similar concept but tries to fit the data within a **margin of tolerance (ε)**.

### Objective:
$$
\min_{w,b} \frac{1}{2} ||w||^2
$$
subject to:
$$
|y_i - (w^T x_i + b)| \leq \varepsilon
$$

- Points outside the ε-tube contribute to the loss function.
- Controlled by **C** (regularization) and **ε** (margin width).

---

## 🔹 9. Advantages & Limitations

| Advantages | Limitations |
|-------------|--------------|
| Works well in high-dimensional spaces | Computationally expensive for large datasets |
| Effective with clear margin of separation | Requires careful parameter tuning (C, γ) |
| Robust against overfitting (with proper kernel) | Not suitable for very large or noisy datasets |
| Can model non-linear boundaries | Hard to interpret compared to simple models |

---

## 🔹 10. Implementation (Scikit-learn)

```python
from sklearn.svm import SVC, SVR

# Classification
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)

# Regression
svm_reg = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_reg.fit(X_train, y_train)
y_pred_reg = svm_reg.predict(X_test)
Support Vector Machines (SVM)
```

---

## 🔹 11. Model Evaluation
| Task	| Metrics | 
|-------|---------|
| Classification	| Accuracy, Precision, Recall, F1-score, ROC-AUC | 
| Regression	| MAE, RMSE, R² | 

- Use GridSearchCV or RandomizedSearchCV to tune parameters like C, gamma, and kernel.

---

## 🔹 12. Visualization Example (2D)

Imagine data with two classes (red & blue):

- Class +1: o o o
- Class -1: x x x

The SVM hyperplane lies between them, maximizing the gap between support vectors on both sides.

---

## 🔹 13. Practical Tips

-  Always scale your features before training SVM (especially for RBF and polynomial kernels).
-  Use linear kernel for large sparse datasets (e.g., text classification).
-  Use RBF kernel as the default for non-linear data.
-  Tune C and gamma via cross-validation.

---

## 🔹 14. Real-World Applications
| Domain |	Application |
|-------------|--------------|
| Text | Classification	Spam detection, sentiment analysis | 
| Image Recognition	| Face recognition, object detection |
| Bioinformatics	| Cancer classification, gene expression |
| Finance	| Credit risk assessment, fraud detection |
| Industrial Systems	| Fault diagnosis, predictive maintenance |

---

## 🧩 15. Exercises

1. Train an SVM classifier on the Iris dataset with linear and RBF kernels.
2. Visualize decision boundaries for different kernels.
3. Perform hyperparameter tuning using GridSearchCV for C and gamma.
4. Compare training times for linear vs. RBF kernels.
5. Implement SVR on a simple regression dataset (e.g., Boston Housing).
6. Analyze how changing C and ε affects the SVR fit.

---

### ✅ Next Topic:  
📘 *Decision Trees & Random Forests*
