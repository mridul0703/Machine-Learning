# 502. k-Nearest Neighbors (k-NN)

The **k-Nearest Neighbors (k-NN)** algorithm is one of the simplest yet most powerful **non-parametric** machine learning algorithms used for both **classification** and **regression**.

It relies entirely on the idea of **similarity** — that similar points are likely to share the same outcomes.

---

## 🧩 1. Introduction

- **Type:** Supervised Learning  
- **Applications:** Classification, Regression, Recommendation Systems, Anomaly Detection  
- **Core Idea:** The output of a data point depends on the outputs of its **k nearest neighbors** in feature space.

Example:  
> To classify a new student as “pass” or “fail,” look at the performance of similar students (nearest neighbors).

---

## 🔹 2. Intuition Behind k-NN

Imagine all data points plotted in an n-dimensional space.

When a new point comes in:
1. Measure the **distance** between the new point and all training points.
2. Select the **k nearest** points.
3. Predict the **most frequent class (for classification)** or the **average (for regression)** among them.

---

## 🔹 3. The k-NN Algorithm

### Step-by-Step Procedure

1. Choose the number of neighbors **k**.  
2. Calculate the **distance** between the new sample and all training samples.  
3. Sort the distances and identify the **k closest** data points.  
4. For classification:
   - Count the class frequencies among the k neighbors.
   - Assign the majority class.
5. For regression:
   - Compute the **mean or weighted average** of neighbors’ target values.

---

## 🔹 4. Distance Metrics

Distance is crucial in k-NN. Common metrics include:

| Distance Metric | Formula | Description |
|------------------|----------|-------------|
| **Euclidean** | $( d(p,q) = \sqrt{\sum_i (p_i - q_i)^2} \)$ | Most common for continuous data |
| **Manhattan** | $( d(p,q) = \sum_i \|p_i - q_i\| \)$ | Grid-based (city block) distance |
| **Minkowski** | $( d(p,q) = (\sum_i \|p_i - q_i\|^p)^{1/p} \)$ | Generalized distance metric |
| **Hamming** | — | Used for categorical variables (0 if same, 1 if different) |
| **Cosine Similarity** | $( \frac{p \cdot q}{\|\|p\|\|\,\|\|q\|\|} \)$ | Measures orientation, not magnitude |

---

## 🔹 5. Choosing the Value of k

- **Small k (e.g., k=1)** → low bias, high variance (may overfit).  
- **Large k** → high bias, low variance (may underfit).  
- Common approach: use **cross-validation** to choose optimal k.

> Rule of thumb: start with √n (where n = number of samples) and tune using validation.

---

## 🔹 6. Weighted k-NN

Not all neighbors should have equal influence.  
We can assign **weights inversely proportional to distance**:

$$
w_i = \frac{1}{d(x, x_i)^2}
$$

Then prediction is based on **weighted majority voting (classification)** or **weighted average (regression)**.

---

## 🔹 7. Feature Scaling and Normalization

Since distance-based methods are **sensitive to feature scales**, you must:
- Normalize or standardize features.
  - **Min-Max Scaling:** $( x' = \frac{x - x_{min}}{x_{max} - x_{min}} \)$
  - **Z-score Standardization:** $( x' = \frac{x - \mu}{\sigma} \)$

Otherwise, features with larger magnitudes dominate the distance computation.

---

## 🔹 8. Advantages & Limitations

| Advantages | Limitations |
|-------------|--------------|
| Simple and intuitive | Computationally expensive for large datasets |
| No training phase (lazy learner) | Memory intensive (stores entire dataset) |
| Works well on low-dimensional data | Struggles in high-dimensional spaces (curse of dimensionality) |
| Naturally handles multi-class problems | Sensitive to noisy data and irrelevant features |

---

## 🔹 9. Handling the Curse of Dimensionality

As the number of features increases:
- Distances between points become less meaningful.
- k-NN performance deteriorates.

**Solutions:**
- Apply **Dimensionality Reduction** (PCA, t-SNE, UMAP).
- Use **feature selection** to retain only relevant variables.

---

## 🔹 10. k-NN in Regression

For regression tasks:
```math
\hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_i
```

Weighted version:
```math
\hat{y} = \frac{\sum_{i=1}^{k} w_i y_i}{\sum_{i=1}^{k} w_i}
```
where $( w_i = \frac{1}{d(x, x_i)^2} \)$.

---

## 🔹 11. Implementation with Scikit-learn

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Classification
knn_clf = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)

# Regression
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train, y_train)
y_pred_reg = knn_reg.predict(X_test)
```

---

## 🔹 12. Model Evaluation
| Task	| Common Metrics |
|-------|----------------|
| Classification	| Accuracy, Precision, Recall, F1, ROC-AUC |
| Regression	| MAE, MSE, RMSE, R² |
 
- Use cross-validation and grid search to tune k, distance metric, and weighting strategy.

--- 

## 🔹 13. Use Cases in ML
| Use Case	| Description |
|-------|----------------|
| Recommendation Systems	| Find similar users/items |
| Anomaly Detection	| Identify outliers based on distance |
| Medical Diagnosis	| Classify patients by similarity of symptoms |
| Image Recognition	| Compare pixel intensity or embeddings |
| Credit Risk Scoring	| Find clients with similar profiles |

---

## 🧠 14. Advantages Over Other Models

- Non-parametric → No assumption about data distribution.
- Flexible → Works for both classification and regression.
- Naturally supports multi-class classification.
- Interpretable → Easy to explain results to non-technical users.

---

## 🧩 15. Exercises

1. Implement k-NN on the Iris dataset for classification.
2.Visualize the decision boundary for k = 1, 5, and 15.
3. Compare Euclidean vs. Manhattan distance.
4. Apply feature scaling and observe its effect on accuracy.
5. Use cross-validation to choose optimal k.
6. Implement weighted k-NN and compare with uniform weights.
7. Apply k-NN regression on a simple nonlinear dataset (e.g., sine wave).

---
✅ **Next Topic:**  
📘 **503. Support Vector Machines (SVM)**
