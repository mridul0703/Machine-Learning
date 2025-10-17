# 🌲 504. Decision Trees & Random Forests

Decision Trees and Random Forests are some of the most interpretable and powerful algorithms in Machine Learning.  
They are widely used for **classification, regression, and feature importance** analysis.

---

## 🔹 1. What is a Decision Tree?

A **Decision Tree** is a tree-structured model where:
- Each **internal node** represents a decision based on a feature.
- Each **branch** represents the outcome of that decision.
- Each **leaf node** represents a predicted output.

**Example (Classification Tree):**
```yaml
       [Age > 30?]
       /        \
    Yes          No
 [Income > 50k?]   [Student?]
  /       \          /     \
Yes        No     Yes      No
```


---

## 🔹 2. Key Concepts

| Concept | Description |
|----------|-------------|
| **Feature Split** | Choosing a condition to divide the data (e.g., Age > 30) |
| **Impurity** | Measures how mixed the target values are in a node |
| **Information Gain** | Reduction in impurity after splitting |
| **Stopping Criteria** | Minimum samples per leaf, maximum depth, or pure leaf |
| **Pruning** | Removing branches that add little predictive power (reduces overfitting) |

---

## 🔹 3. Impurity Metrics

### 🧩 Classification
1. **Gini Impurity**
```math
G = 1 - \sum_{i=1}^{k} p_i^2
```
where $( p_i \)$ = probability of class *i* in a node.

2. **Entropy**
```math
H = - \sum_{i=1}^{k} p_i \log_2(p_i)
```
Used in **ID3 / C4.5 / C5.0 algorithms**.

### 🧮 Regression
**Variance Reduction**  
A split is chosen to minimize the variance of the target variable in child nodes.

---

## 🔹 4. Building a Decision Tree

1. Start with the full dataset.
2. Choose the best feature and threshold that gives the **highest information gain**.
3. Split the dataset.
4. Repeat recursively until stopping conditions are met.
5. Optionally prune the tree to reduce overfitting.

---

## 🔹 5. Implementation of Decision Tree (Scikit-learn)

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# --- Classification ---
dt_clf = DecisionTreeClassifier(
    criterion='gini',      # or 'entropy'
    max_depth=None,        # can limit depth for regularization
    random_state=42
)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)

# --- Regression ---
dt_reg = DecisionTreeRegressor(
    criterion='squared_error',  # or 'friedman_mse', 'absolute_error'
    max_depth=None,
    random_state=42
)
dt_reg.fit(X_train, y_train)
y_pred_reg = dt_reg.predict(X_test)
```

---

## 🔹 6. Advantages & Limitations

| Advantages | Limitations |
|-------------|-------------|
| Easy to interpret and visualize | Prone to overfitting |
| Handles numerical & categorical data | Unstable (small data changes → big tree change) |
| No feature scaling required | Greedy algorithm → not globally optimal |
| Works well with missing values | Can create biased splits if classes are imbalanced |

---

## 🌲 7. Random Forests

A **Random Forest** is an ensemble of multiple Decision Trees trained on:
- **Bootstrap samples** (random subsets of data)
- **Random subsets of features** at each split

Final prediction:
- Classification → majority voting
- Regression → average of predictions

**Formula:**
```math
\hat{y} = \frac{1}{N} \sum_{i=1}^{N} f_i(x)
```
where $( f_i \)$ = prediction from tree *i*.

---

## 🔹 8. Implementation of Random Forest (Scikit-learn)

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# --- Classification ---
rf_clf = RandomForestClassifier(
    n_estimators=100,     # number of trees
    criterion='gini',     # or 'entropy'
    max_depth=None,
    random_state=42
)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

# --- Regression ---
rf_reg = RandomForestRegressor(
    n_estimators=100,
    criterion='squared_error',  # or 'absolute_error'
    max_depth=None,
    random_state=42
)
rf_reg.fit(X_train, y_train)
y_pred_reg = rf_reg.predict(X_test)
```

---


## 🔹 7. Why Random Forests Work

| Technique | Purpose |
|------------|----------|
| **Bagging (Bootstrap Aggregation)** | Reduces variance and overfitting |
| **Feature Randomness** | De-correlates trees to improve generalization |
| **Averaging** | Stabilizes predictions |

---

## 🔹 8. Hyperparameters

| Parameter | Description |
|------------|-------------|
| `n_estimators` | Number of trees in the forest |
| `max_depth` | Maximum depth of each tree |
| `min_samples_split` | Minimum samples to split a node |
| `min_samples_leaf` | Minimum samples per leaf |
| `max_features` | Number of features considered at each split |
| `bootstrap` | Whether to use bootstrapping samples |

---

## 🔹 9. Feature Importance

Random Forests naturally measure **feature importance**:
- Based on decrease in impurity (Gini or Entropy)
- Or permutation importance (change in model accuracy when a feature is shuffled)

This helps in **feature selection** and **model interpretability**.

---

## 🔹 10. Use Cases

| Use Case | Description |
|-----------|-------------|
| Credit scoring | Predicting loan defaults |
| Medical diagnosis | Disease prediction |
| Customer churn | Identify customers likely to leave |
| Stock market | Trend prediction |
| NLP / CV | Feature-based classification tasks |

---

## 🧾 11. Comparison: Decision Tree vs Random Forest

| Aspect | Decision Tree | Random Forest |
|---------|----------------|---------------|
| Model Type | Single tree | Ensemble of trees |
| Bias | Low | Slightly higher |
| Variance | High | Low |
| Interpretability | Excellent | Moderate |
| Overfitting | Common | Reduced |
| Accuracy | Moderate | High |

---

## 🧮 12. Evaluation Metrics

| Task | Metric |
|------|--------|
| Classification | Accuracy, Precision, Recall, F1, ROC-AUC |
| Regression | MSE, RMSE, MAE, R² |

---

## 🧩 13. Exercises

1. Build a Decision Tree for classification using Scikit-learn.  
2. Visualize the tree and interpret each split.  
3. Train a Random Forest and compare its accuracy with the single Decision Tree.  
4. Analyze the top 5 most important features.  
5. Tune `n_estimators` and `max_depth` to observe bias-variance changes.

---

### ✅ Next Topic:
📘 505. Gradient Boosting (XGBoost, LightGBM, CatBoost)


