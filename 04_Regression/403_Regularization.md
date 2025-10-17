# 403. Regularization: Ridge, Lasso, ElasticNet

**Regularization** is a technique to prevent **overfitting** in Machine Learning models by adding a penalty to the loss function.  
It constrains the model complexity, especially when dealing with high-dimensional or multicollinear data.

---

## 🧩 1. Why Regularization?

- Linear/Polynomial Regression can **overfit** when:
  - Features are highly correlated
  - Number of features is large
  - Model degree is high
- Regularization adds a **penalty term** to shrink coefficients and reduce variance.

---

## 🔹 2. Ridge Regression (L2 Regularization)

### Loss Function:
$$
J(\beta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p \beta_j^2
$$

Where:
- $( \lambda \)$= regularization strength
- Larger $( \lambda \)$ → more shrinkage of coefficients  
- No coefficients are set exactly to zero.

### Characteristics:
- Good for **multicollinearity**
- Keeps all features in the model
- Shrinks coefficients evenly

---

## 🔹 3. Lasso Regression (L1 Regularization)

### Loss Function:
$$
J(\beta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j|
$$

### Characteristics:
- Can shrink some coefficients **exactly to zero** → **feature selection**
- Useful for sparse models
- Sensitive to correlated features (may select one and ignore others)

---

## 🔹 4. ElasticNet Regression

### Loss Function:
$$
J(\beta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^p |\beta_j| + \lambda_2 \sum_{j=1}^p \beta_j^2
$$

### Characteristics:
- Combines **L1 + L2 penalties**
- Handles **correlated features better** than Lasso
- Balances feature selection and coefficient shrinkage

---

## 🔹 5. Choosing Hyperparameters

- $( \lambda \)$ $(alpha)$ controls strength of regularization
- Use **cross-validation** to select optimal $( \lambda \)$
- ElasticNet uses **mixing parameter** $( \rho \)$ to balance L1/L2:
  - $( \rho = 1 \)$ → Lasso  
  - $( \rho = 0 \)$ → Ridge

---

## 🔹 6. Implementation (Scikit-learn)

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# ElasticNet
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
```

---
## 🔹 7. Use Cases in ML

| Use Case | Description |
|-----------|--------------|
| **High-dimensional data** | Ridge prevents overfitting when the number of features (p) is much larger than the number of samples (n). |
| **Sparse feature selection** | Lasso performs feature selection by setting less important coefficients exactly to zero. |
| **Correlated features** | ElasticNet handles correlated predictors better by combining L1 and L2 regularization. |
| **Polynomial regression** | Regularization helps prevent overfitting when using high-degree polynomial terms. |
| **Text and NLP tasks** | Lasso is used for selecting the most relevant words/features in sparse text data. |
| **Genomics / Bioinformatics** | Ridge and ElasticNet are used where datasets have thousands of correlated variables. |
| **Finance and Econometrics** | Lasso helps select key economic indicators from large feature sets. |

---

## 🧩 8. Exercises

1. **Fit Ridge and Lasso Regression:**  
   Train Ridge and Lasso regression models on a dataset and compare the learned coefficients.  
   - Observe which features are shrunk to zero (Lasso) and which are only reduced (Ridge).

2. **Cross-validation for Alpha Selection:**  
   Use `GridSearchCV` or `RidgeCV` / `LassoCV` in scikit-learn to find the best regularization strength (`alpha`).

3. **ElasticNet Exploration:**  
   Experiment with varying `l1_ratio` values (0.1 → 0.9) and note how the coefficient sparsity and performance change.

4. **Performance Comparison:**  
   Compute **MSE**, **RMSE**, and **R²** for Ridge, Lasso, and ElasticNet.  
   Compare them with a plain Linear Regression model.

5. **Conceptual Analysis:**  
   - When is **Lasso** better than **Ridge**?  
   - When should **ElasticNet** be preferred?  
   - Discuss how multicollinearity and sparsity affect each method.

---

### ✅ Next Topic:
📘 **404. Logistic Regression: Classification Basics]**
