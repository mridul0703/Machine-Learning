# 402. Polynomial Regression

**Polynomial Regression** is an extension of Linear Regression that models **non-linear relationships** between the independent variable(s) and the dependent variable by introducing polynomial terms.

---

## 🧩 1. What is Polynomial Regression?

Linear Regression assumes a linear relationship:

$$
y = \beta_0 + \beta_1 x + \varepsilon
$$

Polynomial Regression adds **powers of the input feature(s)** to capture curvature:

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_d x^d + \varepsilon
$$

Where:
- $(d\)$ = degree of the polynomial
- $(x^i\)$ = polynomial feature of order \(i\)

**Example:**  
- Degree 2 (quadratic): $( y = \beta_0 + \beta_1 x + \beta_2 x^2 + \varepsilon \) $ 
- Degree 3 (cubic): $( y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \varepsilon \)$

---

## 🔹 2. Why Polynomial Regression?

- Captures **non-linear patterns** while remaining linear in coefficients.  
- Avoids underfitting in datasets with curvature.  
- Can be used with **multiple features** by including interaction terms.

---

## ⚙️ 3. Feature Transformation

Polynomial Regression uses **feature expansion**:

$$
X = [x_1, x_2, ..., x_n] \rightarrow [1, x_1, x_1^2, ..., x_1^d, x_2, x_2^2, ..., x_n^d]
$$

**Scikit-learn Example:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Create a pipeline
poly_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('linear_model', LinearRegression())
])

poly_model.fit(X_train, y_train)
y_pred = poly_model.predict(X_test)
```
---

## 🔹 4. Model Complexity and Degree Selection

- **Low degree** → may underfit (high bias)  
- **High degree** → may overfit (high variance)  
- **Rule of thumb:** Use cross-validation to select an optimal degree \(d\)

---

## 🔹 5. Assumptions

Polynomial Regression inherits **Linear Regression assumptions**:

1. Linearity in coefficients (not necessarily in X)  
2. Independence of residuals  
3. Homoscedasticity (constant variance)  
4. Normality of residuals  
5. No multicollinearity among polynomial terms (especially high-degree terms)

> **Tip:** High-degree polynomials often introduce multicollinearity. Regularization (Ridge/Lasso) is recommended.

---

## 🔹 6. Evaluation Metrics

Same as Linear Regression:

| Metric | Formula | Description |
|--------|---------|-------------|
| MSE | $( \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \)$ | Penalizes large errors |
| RMSE | $( \sqrt{MSE} \)$ | Standard deviation of residuals |
| MAE | $( \frac{1}{n} \sum_{i=1}^{n} \|y_i - \hat{y}_i\| \)$ | Average absolute error |
| R² | $( 1 - \frac{SS_{res}}{SS_{tot}} \)$ | Explained variance |

---

## 🔹 7. Visual Intuition

<img width="734" height="482" alt="image" src="https://github.com/user-attachments/assets/209f64e6-9991-443e-9240-d11fb9fa01a6" />

- Polynomial regression fits the curved pattern, while linear regression may underfit.

---

## 🔹 8. Use Cases in ML

| Use Case | Description |
|----------|-------------|
| Non-linear trends | Stock price modeling, temperature prediction |
| Curve fitting | Engineering and physics data |
| Feature expansion | Basis for interaction terms in multiple regression |
| Preparing for regularization | Polynomial features + Ridge/Lasso for robust fitting |

---

## 🧩 9. Exercises

1. Fit a polynomial regression (degree 2 and 3) on a dataset and compare R².  
2. Plot residuals and check homoscedasticity.  
3. Observe multicollinearity with degree 5 or higher and suggest solutions.  
4. Combine polynomial regression with Ridge regularization.

---

### ✅ Next Topic:

📘 **403. Regularization: Ridge, Lasso, ElasticNet**
