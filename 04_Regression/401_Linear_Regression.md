# 401. Linear Regression and Its Assumptions

**Linear Regression** is one of the most fundamental and widely used algorithms in Machine Learning and Statistics.  
It models the relationship between a **dependent variable (target)** and one or more **independent variables (features)** by fitting a straight line.

---

## 🧩 1. What is Linear Regression?

Linear Regression assumes a **linear relationship** between input features `X` and target output `y`.

### 🔹 Simple Linear Regression Equation

$$ y = \beta_0 + \beta_1 x + \varepsilon $$

- Slope (m):
  
$$ \beta_1 \ = \frac{n(\sum xy) - (\sum x)(\sum y)}{n(\sum x^2) - (\sum x)^2} $$

- Intercept (c):

$$ \beta_0 \ = \frac{\sum y - m(\sum x)}{n} $$

Where:
- $( y \)$: Dependent variable (output)
- $( x \)$: Independent variable (input)
- $( \beta_0 \)$: Intercept term
- $( \beta_1 \)$: Coefficient (slope)
- $( \varepsilon \)$: Random error term

---

### 🔹 Multiple Linear Regression

For multiple features:

$$ y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \varepsilon $$

or in matrix form:

$$ \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon} $$

Where:
- $( \mathbf{X} \)$: Feature matrix (m × n)
- $( \boldsymbol{\beta} \)$: Coefficient vector
- $( \boldsymbol{\varepsilon} \)$: Error vector

---

## ⚙️ 2. Objective of Linear Regression

The goal is to find coefficients $( \boldsymbol{\beta} \)$ that minimize the **sum of squared errors (SSE)**:

$$ SSE = \sum_{i=1}^{m}(y_i - \hat{y}_i)^2 $$

where,
```math
\hat{y}_i = \beta_0 + \beta_1x_{i1} + \dots + \beta_nx_{in}
```


### 🧮 Closed-form Solution (Normal Equation)

```math
\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
```

This directly computes optimal coefficients for smaller datasets.

---

## 🔍 3. Interpretation of Coefficients

Each coefficient $( \beta_i \)$ represents:
- The expected change in `y` when `x_i` increases by 1 unit,  
  **keeping all other variables constant.**

For example:
- If $( \beta_1 = 2.5 \)$, then increasing `x1` by 1 unit increases `y` by 2.5 (on average).

---

## 📊 4. Visual Intuition
<img width="500" height="333" alt="image" src="https://github.com/user-attachments/assets/1a660977-3d00-4e3e-be49-f76d966715cf" />

---
### 🔍 4.1 Code Behind Linear Regression (From Scratch)
```python
# --- Linear Regression From Scratch ---

class LinearRegressionScratch:
    """
    A simple implementation of Linear Regression using the Least Squares Method.
    It finds the best fit line of the form: y = m*x + c
    """

    def __init__(self):
        # Initialize slope (m) and intercept (c) to zero
        self.m = 0
        self.c = 0

    def fit(self, X, y):
        """
        Train the linear regression model.
        Calculates the slope (m) and intercept (c) using the closed-form formula.

        Parameters:
        X : list of feature values (independent variable)
        y : list of target values (dependent variable)
        """

        # Total number of data points
        n = len(X)

        # Calculate the mean of X and Y
        mean_x = sum(X) / n
        mean_y = sum(y) / n

        # Numerator and Denominator for slope (m)
        # m = Σ((x - mean_x) * (y - mean_y)) / Σ((x - mean_x)^2)
        numerator = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((X[i] - mean_x) ** 2 for i in range(n))

        # Compute slope (m)
        self.m = numerator / denominator

        # Compute intercept (c)
        # c = mean_y - m * mean_x
        self.c = mean_y - self.m * mean_x

    def predict(self, X):
        """
        Predict the target values for given input data.

        Parameters:
        X : list of feature values

        Returns:
        A list of predicted target values using the equation y = m*x + c
        """
        return [self.m * x + self.c for x in X]


# --- Example Usage ---

# Input data
X = [1, 2, 3, 4, 5]  # Independent variable
y = [2, 4, 5, 4, 5]  # Dependent variable

# Create the model
model = LinearRegressionScratch()

# Train the model using training data
model.fit(X, y)

# Predict new values for unseen data points
predictions = model.predict([6, 7, 8])

# Display the results
print("=== Linear Regression Results ===")
print(f"Slope (m): {model.m:.2f}")
print(f"Intercept (c): {model.c:.2f}")
print(f"Predictions for [6, 7, 8]: {predictions}")

```
Explanation:

- The fit method calculates slope and intercept using the closed-form formula.
- The predict method uses those parameters to make predictions.

---
### 🔍 4.2 Simple Linear Regression Using `scikit-learn`

- When using libraries like Scikit-learn, much of this math is abstracted away.
```python
# --- Simple Linear Regression using Scikit-learn ---

# Import the necessary libraries
from sklearn.linear_model import LinearRegression  # For performing linear regression
import numpy as np                                 # For numerical operations

# -------------------------------
# STEP 1: Prepare the Data
# -------------------------------

# Feature values (independent variable)
# Each inner list represents one data point [x]
X = np.array([[1], [2], [3], [4], [5]])

# Target values (dependent variable)
# These are the corresponding y values for each x
y = np.array([2, 4, 5, 4, 5])

# -------------------------------
# STEP 2: Create the Model
# -------------------------------
model = LinearRegression()

# -------------------------------
# STEP 3: Train (Fit) the Model
# -------------------------------
# The model will find the best-fit line:
#   y = m*x + c
# where:
#   m = slope (coefficient)
#   c = intercept
model.fit(X, y)

# -------------------------------
# STEP 4: Make Predictions
# -------------------------------
# Now let's predict the values for new, unseen X data
X_test = np.array([[6], [7], [8]])   # New data points
y_pred = model.predict(X_test)       # Model predicts y values for them

# -------------------------------
# STEP 5: Display the Results
# -------------------------------

print("=== Simple Linear Regression Results ===")
print(f"Slope (Coefficient): {model.coef_[0]:.2f}")   # The value of m
print(f"Intercept: {model.intercept_:.2f}")           # The value of c
print(f"Predicted values for X_test {X_test.flatten()}: {y_pred}")


```
---

## ⚖️ 5. Cost Function — Mean Squared Error (MSE)

To train the model, we minimize:

$$
J(\beta_0, \beta_1, ..., \beta_n) = \frac{1}{m} \sum_{i=1}^{m}(y_i - \hat{y}_i)^2
$$

Minimizing `J` ensures the line fits as close as possible to the data points.

---

## 🔹 6. Assumptions of Linear Regression

For Linear Regression to produce valid and unbiased estimates, the following **five assumptions** must hold:

| # | Assumption | Description | How to Check | Violation Effect |
|---|-------------|--------------|---------------|------------------|
| 1 | **Linearity** | Relationship between X and y is linear | Scatter plots, residual plots | Model underfits data |
| 2 | **Independence of Errors** | Errors (residuals) are independent | Durbin-Watson test | Inflated Type I errors |
| 3 | **Homoscedasticity** | Constant variance of residuals | Plot residuals vs predicted values | Model inefficiency |
| 4 | **Normality of Errors** | Residuals follow normal distribution | Q-Q plot, Shapiro–Wilk test | Invalid confidence intervals |
| 5 | **No Multicollinearity** | Features not highly correlated | Variance Inflation Factor (VIF) | Unstable coefficients |

---

### 🔍 6.1 Linearity
Linear regression assumes that the relationship between predictors and target is linear.  
If not, transformations like `log(x)`, `x²`, or polynomial regression may help.

---

### 🔍 6.2 Independence of Errors
Residuals should be independent across observations.
- Checked using **Durbin-Watson test** (values near 2 = independent).

---

### 🔍 6.3 Homoscedasticity
Variance of residuals should be constant for all levels of predicted values.
- If variance increases/decreases → **heteroscedasticity**.
- Fix: log transformation or weighted least squares.

---

### 🔍 6.4 Normality of Errors
Residuals should be normally distributed for reliable inference.
- Use **Q-Q plot** or **histogram** of residuals.
- Fix: remove outliers, apply transformations.

---

### 🔍 6.5 No Multicollinearity
Independent variables should not be highly correlated.
- Compute **VIF (Variance Inflation Factor)**:
  - VIF > 5 (or 10) indicates strong collinearity.
- Fix: drop correlated features or apply **Principal Component Analysis (PCA)**.

---

## ⚙️ 7. Implementation (Conceptual Overview)

Although implementation isn’t the focus yet, conceptually the steps are:

1. Import dataset and explore.
2. Split data into training and testing sets.
3. Fit linear regression using Scikit-learn:
   - `from sklearn.linear_model import LinearRegression`
4. Evaluate using MSE, R², and residual plots.

---

## 🔢 8. Model Evaluation Metrics

| Metric | Formula | Interpretation |
|---------|----------|----------------|
| **R² (Coefficient of Determination)** | $( 1 - \frac{SS_{res}}{SS_{tot}} \)$ | Proportion of variance explained |
| **MSE (Mean Squared Error)** | $( \frac{1}{n}\sum (y_i - \hat{y}_i)^2 \)$ | Penalizes large errors |
| **MAE (Mean Absolute Error)** | $\( \frac{1}{n}\sum \|y_i - \hat{y}_i\| \)$ | Average absolute deviation |
| **Adjusted R²** | Adjusts R² for number of predictors | Prevents overfitting with many features |

---

## 🧠 9. Key Takeaways

- Linear Regression is interpretable, simple, and foundational.
- Assumptions must be tested to ensure reliable inference.
- Violating assumptions can lead to biased or inefficient estimates.
- It forms the basis for more advanced models like Ridge, Lasso, and ElasticNet.

---

## 🧩 10. Exercises

1. Derive the normal equation for multiple linear regression.  
2. Plot residuals for a given dataset and check homoscedasticity.  
3. Compute VIF for a dataset and interpret.  
4. Explain why violating multicollinearity affects coefficient stability.  
5. Compare R² and Adjusted R² in model evaluation.

---

### ✅ Next Topic:
📘 **402. Polynomial and Regularized Regression (Ridge, Lasso, ElasticNet)**

For **Simple Linear Regression**, the model fits the “best-fit” line through data points to minimize residuals.

