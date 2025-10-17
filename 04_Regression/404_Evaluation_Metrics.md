# 404. Evaluation Metrics: RMSE, MAE, R² Score

Evaluation metrics help us measure **how well a regression model performs** — how close its predictions are to the actual values.  
In regression tasks, our goal is to **minimize the error** between predicted $(\hat{y}\)$ and actual $(y\)$ outputs.

---

## 🧩 1. Introduction

When training a regression model, we need to answer:
- How accurate are our predictions?
- Are errors large or small?
- Does the model generalize well to unseen data?

To quantify this, we use **error metrics** like:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score (Coefficient of Determination)**

Each provides a different perspective on performance.

---

## 🔹 2. Mean Absolute Error (MAE)

### 📘 Definition
$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

### 🧠 Intuition
- Measures the **average magnitude of errors**, without considering direction (positive/negative).  
- Treats all errors **equally**, giving a linear penalty.

### ✅ Properties
- Always non-negative.
- Easy to interpret — directly in the same units as the target variable.
- Less sensitive to **outliers** than RMSE.

### 💡 Example
If actual values are [3, 5, 2, 7] and predicted [2, 5, 4, 8]:

$$
MAE = \frac{|3-2| + |5-5| + |2-4| + |7-8|}{4} = \frac{1+0+2+1}{4} = 1
$$

---

## 🔹 3. Root Mean Squared Error (RMSE)

### 📘 Definition
$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

### 🧠 Intuition
- Penalizes **larger errors more heavily** (because of the square term).
- Useful when large deviations are particularly undesirable.

### ✅ Properties
- Always ≥ MAE (since squares inflate larger errors).
- More sensitive to **outliers**.
- Same units as the target variable.

### 💡 Example
$$
RMSE = \sqrt{\frac{(1)^2 + (0)^2 + (-2)^2 + (-1)^2}{4}} = \sqrt{\frac{6}{4}} = 1.225
$$

---

## 🔹 4. R² Score (Coefficient of Determination)

### 📘 Definition
$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

Where:
- $( SS_{res} = \sum (y_i - \hat{y}_i)^2 \)$ → residual sum of squares  
- $( SS_{tot} = \sum (y_i - \bar{y})^2 \)$ → total sum of squares  

### 🧠 Intuition
- Measures how much variance in the dependent variable is explained by the model.  
- **R² = 1** → perfect predictions  
- **R² = 0** → model predicts no better than the mean  
- **R² < 0** → model performs worse than simply predicting the mean

### ✅ Properties
- Unitless metric (dimensionless).  
- Easy to interpret as a “percentage of explained variance.”  
- Can be misleading with **non-linear** models — may not fully capture quality.

### 💡 Example
If your model explains 90% of the variance in data,  
then $( R² = 0.9 \)$.

---

## 🔹 5. Comparison of Metrics

| Metric | Formula | Sensitivity | Handles Outliers | Range | Interpretation |
|---------|----------|--------------|------------------|--------|----------------|
| **MAE** | $( \frac{1}{n} \sum \|y - \hat{y}\| \)$ | Linear | Robust | ≥ 0 | Average error |
| **RMSE** | $( \sqrt{\frac{1}{n} \sum (y - \hat{y})^2} \)$ | Quadratic | Sensitive | ≥ 0 | Penalizes large errors |
| **R²** | $( 1 - \frac{SS_{res}}{SS_{tot}} \)$ | Variance-based | Depends | -∞ to 1 | Goodness of fit |

---

## 🔹 6. Choosing the Right Metric

| Situation | Recommended Metric | Reason |
|------------|--------------------|--------|
| When **outliers** are present | MAE | Less sensitive to large errors |
| When **large errors** are very costly | RMSE | Penalizes big mistakes |
| When you need **variance explanation** | R² | Shows how much variance is captured |
| When comparing **models** | All three | Provides a holistic performance view |

---

## 🧮 7. Practical Example (Conceptual)

Let’s say we’re predicting **house prices**:

| Actual Price (₹) | Predicted Price (₹) |
|-------------------|----------------------|
| 45,00,000 | 44,00,000 |
| 60,00,000 | 63,00,000 |
| 52,00,000 | 54,00,000 |

Then:
- MAE = average(|error|) = ₹1,6666
- RMSE = slightly higher (because of squaring)
- R² ≈ 0.93 → 93% variance explained

---

## 🧩 8. Exercises

1. Compute **MAE**, **RMSE**, and **R²** for a regression model on a sample dataset.  
2. Compare MAE vs. RMSE when outliers are added — what changes?  
3. Given two models:  
   - Model A: RMSE = 5, MAE = 4, R² = 0.85  
   - Model B: RMSE = 7, MAE = 5, R² = 0.92  
   Which one performs better overall? Why?  
4. Why might **R²** decrease when adding irrelevant features?  
5. Create a visualization comparing predicted vs. actual values and interpret residual patterns.

---
