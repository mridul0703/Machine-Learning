# 505. Gradient Boosting (XGBoost, LightGBM, CatBoost)

Gradient Boosting is one of the most powerful and widely used ensemble techniques in Machine Learning.  
It builds a **strong predictive model** by combining multiple **weak learners (usually decision trees)** in a sequential manner — each correcting the errors of the previous one.

---

## 🧩 1. What is Boosting?

**Boosting** is an **ensemble method** that combines several **weak learners** to create a **strong learner**.

### Idea:
- Train a sequence of models.
- Each model tries to **fix the mistakes** made by the previous one.
- Final prediction = weighted sum of all weak learners.

### Comparison of Ensemble Methods:

| Method | Combines Models | Sequence | Example |
|---------|------------------|-----------|----------|
| **Bagging** | In parallel | No | Random Forest |
| **Boosting** | Sequentially | Yes | AdaBoost, Gradient Boosting, XGBoost |

---

## 🔹 2. Intuition Behind Gradient Boosting

Gradient Boosting uses **gradient descent** to minimize the **loss function** by adding weak learners sequentially.

### Step-by-step intuition:
1. Start with an initial prediction (often mean for regression).
2. Compute the **residuals** (errors).
3. Fit a weak learner (e.g., shallow decision tree) to these residuals.
4. Update the model by adding the new tree’s predictions, scaled by a learning rate.
5. Repeat until convergence.

---

## 🔹 3. Mathematical Formulation

We want to minimize a **loss function** $( L(y, F(x)) \)$ :

1. Initialize model:
```math
F_0(x) = \arg\min_c \sum_i L(y_i, c)
```

2. For each iteration $( m = 1, 2, ..., M \)$ :
   - Compute pseudo-residuals:
```math
r_{im} = - \left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x)=F_{m-1}(x)}
```
   - Fit a weak learner $( h_m(x) \)$ to predict $( r_{im} \)$
   - Compute optimal step size:
```math
\gamma_m = \arg\min_\gamma \sum_i L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))
```
   - Update model:
```math
F_m(x) = F_{m-1}(x) + \eta \gamma_m h_m(x)
```
where $( \eta \)$ = **learning rate**

---

## 🔹 4. Key Parameters

| Parameter | Description |
|------------|-------------|
| **n_estimators** | Number of boosting rounds (trees) |
| **learning_rate (η)** | Shrinks contribution of each tree (trade-off: small η → more trees) |
| **max_depth** | Controls tree complexity |
| **subsample** | Fraction of samples used per tree (for regularization) |
| **colsample_bytree** | Fraction of features used per tree |
| **loss** | Objective function (e.g., `mse`, `logloss`) |

---

## 🔹 5. Regularization in Gradient Boosting

Regularization prevents overfitting by controlling the model complexity.

- **Shrinkage**: Multiply each tree’s contribution by a small learning rate $( \eta \)$ .
- **Subsampling**: Train each tree on a random subset of data.
- **Tree pruning**: Limit depth or minimum samples per leaf.
- **L1/L2 penalties**: Used in advanced GBM libraries like XGBoost.

---

## 🔹 6. Advantages & Limitations

| Advantages | Limitations |
|-------------|-------------|
| High predictive accuracy | Sensitive to hyperparameters |
| Works well with structured/tabular data | Slower to train compared to Random Forest |
| Handles different loss functions | Can overfit with too many trees |
| Supports feature importance interpretation | Harder to parallelize (sequential learning) |

---

## 🔹 7. Popular Implementations

### ⚙️ XGBoost (Extreme Gradient Boosting)
- Highly optimized, parallelized version of GBM.
- Supports L1/L2 regularization.
- Handles missing values automatically.
- Supports classification, regression, ranking.

---

### ⚡ LightGBM (Light Gradient Boosting Machine)
- Developed by Microsoft.
- Optimized for **large datasets** and **high-dimensional features**.
- Uses **leaf-wise tree growth** for faster convergence.

---

### 🐱 CatBoost
- Developed by Yandex.
- **Handles categorical features automatically** (no need for one-hot encoding).
- Faster convergence and less hyperparameter tuning required.

---

## 🔹 8. Evaluation Metrics

| Task | Common Metrics |
|------|----------------|
| **Regression** | MSE, RMSE, MAE, R² |
| **Classification** | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Ranking** | NDCG, MAP |

---

## 🔹 9. Feature Importance

All three libraries (XGBoost, LightGBM, CatBoost) support **feature importance visualization**:
- **Gain-based**: Contribution to reducing loss.
- **Split-based**: Number of times a feature is used for splitting.

---

## 🔹 10. Practical Tips

✅ Use **early stopping** to prevent overfitting.  
✅ Tune **learning_rate** and **n_estimators** together (small η → large trees).  
✅ For categorical data, prefer **CatBoost**.  
✅ For massive datasets, prefer **LightGBM**.  
✅ Use **XGBoost** as a general-purpose, reliable baseline.

---

## 🔹 11. Real-World Applications

| Domain | Use Case |
|---------|----------|
| **Finance** | Credit scoring, fraud detection |
| **Healthcare** | Disease prediction, patient risk modeling |
| **E-commerce** | Recommendation systems, customer churn |
| **Insurance** | Risk modeling, premium prediction |
| **Competitions (Kaggle)** | Dominates leaderboard in structured data problems |

---

## 🧩 12. Exercises

1. Train XGBoost, LightGBM, and CatBoost on the **same dataset** — compare accuracy and training time.  
2. Perform **hyperparameter tuning** using `GridSearchCV` or `Optuna`.  
3. Plot **feature importance** and interpret top predictors.  
4. Enable **early stopping** and observe its effect.  
5. Try **regression** and **classification** tasks using all three methods.  
6. Compare performance on large vs small datasets.

---
