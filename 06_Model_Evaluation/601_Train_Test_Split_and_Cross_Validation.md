# 📘 601. Train-Test Split & Cross-Validation

Splitting data properly is a critical part of building **robust and generalizable machine learning models**.  
This section covers the fundamental techniques: **Train-Test Split** and **Cross-Validation**.

---

## 🔹 1. Why Data Splitting is Necessary

When we train a model, we want it to perform well on **unseen data**, not just the data it was trained on.  
If we test the model on the same data used for training, it can **memorize** the data and **overfit**, giving a false impression of accuracy.

Data splitting ensures that:
- The model learns from one part of the data (**training set**)
- And is tested on another unseen part (**testing set**)

---

## 🔹 2. The Basic Split

| Dataset | Purpose | Typical Size |
|----------|----------|--------------|
| **Training Set** | Used to train the model | 70–80% |
| **Testing Set** | Used to evaluate the model’s performance | 20–30% |

**Example:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- `test_size=0.2` → 20% for testing  
- `random_state` → ensures reproducibility  

---

## 🔹 3. Validation Set

Sometimes we need **three splits**:
| Dataset | Purpose |
|----------|----------|
| **Training Set** | Fit the model |
| **Validation Set** | Tune hyperparameters |
| **Testing Set** | Final unbiased evaluation |

**Process:**
1. Train the model on the training set.  
2. Adjust parameters using the validation set.  
3. Once the model is finalized, evaluate on the test set.  

However, using a single validation set can still be unstable — results may depend on which data points end up in it.

---

## 🔹 4. Cross-Validation (CV)

Cross-validation improves model reliability by testing it on **multiple data folds** instead of one fixed test set.

### 🧩 a. k-Fold Cross-Validation
1. Split dataset into **k equal parts (folds)**.  
2. Train on **k-1 folds**, validate on the remaining one.  
3. Repeat k times, each time using a different fold for validation.  
4. Average the results.

**Example (k=5):**
```
Fold 1 → Test on part 1, train on 2–5  
Fold 2 → Test on part 2, train on 1,3–5  
...
Fold 5 → Test on part 5, train on 1–4
```

**Code Example:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("Average R²:", scores.mean())
```

---

### 🧩 b. Stratified k-Fold (for Classification)
Ensures that each fold maintains the **same class distribution** as the original dataset (important for imbalanced data).

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X, y):
    print("TRAIN:", train_idx, "TEST:", test_idx)
```

---

### 🧩 c. Leave-One-Out Cross-Validation (LOOCV)

- A special case of k-fold CV where **k = number of samples**.  
- Each sample acts once as a test set, and the rest as training.  
- Very accurate but computationally expensive.

---

## 🔹 5. Comparison of Methods

| Method | Pros | Cons | Best Used For |
|---------|------|------|---------------|
| **Train-Test Split** | Fast and simple | High variance in results | Large datasets |
| **k-Fold CV** | More reliable estimates | Computationally heavier | Moderate datasets |
| **Stratified k-Fold** | Maintains class balance | Slightly slower | Classification tasks |
| **LOOCV** | Almost unbiased | Very slow for large n | Small datasets |

---

## 🔹 6. Choosing the Right Method

| Dataset Size | Recommended Split |
|---------------|-------------------|
| Small (<1000 samples) | 5–10 fold cross-validation |
| Medium (1k–50k) | 5-fold CV or 80/20 split |
| Large (>50k) | Simple train-test split (e.g., 90/10) |

---

## 🔹 7. Common Pitfalls

1. **Data Leakage:**  
   Never split after applying transformations (like scaling or PCA).  
   Always apply transformations **after splitting** and **fit them only on training data**.

2. **Imbalanced Data:**  
   Use **stratified** sampling to ensure equal class proportions.

3. **Time-Series Data:**  
   Never shuffle; always split chronologically.

---

## 🔹 8. Evaluation Metrics Recap

During each fold/test phase, you can measure:
- **Classification:** Accuracy, Precision, Recall, F1, ROC-AUC  
- **Regression:** MSE, RMSE, MAE, R²  

Then average across folds for overall performance.

---

## 🧩 9. Exercises

1. Perform a simple 80/20 train-test split on a dataset.  
2. Implement 5-fold and 10-fold CV on the same data and compare metrics.  
3. Try StratifiedKFold on a classification dataset.  
4. Measure model variance between folds.  
5. Apply TimeSeriesSplit for sequential data and observe differences.

---

### ✅ Next Topic:
📘 602. Model Evaluation: Bias-Variance Tradeoff
