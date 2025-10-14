# 304. Introduction to Scikit-learn

Scikit-learn (or `sklearn`) is one of the most popular and powerful Python libraries for **Machine Learning**.  
It provides a consistent, easy-to-use API for implementing a wide range of algorithms — from regression and classification to clustering and dimensionality reduction — along with data preprocessing, model evaluation, and deployment tools.

---

## 🧠 1. What is Scikit-learn?

Scikit-learn is a **high-level Machine Learning library** built on top of:
- **NumPy** → numerical computations
- **SciPy** → scientific and statistical functions
- **Matplotlib** → visualizations

It is designed to **simplify the ML workflow** — from data preprocessing to model training, tuning, and evaluation.

---

## ⚙️ 2. Why Use Scikit-learn?

| Feature | Description |
|----------|--------------|
| 🧩 **Unified API** | All algorithms follow a consistent `.fit()`, `.predict()`, `.score()` interface |
| ⚡ **Efficiency** | Built on optimized NumPy/SciPy operations |
| 🧰 **Wide Range of Algorithms** | Regression, Classification, Clustering, Dimensionality Reduction |
| 🧼 **Preprocessing Tools** | Scaling, Encoding, Imputation, Feature Selection |
| 📈 **Evaluation Tools** | Metrics, Cross-validation, GridSearchCV |
| 🚀 **Integration** | Works seamlessly with Pandas, NumPy, Matplotlib, and joblib |

---

## 🧩 3. Scikit-learn Workflow Overview

A standard ML pipeline in Scikit-learn follows these steps:

1. **Load and Prepare Data**  
   - Import dataset (CSV, API, or built-in `sklearn.datasets`)
   - Handle missing values, categorical encoding, scaling

2. **Split Data**
   - Use `train_test_split()` to separate training and testing data

3. **Choose an Algorithm**
   - Regression, Classification, or Clustering model

4. **Train Model**
   - `model.fit(X_train, y_train)`

5. **Make Predictions**
   - `y_pred = model.predict(X_test)`

6. **Evaluate Model**
   - Use metrics like accuracy, precision, recall, RMSE

7. **Optimize Hyperparameters**
   - Use `GridSearchCV` or `RandomizedSearchCV`

8. **Deploy**
   - Save model using `joblib` or `pickle`

---

## 🧰 4. Core Components of Scikit-learn

### 4.1 Datasets
Scikit-learn provides many small, clean datasets for practice:
- `load_iris()`
- `load_digits()`
- `load_wine()`
- `load_breast_cancer()`

Also includes:
```python
from sklearn.datasets import fetch_openml
data = fetch_openml(name="titanic", version=1)
```

### 4.2 Data Preprocessing
Modules to prepare raw data:
- **Scaling:** `StandardScaler`, `MinMaxScaler`
- **Encoding:** `OneHotEncoder`, `LabelEncoder`
- **Missing Values:** `SimpleImputer`, `KNNImputer`
- **Feature Selection:** `SelectKBest`, `RFE`

---

## 🤖 5. Machine Learning Algorithms in Scikit-learn

| Category | Algorithms |
|-----------|-------------|
| **Regression** | Linear Regression, Ridge, Lasso, ElasticNet, SVR |
| **Classification** | Logistic Regression, SVM, Random Forest, Gradient Boosting, k-NN |
| **Clustering** | K-Means, DBSCAN, Agglomerative Clustering |
| **Dimensionality Reduction** | PCA, t-SNE, TruncatedSVD |
| **Model Ensemble** | Bagging, Boosting, Stacking |

---

## 🧮 6. Model Evaluation Tools

Scikit-learn provides metrics to evaluate performance:

| Type | Common Metrics |
|------|----------------|
| **Regression** | MSE, RMSE, MAE, R² |
| **Classification** | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Clustering** | Silhouette Score, Davies-Bouldin Index |

**Cross-validation:**  
- `cross_val_score()` for robust performance estimates  
**Train-test split:**  
- `train_test_split()` from `sklearn.model_selection`

---

## 🔍 7. Model Optimization & Hyperparameter Tuning

| Technique | Description |
|------------|-------------|
| **GridSearchCV** | Exhaustively searches best hyperparameters |
| **RandomizedSearchCV** | Random search within defined distributions |
| **Pipeline + GridSearch** | Automate preprocessing + model tuning |
| **Cross-validation** | K-fold validation ensures generalization |

---

## 🧠 8. Model Persistence (Saving and Loading Models)

To deploy or reuse trained models:
```python
from joblib import dump, load

# Save model
dump(model, 'model.joblib')

# Load model
model = load('model.joblib')
```

This ensures you can reload trained models without retraining.

---

## 🧩 9. Integrating Scikit-learn with Other Libraries

| Library | Integration Benefit |
|----------|---------------------|
| **NumPy / Pandas** | Efficient data structures for X, y |
| **Matplotlib / Seaborn** | Visualization of results and metrics |
| **TensorFlow / PyTorch** | Use for hybrid models (e.g., feature preprocessing) |
| **MLflow / DVC** | Track experiments and model versions |

---

## 📚 10. Practical Use Cases

| Use Case | Algorithm | Goal |
|-----------|------------|------|
| Predict house prices | Linear Regression | Continuous prediction |
| Spam email detection | Logistic Regression, SVM | Binary classification |
| Customer segmentation | K-Means | Unsupervised grouping |
| Reduce image noise | PCA | Dimensionality reduction |
| Predict disease presence | Random Forest | Classification |

---

## 🧾 11. Key Advantages

- Beginner-friendly and production-ready  
- Easily integrates with Python data ecosystem  
- Great for small to medium-sized datasets  
- Consistent, reliable API  
- Ideal for fast prototyping

---

## 🧩 12. Limitations

- Not optimized for massive-scale data (use Spark MLlib or TensorFlow)
- Limited GPU utilization
- Deep learning support is minimal

---

## 🧮 13. Summary

Scikit-learn is your **go-to library for classical Machine Learning**:
- Clean, consistent APIs  
- Covers the full ML pipeline  
- Excellent for experimentation, research, and education  
- A must-know foundation before moving to deep learning frameworks

---

## 🧠 14. Exercises

1. Load the `iris` dataset and train a Decision Tree classifier.  
2. Try scaling features using `StandardScaler` and compare model performance.  
3. Perform 5-fold cross-validation with `LogisticRegression`.  
4. Use `GridSearchCV` to tune hyperparameters of a `RandomForestClassifier`.  
5. Save your best model using `joblib` and reload it for prediction.

---

### ✅ Next Topic:
📘 **Building Basic ML Workflows & Pipelines**
