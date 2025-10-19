# 802. Feature Selection vs. Feature Extraction

Feature engineering is a critical step in Machine Learning. **Reducing dimensionality** or selecting the right features improves **model performance**, **reduces overfitting**, and **decreases computational cost**.  

Two main approaches: **Feature Selection** and **Feature Extraction**.

---

## 🔹 1. Feature Selection

**Definition:**  
Selecting a **subset of original features** that are most relevant to the prediction task.

- Does **not transform features**; keeps original feature semantics.
- Reduces noise and improves interpretability.

### Methods:

1. **Filter Methods**
   - Use **statistical measures** to select features.
   - Examples: Correlation, Chi-square, ANOVA F-test, Mutual Information.
   - Independent of ML algorithms.

2. **Wrapper Methods**
   - Use **ML model performance** to evaluate feature subsets.
   - Examples: Recursive Feature Elimination (RFE), Forward/Backward Selection.
   - Computationally expensive.

3. **Embedded Methods**
   - Feature selection happens **during model training**.
   - Examples: Lasso (L1 regularization), Tree-based feature importance (Random Forest, XGBoost).

### Applications:
- Reducing dimensionality for regression/classification.
- Improving interpretability.
- Removing irrelevant or redundant features.

---

## 🔹 2. Feature Extraction

**Definition:**  
Transforms original features into a **new set of features** (lower-dimensional representation).

- Original features may lose semantic meaning.
- Often used for **dimensionality reduction**.

### Methods:

1. **Principal Component Analysis (PCA)**
   - Projects data onto orthogonal components capturing maximum variance.

2. **Linear Discriminant Analysis (LDA)**
   - Projects data maximizing class separability.

3. **Autoencoders (Deep Learning)**
   - Learn compressed representations in the bottleneck layer.

4. **t-SNE / UMAP**
   - Non-linear transformations for visualization.

### Applications:
- Image compression.
- Noise reduction.
- Feature engineering for ML models.

---

## 🔹 3. Comparison Table

| Aspect | Feature Selection | Feature Extraction |
|--------|-----------------|-----------------|
| Approach | Select subset of original features | Transform features into new space |
| Semantics | Preserves original feature meaning | May lose original interpretation |
| Dimensionality Reduction | Optional | Always reduces dimensions |
| Computation | Usually faster | Can be computationally intensive |
| Examples | Lasso, RFE, Mutual Information | PCA, LDA, Autoencoders |

---

## 🧩 4. Exercises

1. Apply **feature selection** using correlation and Lasso on a dataset. Compare model performance.  
2. Apply **PCA** to reduce features to 2D and visualize clusters.  
3. Compare model accuracy using **selected features** vs **extracted features**.  
4. Discuss scenarios where feature selection is preferable over extraction and vice versa.

---

✅ **Next Topic:**  
📘 *803. Handling Missing Data & Imbalanced Features*
