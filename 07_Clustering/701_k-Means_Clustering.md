
# 701. k-Means Clustering

k-Means is one of the most popular **unsupervised learning algorithms** used for **clustering** — grouping similar data points into clusters based on feature similarity.

---

## 🔹 1. Introduction to Clustering

- **Clustering** is the process of dividing a dataset into groups (clusters) where:
  - Data points in the **same cluster** are similar to each other.
  - Data points in **different clusters** are dissimilar.

- **Applications:**
  - Customer segmentation in marketing
  - Image compression
  - Document categorization
  - Anomaly detection

---

## 🔹 2. What is k-Means?

- k-Means aims to **partition n data points into k clusters**, where each data point belongs to the cluster with the nearest mean (centroid).

### Objective Function:
The goal is to minimize the **Within-Cluster Sum of Squares (WCSS)**:

```math
J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
```

Where:
- $( C_i \)$ → Cluster i  
- $( \mu_i \)$ → Centroid (mean) of cluster i  
- $( \|\|x - \mu_i\|\|^2 \)$ → Squared distance between data point and cluster center

---

## 🔹 3. k-Means Algorithm (Step-by-Step)

1. **Choose k**, the number of clusters.
2. **Initialize centroids** randomly from the dataset.
3. **Assign each data point** to the nearest centroid (based on Euclidean distance).
4. **Recompute centroids** as the mean of all assigned points in each cluster.
5. **Repeat Steps 3–4** until convergence (centroids stop moving or changes are minimal).

---

## 🔹 4. Distance Metrics

The most common distance measure is **Euclidean distance**:

```math
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
```

Other distances:
- **Manhattan distance:** $( \sum \|x_i - y_i\| \)$
- **Cosine similarity:** $( 1 - \frac{x \cdot y}{\|\|x\|\| \|\|y\|\|} \)$

---

## 🔹 5. Choosing the Number of Clusters (k)

Finding the right number of clusters is crucial.

### a. Elbow Method
- Plot **WCSS vs. k** and look for the “elbow” point where improvement slows down.

### b. Silhouette Score
Measures how similar a point is to its own cluster vs. others.

```math
S = \frac{b - a}{\max(a, b)}
```

Where:
- $( a \)$ = average intra-cluster distance  
- $( b \)$ = average nearest-cluster distance  
- $( S \in [-1, 1] \)$, where higher is better.

### c. Gap Statistic
Compares WCSS with expected WCSS under a random reference distribution.

---

## 🔹 6. Strengths and Weaknesses

| Strengths | Weaknesses |
|------------|-------------|
| Simple and fast | Must predefine k |
| Scales to large datasets | Sensitive to outliers |
| Works well for spherical clusters | Fails for non-convex shapes |
| Easy to interpret | Sensitive to initialization |

---

## 🔹 7. Initialization Techniques

Poor initialization may lead to bad clustering.

### Random Initialization
- Choose random points as initial centroids.

### k-Means++
- Improves initialization by spreading out initial centroids.
- Reduces chances of poor convergence.
- Now the **default in scikit-learn**.

---

## 🔹 8. Variants of k-Means

| Variant | Description |
|----------|--------------|
| **Mini-Batch k-Means** | Faster for large datasets; updates centroids using small random batches. |
| **Fuzzy c-Means** | Allows partial membership in clusters. |
| **Bisecting k-Means** | Divides clusters hierarchically for better results. |

---

## 🔹 9. Mathematical Insights

At convergence:
```math
\frac{\partial J}{\partial \mu_i} = 0 \Rightarrow \mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x
```

This shows that each cluster’s centroid is simply the **mean of all its points**.

---

## 🔹 10. k-Means in Scikit-learn

```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,2],[1,4],[1,0],
               [10,2],[10,4],[10,0]])

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)
```

---

## 🔹 11. Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **Inertia (WCSS)** | Measures internal cluster compactness |
| **Silhouette Score** | Measures separation between clusters |
| **Davies–Bouldin Index** | Lower value = better separation |

---

## 🔹 12. Common Issues & Solutions

| Issue | Solution |
|--------|-----------|
| Poor initialization | Use k-Means++ |
| Outliers | Remove or scale data |
| Non-spherical clusters | Try DBSCAN or GMM |
| Scaling effects | Standardize data before clustering |

---

## 🔹 13. Visual Example

Example: k=3 Clustering on 2D dataset

```
   Cluster 1 ●●●
   Cluster 2 ▲▲▲
   Cluster 3 ◆◆◆
   Each cluster’s centroid shown as ⊕
```

---

## 🔹 14. Applications in ML

| Application | Description |
|--------------|-------------|
| Customer Segmentation | Group users based on behavior |
| Image Compression | Represent pixels using k colors |
| Anomaly Detection | Detect unusual data patterns |
| Document Clustering | NLP topic modeling |
| Feature Engineering | Pre-clustering before supervised learning |

---

## 🧩 15. Exercises

1. Implement k-Means on a real dataset (e.g., Iris dataset).
2. Use the **Elbow Method** and **Silhouette Score** to find the optimal k.
3. Compare performance with **Mini-Batch k-Means**.
4. Visualize clusters in 2D using PCA.
5. Experiment with **different distance metrics** and report the impact.

---

## ✅ Next Topic:
📘 702. Hierarchical Clustering
