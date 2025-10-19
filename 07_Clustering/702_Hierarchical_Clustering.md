# 702. Hierarchical Clustering

Hierarchical Clustering is an **unsupervised learning algorithm** that builds a hierarchy of clusters. Unlike k-Means (which requires specifying *k*), hierarchical clustering does **not require predefining** the number of clusters — it creates a **dendrogram** that shows the entire clustering process.

---

## 🔹 1. Overview

Hierarchical clustering works by either:
- **Agglomerative (Bottom-Up)** – start with each data point as its own cluster, then successively merge the most similar clusters.
- **Divisive (Top-Down)** – start with all data in one cluster, and successively divide into smaller clusters.

👉 Most implementations use **Agglomerative Clustering**.

---

## 🔹 2. Steps in Agglomerative Clustering

1. Compute the **distance (similarity)** between all pairs of data points.  
2. Treat each point as an individual cluster.  
3. Merge the **two closest clusters** based on a linkage criterion.  
4. Recompute distances between clusters.  
5. Repeat until all points belong to a single cluster.  
6. The process is visualized using a **dendrogram**.

---

## 🔹 3. Distance Metrics

The distance between points (or clusters) can be calculated using various metrics:

| Metric | Formula | Description |
|---------|----------|-------------|
| **Euclidean** | √Σ(xᵢ - yᵢ)² | Straight-line distance |
| **Manhattan** | Σ\|xᵢ - yᵢ\| | Absolute distance |
| **Cosine** | 1 - (x·y / \|\|x\|\| \|\|y\|\|) | Angle-based similarity |
| **Correlation** | 1 - corr(x, y) | Measures statistical relationship |

---

## 🔹 4. Linkage Methods

When merging clusters, linkage defines **how distance between clusters** is calculated.

| Linkage | Formula | Description |
|----------|----------|-------------|
| **Single Linkage** | min(distance between points in clusters) | Sensitive to noise |
| **Complete Linkage** | max(distance between points in clusters) | Produces compact clusters |
| **Average Linkage** | mean(distance between all pairs) | Balanced approach |
| **Ward’s Linkage** | Minimizes variance increase | Most common in practice |

**Ward’s method** often performs best for numeric data because it aims to minimize total within-cluster variance.

---

## 🔹 5. Dendrogram

A **dendrogram** is a tree-like diagram that shows how clusters are merged or split at each step.

- The **y-axis** represents the distance (or dissimilarity).  
- The **x-axis** represents individual data points.  
- Cutting the dendrogram at a certain height gives a chosen number of clusters.

📊 *Interpretation Example:*
- Short vertical lines → closely related points.  
- Long vertical lines → distinct clusters.

---

## 🔹 6. Choosing the Number of Clusters

To determine an optimal number of clusters:
1. Plot a **dendrogram**.
2. Choose a **cut-off height** where the vertical distance (distance between merges) is large.
3. The number of vertical lines cut by this threshold = number of clusters.

This is similar to the **elbow method** used in k-Means.

---

## 🔹 7. Advantages & Disadvantages

| Advantages | Disadvantages |
|-------------|----------------|
| No need to specify number of clusters beforehand | Computationally expensive (O(n²)) |
| Dendrogram provides visual interpretation | Sensitive to noise and outliers |
| Works well with small datasets | Hard to handle large datasets efficiently |

---

## 🔹 8. Use Cases in Machine Learning

| Use Case | Description |
|-----------|-------------|
| **Document Clustering** | Grouping similar documents or news articles |
| **Customer Segmentation** | Identifying groups of similar users or customers |
| **Genetic Data Analysis** | Grouping genes based on similarity |
| **Image Segmentation** | Clustering pixels into meaningful regions |
| **Anomaly Detection** | Outliers appear as separate clusters in dendrogram |

---

## 🔹 9. Example Workflow

1. **Import libraries:**

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(data, method='ward')

dendrogram(Z)
plt.show()

model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(data)
```

---

## 🔹 10. Comparison with k-Means
| Feature	| Hierarchical Clustering |	k-Means |
|---------|-------------------------|---------|
| Need to specify k	| ❌ Optional | ✅ Required |
| Output	| Dendrogram	| Cluster centroids |
| Cluster shape	| Non-spherical	| Spherical |
| Computation	| O(n²)	| O(n) per iteration |
| Suitable for	| Small datasets, visual analysis	| Large datasets |

---

## 🧩 11. Exercises

1. Perform Agglomerative Clustering with different linkage methods.
2. Plot dendrogram and interpret cluster formation.
3. Compare hierarchical clustering with k-Means on a small dataset.
4. Try hierarchical clustering with cosine distance on text embeddings.
5. Use dendrogram cut-off to select optimal number of clusters.

---

## ✅ Next Topic:
📘 703. DBSCAN (Density-Based Clustering)

