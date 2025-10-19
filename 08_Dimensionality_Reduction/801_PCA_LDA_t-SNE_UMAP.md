# 801. Dimensionality Reduction: PCA, LDA, t-SNE, UMAP

Dimensionality reduction is a key technique in Machine Learning to **reduce the number of features** while retaining the **essential information**.  
It helps in **visualization**, **computational efficiency**, and **reducing overfitting**.

---

## 🔹 1. Why Dimensionality Reduction?

- High-dimensional data (many features) can lead to the **curse of dimensionality**.
- Reduces computational cost and memory usage.
- Helps improve **model generalization**.
- Facilitates **data visualization** in 2D or 3D.

---

## 🔹 2. Principal Component Analysis (PCA)

**Concept:**  
PCA finds **orthogonal directions (principal components)** that capture the **maximum variance** in data.

- **Goal:** Project data onto fewer dimensions while retaining variance.
- **Unsupervised** method.

### Steps:
1. Standardize features.
2. Compute covariance matrix.
3. Compute eigenvalues & eigenvectors.
4. Sort eigenvectors by descending eigenvalues.
5. Project data onto top k eigenvectors.

### Formula:
\[
Z = X W
\]  
Where:  
- \(X\) = original data  
- \(W\) = matrix of top k eigenvectors  
- \(Z\) = reduced data

### Applications:
- Noise reduction
- Preprocessing for ML models
- Visualization of high-dimensional data

---

## 🔹 3. Linear Discriminant Analysis (LDA)

**Concept:**  
LDA is a **supervised** method that finds linear combinations of features that **maximize class separability**.

- Reduces dimensionality while maintaining **class-discriminative information**.

### Steps:
1. Compute mean vectors for each class.
2. Compute **within-class** and **between-class** scatter matrices.
3. Solve eigenvalue problem: maximize ratio of between-class to within-class scatter.
4. Project data onto top discriminant vectors.

### Applications:
- Face recognition
- Text classification
- Multi-class classification preprocessing

---

## 🔹 4. t-Distributed Stochastic Neighbor Embedding (t-SNE)

**Concept:**  
t-SNE is a **non-linear technique** for high-dimensional data visualization.

- Focuses on **preserving local structure** (neighbors) rather than global distances.
- Maps high-dimensional data into 2D or 3D space for visualization.

### Steps:
1. Compute pairwise similarities in high-dimensional space.
2. Define similarities in low-dimensional space.
3. Minimize **Kullback–Leibler divergence** between distributions.

### Applications:
- Visualizing embeddings (Word2Vec, BERT)
- Clustering visualization
- Image feature visualization

---

## 🔹 5. Uniform Manifold Approximation and Projection (UMAP)

**Concept:**  
UMAP is a **manifold learning technique** for dimensionality reduction and visualization.

- Preserves both **local and global structure** better than t-SNE.
- Faster and scalable to large datasets.

### Steps:
1. Construct a weighted k-nearest neighbor graph.
2. Optimize low-dimensional representation using stochastic gradient descent.
3. Visualize clusters or structures in 2D/3D.

### Applications:
- Embedding visualization
- Single-cell genomics
- Large-scale NLP embeddings

---

## 🔹 6. Comparison of Techniques

| Technique | Supervised/Unsupervised | Captures | Best Use Case |
|-----------|------------------------|-----------|---------------|
| PCA       | Unsupervised           | Variance  | Preprocessing, noise reduction |
| LDA       | Supervised             | Class separability | Classification preprocessing |
| t-SNE     | Unsupervised           | Local structure | Visualization of embeddings |
| UMAP      | Unsupervised           | Local + Global structure | Large-scale visualization, embeddings |

---

## 🧩 7. Exercises

1. Apply **PCA** on the MNIST dataset and plot the first 2 components.  
2. Compare **t-SNE** and **UMAP** visualizations on the same dataset.  
3. Implement **LDA** on a multi-class dataset and compare with PCA.  
4. Experiment with **number of components** and visualize effect on data separability.  
5. Analyze variance explained by each **principal component** in PCA.

---

✅ **Next Topic:**  
📘 *802. Clustering Revisited: Advanced Techniques & Evaluation*
