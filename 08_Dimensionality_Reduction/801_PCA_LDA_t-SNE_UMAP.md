# 801. Dimensionality Reduction: PCA, LDA, t-SNE, UMAP

Dimensionality reduction is a cornerstone technique in Machine Learning. It reduces the number of features while retaining essential information. This improves computational efficiency, reduces overfitting, and helps visualize high-dimensional data in 2D or 3D.  

---

## 🔹 1. Why Dimensionality Reduction?

High-dimensional data presents challenges:

- **Curse of dimensionality:** Distance metrics lose meaning, sparsity increases, learning becomes harder.
- **Computational cost:** More features → more memory and computation.
- **Overfitting risk:** High-dimensional datasets can lead to models that memorize rather than generalize.
- **Visualization:** Easier interpretation in lower dimensions (2D/3D).

**Applications:**
- Data preprocessing
- Noise reduction
- Visualization of embeddings
- Feature extraction for deep learning

---

## 🔹 2. Principal Component Analysis (PCA)

### Concept
PCA is an **unsupervised linear method** that projects data onto orthogonal directions (principal components) that capture **maximum variance**.

### Mathematical Foundation
Given a dataset $( X \in \mathbb{R}^{n \times d} \)$:

1. **Standardize data** to zero mean and unit variance.
2. Compute the **covariance matrix**:
```math
\Sigma = \frac{1}{n-1} X^T X
```
3. Solve the **eigenvalue problem**:
```math
\Sigma v_i = \lambda_i v_i
```
- $( v_i \)$ = eigenvectors (principal directions)
- $( \lambda_i \)$ = eigenvalues (variance along each direction)
4. Sort eigenvectors by descending eigenvalues.
5. Project data:
```math
Z = X W_k
```
- $( W_k \)$ = top $( k \)$ eigenvectors
- $( Z \)$ = reduced data in $( k \)$ -dimensions

### Intuition
- PCA finds the axes along which data varies the most.
- Projecting onto top components preserves most information while reducing dimensions.

### Applications
- Noise reduction
- Preprocessing for ML models
- Visualization of high-dimensional data (e.g., MNIST, embeddings)

### Practical Tips
- Always **standardize features** before PCA.
- Analyze **explained variance ratio** to select number of components.
- Use PCA as a **pre-step before t-SNE/UMAP** for computational efficiency.

---

## 🔹 3. Linear Discriminant Analysis (LDA)

### Concept
LDA is a **supervised linear method** that reduces dimensionality by maximizing **class separability**. Unlike PCA, it considers labels.

### Mathematical Foundation
1. Compute **class mean vectors** $( \mu_i \)$ for each class.
2. Compute **within-class scatter matrix** $( S_W \)$ and **between-class scatter matrix** $( S_B \)$:
```math
S_W = \sum_i \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T
```
```math
S_B = \sum_i n_i (\mu_i - \mu)(\mu_i - \mu)^T
```
3. Solve the **generalized eigenvalue problem**:
```math
S_W^{-1} S_B v = \lambda v
```
- Eigenvectors = directions that maximize **class separability**

### Applications
- Face recognition
- Text classification
- Multi-class classification preprocessing

### Practical Tips
- Works best when **classes are linearly separable**.
- Maximum number of components = **number of classes - 1**.
- Standardize features before LDA for better stability.

---

## 🔹 4. t-Distributed Stochastic Neighbor Embedding (t-SNE)

### Concept
t-SNE is a **non-linear, unsupervised method** for visualizing high-dimensional data by preserving **local neighborhoods**.

### Mathematical Foundation
1. Compute pairwise similarities in high-dimensional space $( P_{ij} \)$ (conditional probabilities based on Gaussian kernels).
2. Define low-dimensional similarities $( Q_{ij} \)$ using **Student t-distribution**:
```math
Q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
```
3. Minimize **Kullback-Leibler divergence** between $( P \)$ and $( Q \)$ :
```math
KL(P \| Q) = \sum_{i \neq j} P_{ij} \log \frac{P_{ij}}{Q_{ij}}
```

### Applications
- Visualization of word embeddings (Word2Vec, BERT)
- Neural network embedding inspection
- Cluster exploration

### Practical Tips
- Sensitive to **perplexity** (suggest 5–50) and **learning rate**.
- Often preceded by PCA to reduce dimensionality (e.g., 50 → 2/3) for speed.
- Primarily for **visualization**, not feature extraction.

---

## 🔹 5. Uniform Manifold Approximation and Projection (UMAP)

### Concept
UMAP is a **manifold learning method** preserving both **local and global structure**. Faster and scalable than t-SNE.

### Mathematical Foundation
1. Construct **weighted k-nearest neighbor graph** based on high-dimensional data.
2. Optimize low-dimensional representation using **cross-entropy loss** and stochastic gradient descent.
3. Preserves **topological structure** of the manifold.

### Applications
- Large-scale embedding visualization (NLP, genomics)
- Clustering analysis
- Preprocessing for downstream ML tasks

### Practical Tips
- Works well with very **large datasets**.
- Tune **n_neighbors** for local/global balance and **min_dist** for cluster tightness.

---

## 🔹 6. Comparison of Techniques

| Technique | Supervised/Unsupervised | Preserves | Best Use Case |
|-----------|------------------------|-----------|---------------|
| PCA       | Unsupervised           | Variance  | Preprocessing, noise reduction |
| LDA       | Supervised             | Class separability | Classification preprocessing |
| t-SNE     | Unsupervised           | Local structure | Embedding visualization, clusters |
| UMAP      | Unsupervised           | Local + Global | Large-scale visualization, embeddings |

---

## 🔹 7. Exercises (Beginner → Expert)

1. Apply **PCA** on MNIST or CIFAR dataset, plot first 2 components, and analyze explained variance.
2. Implement **LDA** on a multi-class dataset and compare class separability with PCA.
3. Visualize Word2Vec embeddings with **t-SNE**. Experiment with **perplexity** and **learning rate**.
4. Use **UMAP** for single-cell RNA-seq dataset; observe local/global structure preservation.
5. Combine **PCA + t-SNE** and **PCA + UMAP**; compare speed and cluster visualization.
6. Interpret eigenvectors in PCA and LDA in terms of **feature importance**.
7. Analyze stability of t-SNE/UMAP projections across multiple runs.
8. Discuss scenarios where **linear methods** fail and **non-linear methods** are necessary.

---

✅ **Next Topic:**  
📘 *802. Feature Selection vs. Feature Extraction*
