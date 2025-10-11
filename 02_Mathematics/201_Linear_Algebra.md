# 201. Linear Algebra for Machine Learning

Linear Algebra forms the **mathematical backbone** of Machine Learning and Deep Learning. It provides the tools to represent and manipulate data efficiently in the form of **vectors** and **matrices**, which are central to every ML algorithm.

---

## 🧩 1. Introduction

In Machine Learning, data is often represented as **numerical arrays**. Whether it’s features in a dataset, weights in a neural network, or embeddings in NLP — they’re all **vectors or matrices**.

Understanding Linear Algebra allows us to:
- Represent large datasets efficiently.
- Perform transformations (rotation, scaling, projection) on data.
- Understand models like PCA, SVD, and Neural Networks.

---

## 🔹 2. Scalars, Vectors, Matrices, and Tensors

| Concept | Symbol | Example | Description |
|----------|--------|----------|--------------|
| Scalar | x | 5 | A single number (1D) |
| Vector | v | [2, 3, 5] | 1D array of numbers |
| Matrix | A | [[1, 2], [3, 4]] | 2D array of numbers |
| Tensor | T | 3D/4D array | Multi-dimensional generalization of matrices |

**In ML:**  
- A **vector** can represent a data point (features).  
- A **matrix** can represent a dataset or model parameters.  
- A **tensor** is used in Deep Learning for multi-dimensional data (e.g., images).

---

## 🔹 3. Vector Operations

### Vector Addition & Scalar Multiplication
If a = [1, 2] and b = [3, 4]:
```text
a + b = [1+3, 2+4] = [4, 6]  
2 * a = [2, 4]
```
### Dot Product
Dot product of vectors a and b:
```
a · b = a1*b1 + a2*b2 + ... + an*bn  
[1, 2] · [3, 4] = 1*3 + 2*4 = 11
```

**Geometric meaning:** measures how much two vectors point in the same direction.

### Vector Norm (Length)
- ||a|| = sqrt(a1² + a2² + ... + an²)

---

## 🔹 4. Matrix Basics

A **matrix** is a 2D grid of numbers with rows (m) and columns (n).  
- Notation: A ∈ R^(m x n)  
- Example:  
```yaml
A = [[1, 2, 3], [4, 5, 6]]
```
This is a 2x3 matrix.

---

## 🔹 5. Matrix Operations

### Matrix Addition
- A + B = element-wise sum (only if same dimensions)  
- Example:  
```
[[1, 2], [3, 4]] + [[5, 6], [7, 8]] = [[6, 8], [10, 12]]
```

### Matrix Multiplication
- (AB)_ij = sum_k (A_ik * B_kj)  
- Example:  
```
A = [[1, 2], [3, 4]]
B = [[2, 0], [1, 2]]
AB = [[4, 4], [10, 8]]
```

### Transpose
- Flips rows and columns:  
```
A = [[1, 2], [3, 4]]
A^T = [[1, 3], [2, 4]]
```

### Determinant
- For a 2x2 matrix [[a, b], [c, d]]:
```
det(A) = a*d - b*c
```

### Inverse
- A^-1 exists only if det(A) ≠ 0  
- A * A^-1 = Identity Matrix

---

## 🔹 6. Special Matrices

| Type             | Property                                   | Example                  |
|-----------------|--------------------------------------------|--------------------------|
| Identity        | AI = A                                     | [[1,0],[0,1]]           |
| Diagonal        | Non-zero only on diagonal                  | [[5,0],[0,3]]           |
| Symmetric       | A = A^T                                    | [[2,3],[3,4]]           |
| Skew-Symmetric  | A = -A^T                                   | [[0,2],[-2,0]]          |
| Orthogonal      | A^T * A = I                                | Rotation matrices        |

---

### 🔹 6.1 Properties of Symmetric Matrices
1. All **eigenvalues** are real.
2. Eigenvectors corresponding to distinct eigenvalues are **orthogonal**.
3. Can be **diagonalized** by an orthogonal matrix:  
```
A = QΛQ^T
```
4. All **principal minors** are real.
5. Appears in **PCA, Covariance matrices, Quadratic forms**.

---

### 🔹 6.2 Properties of Skew-Symmetric Matrices
1. All **diagonal elements** are zero (a_ii = 0).
2. Eigenvalues are either **0** or **purely imaginary** (complex numbers).
3. Determinant is **non-negative** for odd-sized matrices: det(A) = 0 for odd n.
4. Can represent **rotations** in 2D and 3D space.
5. Useful in **Lie algebras, cross product matrices**, and physics applications.


## 🔹 7. Eigenvalues & Eigenvectors

**Definition:**  
- For a square matrix A:  
```
A * v = λ * v
```
- v = eigenvector  
- λ = eigenvalue  

**Meaning:** multiplying A by v only **scales** v without changing its direction.

**Finding eigenvalues:**  
- Solve det(A - λI) = 0

**Example:**  
```
A = [[2,0],[0,3]] => eigenvalues: 2, 3
```

**Geometric meaning:**  
- Eigenvectors = directions where transformation acts by stretching/compressing  
- Eigenvalues = scale factor in those directions

### 🔹 7.1 Properties of Eigenvalues & Eigenvectors

- A matrix can have **up to n eigenvalues** (for an n x n matrix), which may be repeated.  
- Eigenvectors corresponding to **distinct eigenvalues** are **linearly independent**.  
- The **trace** of a matrix = sum of its eigenvalues.  
- The **determinant** of a matrix = product of its eigenvalues.  
- If the matrix is **symmetric**, all eigenvalues are **real** and eigenvectors are **orthogonal**.  
- Scaling a matrix scales its eigenvalues:  

```text
λ(new) = c * λ(original)   if A_new = c * A
```
### 🔹 7.2 Uses of Eigenvalues & Eigenvectors in Machine Learning

| Use Case | Description |
|----------|-------------|
| Principal Component Analysis (PCA) | Eigenvectors of the covariance matrix represent **principal directions**; eigenvalues indicate **variance explained** along each component |
| Dimensionality Reduction | Select eigenvectors with **largest eigenvalues** to reduce feature space while retaining variance |
| Spectral Clustering | Uses eigenvectors of the **graph Laplacian** for clustering based on connectivity |
| Covariance Analysis | Eigen decomposition helps **understand correlations** among features |
| Linear Transformations | Eigenvectors define **invariant directions** under transformations (rotation, scaling) |
| Stability Analysis | Eigenvalues of weight matrices help analyze **stability of recurrent networks** |
| Recommendation Systems & Graph ML | Eigen decomposition of adjacency/similarity matrices identifies **latent factors** for embeddings |

---

## 🔹 8. Applications of Linear Algebra in ML

| Concept | ML Use Case |
|---------|-------------|
| Vectors | Represent data, weights, gradients |
| Dot Product | Similarity in embeddings (Word2Vec, cosine similarity) |
| Matrix Multiplication | Neural network feed-forward operations |
| Eigenvalues | PCA for dimensionality reduction |
| Determinant/Inverse | Linear regression solutions |

---

## 🧾 9. Summary

- Linear Algebra is foundational to ML.  
- Vectors and matrices represent both data and transformations.  
- Eigenvalues/eigenvectors reveal variance directions in data (used in PCA).  
- Geometric understanding builds strong ML intuition.

---

## 🧮 10. Exercises

1. Compute dot product: [1, 2, 3] · [4, 5, 6]  
2. Find determinant and inverse of [[2,1],[5,3]]  
3. For A = [[3,1],[0,2]], find eigenvalues and eigenvectors  
4. Explain how matrix multiplication is used in neural networks  
5. How do eigenvalues help in PCA for dimensionality reduction?

---

### ✅ Next Topic:  
📘 Probability and Statistics for Machine Learning*
