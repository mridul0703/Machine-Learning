# 105. Linear Algebra for Machine Learning

## 🔢 Why Linear Algebra for ML?

Linear Algebra is the **mathematical foundation** of most Machine Learning algorithms.  
It provides the language to represent data, perform transformations, and solve optimization problems.

- Images → matrices of pixels  
- Text embeddings → vectors in high-dimensional space  
- Neural networks → matrix multiplications  

Without Linear Algebra, ML models cannot exist.

---

## 🧭 Key Topics Covered
1. Vectors  
2. Matrices  
3. Dot Product & Matrix Multiplication  
4. Eigenvalues & Eigenvectors  
5. Applications in Machine Learning  

---

## 🟢 1. Vectors

- A **vector** is an ordered list of numbers (1D array).  
- Represents points, directions, or features in space.  

**Example:**  
x = [2, 5, 7]

markdown
Copy code
Here, `x` is a 3-dimensional vector.  

### Vector Operations
- **Addition**: `[1, 2] + [3, 4] = [4, 6]`  
- **Scalar Multiplication**: `2 × [1, 3] = [2, 6]`  
- **Magnitude (Length)**: `|x| = sqrt(x1² + x2² + ... + xn²)`

---

## 🟡 2. Matrices

- A **matrix** is a rectangular array of numbers arranged in rows and columns.  
- Denoted as **M x N** → (M rows, N columns).  

**Example (2x2 Matrix):**
A = [[2, 0],
[0, 3]]

### Matrix Operations
- **Addition**: Add corresponding elements  
- **Scalar Multiplication**: Multiply each element  
- **Transpose (Aᵀ)**: Swap rows and columns  

---

## 🔴 3. Dot Product & Matrix Multiplication

### Dot Product (Vector • Vector)
a = [a1, a2, a3]
b = [b1, b2, b3]

a · b = a1b1 + a2b2 + a3*b3

**Example:**
[1, 2, 3] · [4, 5, 6] = 14 + 25 + 3*6 = 32

🔹 Interpretation:  
- Large positive → vectors point in similar direction  
- Zero → vectors are orthogonal (perpendicular)  
- Negative → vectors point in opposite directions  

### Matrix Multiplication
C = A × B

Rule: Columns of `A` must equal rows of `B`.

**Example:**
A = [[1, 2],
[3, 4]]

B = [[5, 6],
[7, 8]]

A × B = [[19, 22],
[43, 50]]
---

## 🔵 4. Eigenvalues & Eigenvectors

- An **eigenvector** of a matrix is a vector that only changes in scale (not direction) after multiplication.  
- The **eigenvalue** is the scale factor.  

Mathematically:
A · v = λ · v

Where:  
- `A` → matrix  
- `v` → eigenvector  
- `λ` → eigenvalue  

**Example:**  
If
A = [[2, 0],
[0, 3]]

Then eigenvectors = axis directions, and eigenvalues = {2, 3}.

---

## 🚀 5. Applications in Machine Learning

- **Dimensionality Reduction (PCA)** → uses eigenvectors/eigenvalues to project data into fewer dimensions.  
- **Word Embeddings** → represent words as vectors in NLP.  
- **Neural Networks** → heavy use of matrix multiplications.  
- **Computer Vision** → images as matrices of pixel values.  
- **Recommendation Systems** → factorization using matrix algebra.  

---

## 📌 Summary

- Vectors → represent data/features  
- Matrices → store datasets & transformations  
- Dot Product → measures similarity  
- Eigenvalues/Eigenvectors → reveal key directions of variance  
- Linear Algebra is **the backbone** of Machine Learning 🚀  

---
