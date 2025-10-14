# 302. Python Data Libraries for Machine Learning - NumPy & Pandas

Data manipulation and analysis are at the core of machine learning. Python libraries **NumPy** and **Pandas** provide efficient and flexible tools to handle data, perform mathematical computations, and prepare datasets for ML pipelines.

This guide is structured for learners of all levels: **beginners, intermediate, and advanced**.

---

## 🔹 1. Introduction to NumPy

**NumPy (Numerical Python)** is the fundamental package for numerical computations in Python. It provides:

- N-dimensional arrays (`ndarray`)
- Mathematical functions optimized for vectorized operations
- Linear algebra, statistics, and random number capabilities

### 1.1 Why NumPy?
- Fast operations on large datasets (faster than Python lists)
- Supports **vectorized operations** (avoid loops)
- Essential for ML libraries like **Scikit-learn, TensorFlow, PyTorch**

---

## 🔹 2. NumPy Basics

### 2.1 Creating Arrays
```python
import numpy as np

# 1D array
arr1 = np.array([1, 2, 3, 4])

# 2D array
arr2 = np.array([[1, 2], [3, 4]])

# Array of zeros, ones, or random numbers
zeros = np.zeros((2,3))
ones = np.ones((3,2))
rand = np.random.rand(2,2)
```

### 2.2 Array Properties
```python
arr2.shape      # (2,2)
arr2.ndim       # 2
arr2.dtype      # data type
arr2.size       # total number of elements
```

### 2.3 Array Operations
```python
a = np.array([1,2,3])
b = np.array([4,5,6])

# Element-wise operations
a + b  # [5, 7, 9]
a * b  # [4, 10, 18]

# Vectorized functions
np.sqrt(a)       # [1. 1.414 1.732]
np.exp(a)        # [2.718, 7.389, 20.086]
```

## 🔹 3. Advanced NumPy Features

### 3.1 Indexing & Slicing
```python
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])

# Access single element
arr[0,2]  # 3

# Row slicing
arr[1,:]  # [4,5,6]

# Boolean indexing
arr[arr > 5]  # [6,7,8,9]
```

### 3.2 Broadcasting

- Enables arithmetic between arrays of different shapes
```python
a = np.array([1,2,3])
b = 2
a + b  # [3,4,5]
```

### 3.3 Linear Algebra
```python
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

# Matrix multiplication
np.dot(A,B)

# Transpose
A.T

# Determinant and inverse
np.linalg.det(A)
np.linalg.inv(A)
```

### 3.4 Random Numbers
```python
# Seed for reproducibility
np.random.seed(42)

# Random normal distribution
np.random.randn(3,3)

# Random integers
np.random.randint(0,10, size=(2,3))
```
## 🔹 4. Introduction to Pandas

Pandas is a library for data manipulation and analysis, built on top of NumPy. Its key structures:

- Series: 1D labeled array
- DataFrame: 2D labeled tabular data

### 4.1 Why Pandas?

- Handles heterogeneous data (numbers, strings, dates)
- Powerful filtering, grouping, merging
- Essential for data cleaning and preprocessing before ML

## 🔹 5. Pandas Basics
### 5.1 Series
```python
import pandas as pd

# Create series
s = pd.Series([10,20,30], index=['a','b','c'])

# Access data
s['b']  # 20

# Operations
s + 5   # [15,25,35]
```

### 5.2 DataFrames
```python
# Create DataFrame
data = {'Name': ['Alice','Bob','Charlie'],
        'Age': [25,30,35],
        'Salary': [50000,60000,70000]}
df = pd.DataFrame(data)

# View data
df.head()
df.info()
df.describe()
```

## 🔹 6. Data Selection & Filtering
```python
# Select columns
df['Name']
df[['Name','Salary']]

# Select rows by index
df.iloc[0]    # first row
df.iloc[0:2]  # first two rows

# Select rows by condition
df[df['Age'] > 28]

# Reset index
df.reset_index(drop=True)
```

## 🔹 7. Data Cleaning & Preprocessing

- Handling missing values:
```python
df.dropna()             # drop rows with NaN
df.fillna(value=0)      # fill missing values
```

- Encoding categorical variables:
```python
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
```

- Feature scaling:
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Age','Salary']] = scaler.fit_transform(df[['Age','Salary']])
```
## 🔹 8. Grouping, Merging & Aggregation
```python
# Group by
df.groupby('Department')['Salary'].mean()

# Merge / join
df1 = pd.DataFrame({'ID':[1,2],'Name':['Alice','Bob']})
df2 = pd.DataFrame({'ID':[1,2],'Salary':[50000,60000]})
pd.merge(df1, df2, on='ID')
```
## 🔹 9. NumPy & Pandas in Machine Learning
| Library	| Use Case in ML |
|-------------------------------------|--------------|
| NumPy	| Efficient numerical operations, vectorized computations, linear algebra, embeddings, gradient calculations |
| Pandas | Data cleaning, preprocessing, feature engineering, handling structured datasets, loading CSVs/Excel, merging, filtering |

- Workflow Example:

1. Load CSV with Pandas → df = pd.read_csv('data.csv')
2. Clean/Filter Data → df.dropna()
3. Convert DataFrame to NumPy → X = df.values
4. Apply ML algorithms using Scikit-learn or TensorFlow

## 🔹 10. Best Practices

- Prefer NumPy arrays for numerical computations for speed
- Use Pandas DataFrames for structured/tabular data
- Avoid loops; leverage vectorized operations
- Keep reproducibility in mind (set random seeds)
- Document data cleaning steps for ML pipelines

## 🧾 11. Exercises

1. Create a 5x5 NumPy array of random integers between 10 and 50.
2. Compute the mean, sum, and standard deviation along each axis.
3. Load a CSV dataset with Pandas and display first 10 rows.
4. Filter all rows where Age > 30 and Salary < 70000.
5. Group dataset by a categorical column and compute average of numerical columns.
6. Convert a DataFrame column to NumPy array and apply vectorized operation.

## ✅ Summary

- NumPy: Efficient numerical computation, arrays, broadcasting, linear algebra.
- Pandas: Tabular data handling, preprocessing, filtering, aggregation.
- Both libraries are fundamental for machine learning workflows.
- Mastery of NumPy & Pandas is essential for feature engineering and preprocessing.
