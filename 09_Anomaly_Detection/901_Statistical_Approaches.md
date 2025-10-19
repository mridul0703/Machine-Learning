# 901. Anomaly Detection: Statistical Approaches

Anomaly detection (or outlier detection) is the process of identifying data points that **deviate significantly** from the majority of the data.  
Statistical approaches are the most fundamental and provide a strong foundation for understanding more advanced techniques.

---

## 🔹 1. What is Anomaly Detection?

- **Anomalies / Outliers**: Observations that differ significantly from the expected pattern.  
- **Applications**: Fraud detection, network intrusion detection, industrial equipment monitoring, medical diagnosis.

---

## 🔹 2. Types of Anomalies

| Type | Description | Example |
|------|-------------|---------|
| Point anomaly | Single data point differs from the rest | A transaction of $10,000 in a normal $10–$50 range |
| Contextual anomaly | Abnormal given a context | Temperature spike in winter |
| Collective anomaly | A group of data points is anomalous together | Unusual pattern of server requests |

---

## 🔹 3. Statistical Approaches Overview

Statistical approaches assume that **normal data follows a known distribution**. Anomalies are detected as points with **low probability** under this distribution.

1. **Z-Score / Standard Score**
2. **Modified Z-Score**
3. **Gaussian Distribution**
4. **Chi-Square Distribution**
5. **Boxplot / IQR Method**

---

## 🔹 4. Z-Score Method

- Measures **how many standard deviations** a point is from the mean.  

**Formula:**
```math
Z = \frac{X_i - \mu}{\sigma}
```

Where:  
- $(X_i\)$ = data point  
- $(\mu\)$ = mean of the dataset  
- $(\sigma\)$ = standard deviation  

**Detection Rule:**  
- If $(|Z| > 3\ )$, mark as anomaly (common threshold).

**Python Example:**
```python
import numpy as np

data = [10, 12, 12, 13, 12, 100]  # 100 is an anomaly
mean = np.mean(data)
std = np.std(data)
z_scores = [(x - mean)/std for x in data]
anomalies = [x for x, z in zip(data, z_scores) if abs(z) > 3]
print(anomalies)
```
---

## 🔹 5. Modified Z-Score

- More robust to outliers using **median and MAD (Median Absolute Deviation)**.  
```math
M_i = \frac{0.6745 (X_i - \text{median})}{\text{MAD}}
```

- Threshold typically $(|M_i| > 3.5\ )$

---

## 🔹 6. Gaussian Distribution-Based Detection

- Fit a Gaussian (Normal) distribution to your data.  
- Compute **probability density** for each point.  
- Points with very low probability are anomalies.

**Formula:**
```math
P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
```

- Can extend to **multivariate Gaussian** for multiple features.

---

## 🔹 7. Chi-Square Distribution

- For **categorical or count data**.  
- Measures deviation from expected counts.  
- Commonly used in statistical process control.

---

## 🔹 8. Boxplot / IQR Method

- Uses **interquartile range (IQR)** to detect outliers.

**Formula:**
```math
\text{IQR} = Q3 - Q1
```
```math
\text{Lower Bound} = Q1 - 1.5 \cdot IQR
```
```math
\text{Upper Bound} = Q3 + 1.5 \cdot IQR
```

- Points outside bounds are considered anomalies.

**Python Example:**
```python
import numpy as np

data = [10, 12, 12, 13, 12, 100]
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
anomalies = [x for x in data if x < lower or x > upper]
print(anomalies)
```

---

## 🔹 9. Advantages and Limitations

| Method | Advantages | Limitations |
|--------|-----------|-------------|
| Z-Score | Simple, interpretable | Sensitive to extreme outliers |
| Modified Z-Score | Robust to outliers | Requires median & MAD |
| Gaussian | Works for continuous features | Assumes normality |
| Chi-Square | Good for categorical/count data | Not suitable for continuous variables |
| Boxplot/IQR | Non-parametric, simple | Works best for univariate analysis |

---

## 🔹 10. Practical Tips

- Always **visualize your data** before applying statistical methods.  
- Choose method depending on **data type and distribution**.  
- For multivariate data, consider **Mahalanobis distance** or advanced methods (Isolation Forest, Autoencoders).

---

## 🔹 11. Exercises

1. Identify anomalies using Z-Score in a synthetic dataset.  
2. Apply Modified Z-Score to a dataset with extreme outliers.  
3. Fit a Gaussian distribution and detect anomalies in 2D feature space.  
4. Use IQR method to find anomalies in numeric features of a real dataset.  
5. Compare performance of Z-Score vs IQR on a dataset with skewed distribution.

---

✅ **Next Topic:**  
📘 *902. Isolation Forests & Ensemble-Based Anomaly Detection*
