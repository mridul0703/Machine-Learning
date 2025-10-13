
# 204. Optimization for Machine Learning

Optimization lies at the heart of machine learning — it’s how models learn from data by minimizing loss functions.

---

## 🔹 1. What is Optimization?

**Definition:**  
Optimization is the process of adjusting parameters of a model to minimize (or maximize) an objective function — usually the **loss function**.

```math
\theta^* = \arg\min_{\theta} J(\theta)
```

where:
- &theta; → model parameters  
- J(&theta;) → loss/cost function


---

## 🔹 2. Convex vs. Non-Convex Functions

| Property | Convex Function | Non-Convex Function |
|-----------|------------------|--------------------|
| Shape | Bowl-shaped (single minimum) | Can have multiple minima/maxima |
| Example | f(x) = x² | f(x) = x⁴ - 3x³ + 2 |
| Local Minima | Only one (global) | Multiple local minima |
| Optimization | Easier to solve | Harder to find global minimum |

**Convexity Check:**  
A function f(x) is **convex** if:
```math
f(\alpha x_1 + (1-\alpha)x_2) \leq \alpha f(x_1) + (1-\alpha)f(x_2)
```
for all x₁, x₂ and 0 ≤ α ≤ 1.

---

## 🔹 3. Gradient Descent (GD)

**Idea:**  
Gradient Descent is an iterative optimization algorithm that moves in the **direction opposite to the gradient** of the loss function to minimize it.

```math
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta_t)
```

where:  
- &eta; = learning rate  
- &nabla;<sub>&theta;</sub> J(&theta;<sub>t</sub>) = gradient of loss at step t


**Intuition:**  
The gradient points uphill — moving in the opposite direction brings us downhill toward a minimum.

---

### 🔸 Step-by-Step Process
1. Initialize model parameters θ randomly.  
2. Compute the gradient of loss with respect to parameters.  
3. Update parameters using the gradient.  
4. Repeat until convergence (loss stops decreasing).

---

## 🔹 4. Stochastic Gradient Descent (SGD)

**Problem with Batch GD:**  
Using the entire dataset per update is slow for large data.

**SGD Solution:**  
Use one (or a few) training examples at each iteration.

```math
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta_t; x_i, y_i)
```

**Advantages:**  
- Faster updates  
- Works well with streaming data  
- Helps escape local minima  

**Drawbacks:**  
- More noise in updates → fluctuating convergence  

---

## 🔹 5. Mini-Batch Gradient Descent

A hybrid approach — processes **mini-batches** (e.g., 32 or 128 samples per update).  
Balances stability and speed.

---

## 🔹 6. Momentum

**Problem:**  
SGD may get stuck or oscillate in ravines.

**Solution (Momentum):**  
Incorporates a moving average of past gradients to smooth updates.

```math
v_t = \beta v_{t-1} + (1-\beta) \nabla_\theta J(\theta_t)
```
```math
\theta_{t+1} = \theta_t - \eta v_t
```

where β (e.g., 0.9) controls momentum strength.

**Intuition:**  
Acts like inertia — keeps moving in a consistent direction, avoiding small local minima.

---

## 🔹 7. Adaptive Learning Algorithms

Modern optimizers automatically adjust learning rates based on past gradients.

| Optimizer | Key Idea | Formula/Notes |
|------------|-----------|---------------|
| **Adagrad** | Adapts learning rate per parameter using accumulated gradients | Effective for sparse data but may shrink learning rate too much |
| **RMSProp** | Uses moving average of squared gradients | Fixes Adagrad’s diminishing rate problem |
| **Adam** | Combines Momentum + RMSProp | Most popular; uses bias correction for stability |

**Adam Update Rule:**
```math
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
```
```math
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
```
```math
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
```
```math
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

---

## 🔹 8. Optimization in Machine Learning

| Application | Role of Optimization |
|--------------|----------------------|
| Linear/Logistic Regression | Find coefficients minimizing loss |
| Neural Networks | Train weights via gradient descent |
| SVM | Optimize margin using convex quadratic programming |
| Clustering (K-Means) | Minimize within-cluster distance |
| PCA | Find directions (eigenvectors) maximizing variance |

---

## 🔹 9. Common Challenges

- **Local minima** (non-convex problems)  
- **Vanishing/Exploding gradients** in deep networks  
- **Learning rate tuning**  
- **Overfitting** from too many updates  

---

## 🔹 10. Practical Tips

- Start with **Adam** or **RMSProp**  
- Use **learning rate schedulers**  
- Normalize inputs (helps convergence)  
- Monitor **training/validation loss**  
- Try **gradient clipping** for stability in RNNs  

---
