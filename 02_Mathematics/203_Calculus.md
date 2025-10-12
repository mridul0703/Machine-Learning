# 203. Calculus for Machine Learning
### (Derivatives, Partial Derivatives, Gradients, Chain Rule, Jacobians, Hessians)

---

## 🧩 1. Introduction

Calculus plays a **critical role** in Machine Learning, especially in **optimization** — where we minimize errors and adjust model parameters.

In ML, calculus helps us:
- Optimize cost/loss functions.  
- Understand how models learn through **gradients** and **backpropagation**.  
- Analyze sensitivity and rate of change in parameters.

---

## 🔹 2. Basic Concepts

### **2.1 Function**
A **function** maps input(s) to an output:
```text
y = f(x)
```
Example:  
If f(x) = x², then f(3) = 9.

---

### **2.2 Derivative (Single Variable)**

The derivative measures the **rate of change** of a function with respect to its input variable.

Mathematically:
f'(x) = lim(Δx → 0) [f(x + Δx) - f(x)] / Δx

Example:  
If f(x) = x², then f'(x) = 2x

**Interpretation:**  
- The derivative represents the **slope** or gradient at a point.  
- In ML, it tells us how a **change in a parameter** affects the output.

---

## 🔹 3. Partial Derivatives

When a function depends on multiple variables:
z = f(x, y)
we compute the rate of change with respect to **one variable at a time**, keeping others constant.

Example:
f(x, y) = x² + 3y  
∂f/∂x = 2x,   ∂f/∂y = 3

**Use in ML:**  
Each weight or bias in a model is a variable. Partial derivatives show how **each parameter** affects the loss.

---

## 🔹 4. Gradient

The **gradient** is a **vector** of all partial derivatives of a multivariable function.

∇f(x, y, z) = [ ∂f/∂x, ∂f/∂y, ∂f/∂z ]

It points in the **direction of steepest increase** of the function.

**In ML:**
- We move **opposite to the gradient** (downhill) to minimize loss — called **Gradient Descent**.

---

## 🔹 5. Chain Rule

Used when functions are **nested** — helps compute derivatives of compositions.

dy/dx = (dy/du) * (du/dx)

Example:  
If y = (3x + 1)², then dy/dx = 2(3x + 1) * 3 = 6(3x + 1)

**In ML:**  
Used extensively in **backpropagation** — propagating gradients backward through network layers.

---

## 🔹 6. Jacobian Matrix

The **Jacobian** is a matrix containing all **first-order partial derivatives** of a vector-valued function.

If:  
y = f(x) = [f₁(x₁, x₂), f₂(x₁, x₂)]

Then:  
J = [[∂f₁/∂x₁, ∂f₁/∂x₂], [∂f₂/∂x₁, ∂f₂/∂x₂]]

**In ML:**  
Used in transformations (e.g., data normalization, backpropagation in vector form, and advanced techniques like normalizing flows).

---

## 🔹 7. Hessian Matrix

The **Hessian** is a **square matrix of second-order partial derivatives**.

For a function f(x, y):  
H = [[∂²f/∂x², ∂²f/∂x∂y], [∂²f/∂y∂x, ∂²f/∂y²]]

**In ML:**
- Used in **second-order optimization algorithms** (e.g., Newton’s Method).  
- Provides curvature information — whether the surface is convex or not.

---

## 🔹 8. Applications in Machine Learning

| Concept | Application |
|----------|--------------|
| Derivatives | Measure rate of change of loss functions |
| Partial Derivatives | Compute sensitivity of loss wrt. individual parameters |
| Gradients | Used in **Gradient Descent** for optimization |
| Chain Rule | Core of **Backpropagation** in neural networks |
| Jacobian | Describes transformations between variables in vectorized models |
| Hessian | Used in **advanced optimization** (Newton, quasi-Newton methods) |

---

## 🧮 9. Exercises

1. Find the derivative of f(x) = 3x³ + 2x² - 4x + 7  
2. Compute ∂f/∂x and ∂f/∂y for f(x, y) = x²y + 3y²  
3. Write the gradient vector for f(x, y, z) = x² + y² + z²  
4. Apply the Chain Rule for f(x) = sin(3x² + 2)  
5. Explain how the Hessian helps in identifying local minima in optimization problems.

---

## 🧾 Summary

- Calculus helps ML models **learn** by computing **how outputs change** with inputs.  
- **Gradients** and **partial derivatives** drive parameter updates.  
- **Jacobian** and **Hessian** matrices extend these ideas to higher dimensions.  
- Understanding calculus builds intuition for **optimization and backpropagation**.
