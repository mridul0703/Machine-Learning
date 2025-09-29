# 203. Calculus for Machine Learning  
*(Derivatives, Partial Derivatives, Gradients, Jacobians, Hessians)*

---

## 🎯 Why Calculus in ML?

Calculus is at the **core of Machine Learning optimization**.  
Models “learn” by minimizing a loss function. To do this, we need to understand:

- How a function changes → **Derivatives**  
- How functions behave with many variables → **Partial Derivatives, Gradients**  
- How to optimize efficiently → **Jacobian, Hessian**  

Without calculus, **algorithms like Gradient Descent, Backpropagation, and Optimization** wouldn’t exist.

---

## 🟢 1. Derivatives (Single Variable)

A **derivative** measures how a function changes as its input changes.  

**Definition:**
```
f'(x) = lim(h→0) [ f(x+h) - f(x) ] / h
```

### Intuition:
- Positive derivative → function increasing  
- Negative derivative → function decreasing  
- Zero derivative → possible maximum/minimum  

**Example:**
```
f(x) = x²
f'(x) = 2x
```
- At `x=2`, slope = `4` (increasing fast).  
- At `x=0`, slope = `0` (flat point).  

---

## 🟡 2. Partial Derivatives (Multi-variable)

In ML, most functions depend on **multiple variables**.  
A **partial derivative** is the rate of change of a function with respect to **one variable**, keeping others constant.

**Notation:**
```
∂f/∂x → partial derivative of f with respect to x
```

**Example:**
```
f(x, y) = x² + y²
∂f/∂x = 2x
∂f/∂y = 2y
```

These are the building blocks for **gradients**.

---

## 🔵 3. Gradient

The **gradient** is a vector of all partial derivatives.  
It points in the direction of the **steepest increase** of a function.

**Definition:**
```
∇f(x,y) = [ ∂f/∂x , ∂f/∂y ]
```

**Example:**
```
f(x, y) = x² + y²
∇f = [2x, 2y]
```

### Importance in ML:
- Used in **Gradient Descent** to minimize loss functions.  
- Gradient tells us *which direction to move* to reduce error.  

---

## 🟠 4. Jacobian (Vector-Valued Functions)

When we have **multiple outputs** depending on **multiple inputs**, the gradient generalizes to the **Jacobian matrix**.

**Definition:**
```
J = [ ∂f_i / ∂x_j ]
```
where `f` = vector function, `x` = vector of inputs.

**Example:**
```
f(x, y) = [ x² + y ,  y² + x ]
Jacobian J =
[ ∂f1/∂x   ∂f1/∂y ]
[ ∂f2/∂x   ∂f2/∂y ]

= [ 2x   1 ]
  [ 1   2y ]
```

### ML Applications:
- Used in **Backpropagation** to compute derivatives across neural networks.  
- Helps optimize **vector-valued loss functions**.  

---

## 🔴 5. Hessian (Second-Order Derivatives)

The **Hessian** is a square matrix of **second-order partial derivatives**.  
It describes the local curvature of a function.  

**Definition:**
```
H = [ ∂²f / ∂xi∂xj ]
```

**Example:**
```
f(x, y) = x² + y²
H =
[ 2   0 ]
[ 0   2 ]
```

### Why it matters:
- If Hessian is **positive definite** → function has a local minimum.  
- If Hessian is **negative definite** → function has a local maximum.  
- Mixed signs → saddle point.  

### ML Applications:
- **Second-order optimization** (Newton’s method, Quasi-Newton methods).  
- Helps analyze **convexity** of loss functions.  

---

## 🚀 6. Applications in Machine Learning

- **Derivatives** → training linear/logistic regression, SVMs  
- **Partial Derivatives** → handling multi-variable loss functions  
- **Gradients** → Gradient Descent, Backpropagation  
- **Jacobian** → Neural networks with multiple outputs, backprop computations  
- **Hessian** → Advanced optimization, curvature analysis  

---

## 📌 Summary

- **Derivatives** → change of function w.r.t one variable  
- **Partial Derivatives** → change in multivariable functions  
- **Gradient** → vector of all partial derivatives (direction of steepest increase)  
- **Jacobian** → matrix for vector-valued functions  
- **Hessian** → curvature information via second derivatives  

Calculus provides the **mathematical machinery that powers optimization in ML** 🚀  
