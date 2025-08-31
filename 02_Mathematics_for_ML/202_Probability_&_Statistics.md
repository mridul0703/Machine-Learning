# 202. Probability & Statistics for Machine Learning  
*(Random Variables, Distributions, Bayes' Theorem, Expectation & Variance)*

---

## 🎯 Why Probability & Statistics in ML?

Machine Learning models often **make predictions under uncertainty**.  
Probability & Statistics provide the tools to **quantify uncertainty**, **model randomness**, and **evaluate outcomes**.

- Probability → Theoretical foundation (what *might* happen).  
- Statistics → Practical estimation from data (what *did* happen).  

---

## 🟢 1. Random Variables

A **Random Variable** is a variable whose value is determined by the outcome of a random process.  

### Types:
- **Discrete Random Variable** → Takes specific values (e.g., number of heads in 10 coin flips).  
- **Continuous Random Variable** → Takes values from a continuous range (e.g., height of a person).  

**Notation Example:** 
```yaml
X = number of heads when flipping 3 coins
Possible values: {0, 1, 2, 3}
```
---

## 🟡 2. Probability Distributions

A **probability distribution** describes how the values of a random variable are spread.  

### Discrete Distributions
- **Bernoulli Distribution**:  
  One trial, outcome = {0,1}. Example: coin toss.  
- **Binomial Distribution**:  
  Number of successes in *n* Bernoulli trials. Example: number of heads in 10 flips.  
- **Poisson Distribution**:  
  Number of events in fixed time/space. Example: emails received per hour.  

### Continuous Distributions
- **Uniform Distribution**:  
  All values equally likely. Example: random number from [0,1].  
- **Normal Distribution (Gaussian)**:  
  Bell-shaped curve, most data near mean. Example: heights, test scores.  
- **Exponential Distribution**:  
  Time between events in a Poisson process. Example: time between bus arrivals.  

**Visual Intuition:**  
- Normal → bell curve  
- Uniform → flat line  
- Binomial → histogram of successes  

---

## 🔵 3. Bayes' Theorem

Bayes’ theorem allows us to **update probabilities** when new information is available.  

**Formula**:
```markdown
P(A|B) = [ P(B|A) * P(A) ] / P(B)
```

Where:  
- `P(A|B)` = Probability of A given B (posterior)  
- `P(B|A)` = Probability of B given A (likelihood)  
- `P(A)` = Prior probability of A  
- `P(B)` = Probability of B (normalizing constant)  

**Example (Medical Diagnosis):**
- 1% of population has a disease → `P(Disease) = 0.01`  
- Test detects disease correctly 99% → `P(Positive|Disease) = 0.99`  
- False positive rate 5% → `P(Positive|NoDisease) = 0.05`  

If a person tests positive, Bayes’ theorem tells us **true chance of having disease** is much lower than 99% (because false positives matter).  

---

## 🔴 4. Expectation & Variance

### Expectation (Mean)
The **expected value** is the weighted average of all possible values of a random variable.  

**Discrete:**  
```makefile
E[X] = Σ [ x * P(X=x) ]
```

**Continuous:**
```csharp
E[X] = ∫ x * f(x) dx
```

### Variance
Variance measures how much values spread out from the mean.  
```shell
Var(X) = E[(X - μ)²]
= E[X²] - (E[X])²
```

### Standard Deviation
```yaml
σ = sqrt(Var(X))
```

---

## 🚀 5. Applications in Machine Learning

- **Random Variables** → Model uncertainty in outcomes (classification, regression).  
- **Distributions** → Assumptions in algorithms (Naive Bayes, Gaussian Mixture Models).  
- **Bayes’ Theorem** → Probabilistic classifiers (Naive Bayes, Bayesian Networks).  
- **Expectation** → Used in loss functions and decision theory.  
- **Variance** → Measures overfitting & uncertainty of predictions.  

---

## 📌 Summary

- Random Variables → represent uncertain quantities  
- Distributions → describe how values occur (Bernoulli, Normal, Poisson, etc.)  
- Bayes’ Theorem → updates beliefs with new evidence  
- Expectation → average outcome  
- Variance → measure of spread/uncertainty  

Probability & Statistics are **the backbone of uncertainty modeling** in ML 🚀  

---
