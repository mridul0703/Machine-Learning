# 202. Probability & Statistics for Machine Learning

Probability and Statistics provide the **foundation for reasoning under uncertainty** in Machine Learning. They help us understand data, make predictions, and evaluate models effectively.

---

## 🧩 1. Introduction

- Machine Learning often deals with **uncertain data**.  
- Probability and statistics allow us to **quantify uncertainty** and make informed decisions.  
- Applications include:
  - Predicting outcomes (classification, regression)
  - Analyzing model performance
  - Feature selection and hypothesis testing

---

## 🔹 2. Random Variables

- **Definition:** A random variable (RV) is a variable that takes different values **based on chance**.  
- **Types of RVs:**
  - **Discrete RV:** Takes countable values (e.g., number of heads in 10 coin flips)  
  - **Continuous RV:** Takes uncountable values in a range (e.g., height, weight, temperature)

**Example:**  

```text
X = number of heads in 3 coin flips
Possible values: 0, 1, 2, 3
```

---

## 🔹 3. Probability Distributions

### 3.1 Discrete Distributions
- **Bernoulli Distribution:** Success (1) or failure (0)  
- **Binomial Distribution:** Number of successes in n trials  
- **Poisson Distribution:** Number of events in a fixed interval  

### 3.2 Continuous Distributions
- **Uniform Distribution:** Equal probability for all outcomes in range [a, b]  
- **Normal (Gaussian) Distribution:** Bell-shaped curve defined by mean μ and variance σ²  
- **Exponential Distribution:** Time between events in a Poisson process

**Example – Normal Distribution PDF:**  

```
f(x) = (1 / sqrt(2πσ²)) * exp(-(x-μ)² / (2σ²))
```

---

## 🔹 4. Expectation & Variance

- **Expectation (Mean) E[X]:** Average or expected value of RV X  

```
E[X] = Σ x_i * P(X=x_i)      # discrete
E[X] = ∫ x f(x) dx           # continuous
```

- **Variance Var(X):** Measure of spread of X  

```
Var(X) = E[(X - E[X])²]
```

- **Standard Deviation:**  

```
σ = sqrt(Var(X))
```

**Intuition:**  
- High variance → data is spread out  
- Low variance → data is concentrated near the mean

---

## 🔹 5. Bayes’ Theorem

- Allows **updating probability estimates** given new evidence.  

```
P(A|B) = [P(B|A) * P(A)] / P(B)
```

**Example in ML:**  
- Naive Bayes classifier uses Bayes’ theorem to compute the probability of class labels given features.

---

## 🔹 6. Hypothesis Testing

- A method to **test assumptions about data**.  
- **Steps:**
  1. Define null hypothesis H₀ and alternative hypothesis H₁
  2. Choose significance level α (e.g., 0.05)
  3. Compute test statistic (e.g., z-score, t-score)
  4. Calculate p-value
  5. Reject or fail to reject H₀ based on p-value

**Example:**  
- Testing whether a new treatment is better than the existing one using a t-test.  

---

## 🔹 7. Applications in Machine Learning

| Concept | ML Use Case |
|---------|------------|
| Probability Distributions | Model uncertainty, probabilistic models (Naive Bayes, Bayesian Networks) |
| Expectation & Variance | Evaluate model predictions, feature normalization |
| Bayes’ Theorem | Naive Bayes classifier, spam detection, medical diagnosis |
| Hypothesis Testing | Feature selection, A/B testing, evaluating model improvements |

---

## 🧾 8. Summary

- Probability & Statistics help **quantify uncertainty** and make predictions.  
- Random variables and distributions describe how data behaves.  
- Expectation, variance, and standard deviation describe **central tendency and spread**.  
- Bayes’ theorem enables **updating beliefs** with evidence.  
- Hypothesis testing helps **make data-driven decisions** in ML.  

---

## 🧮 9. Exercises

1. Compute the probability of getting **exactly 2 heads in 4 coin flips**.  
2. For a normal distribution with μ = 5, σ = 2, compute **P(X < 7)**.  
3. Use Bayes’ theorem to calculate **P(Disease | Positive Test)** given P(Disease)=0.01, P(Positive|Disease)=0.95, P(Positive|No Disease)=0.05.  
4. Perform a hypothesis test to determine if the **mean of a sample is significantly different** from a given population mean.  
