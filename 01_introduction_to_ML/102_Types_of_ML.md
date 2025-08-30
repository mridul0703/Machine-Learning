# 102. Types of Machine Learning

## 🧩 Overview

Machine Learning (ML) can be broadly categorized into three main types based on the kind of data available and how the system learns:

- **Supervised Learning**
- **Unsupervised Learning**
- **Reinforcement Learning**

These types define how models are trained, what data they use, and what tasks they solve.

---

## 🎯 Supervised Learning
### 🔧 Definition

Supervised Learning is when a model learns from labeled data — data that already contains both the input (X) and the expected output (Y).

The algorithm’s task: Learn the mapping  
**f:** X → Y

### 🛠️ Examples

- Predicting house prices based on features (size, location).  
- Email spam classification (spam/ham).  
- Predicting whether a customer will churn.

### 📌 Key Points

- Needs large labeled datasets.  
- Most common type of ML in industry.  
- Used for regression (continuous values) and classification (discrete labels).

---

## 🎯 Unsupervised Learning
### 🔧 Definition

Unsupervised Learning is when a model learns from unlabeled data — only input (X) is provided, no outputs (Y).

The algorithm’s task: Find patterns, structures, or groups in the data.

### 🛠️ Examples

- Customer segmentation (grouping buyers by behavior).  
- Topic modeling in documents.  
- Anomaly detection (fraud, system failures).

### 📌 Key Points

- Useful for exploratory analysis.  
- No labels required (cheaper data collection).  
- Often used for clustering and dimensionality reduction.

---

## 🎯 Reinforcement Learning
### 🔧 Definition

Reinforcement Learning (RL) is when an agent learns by interacting with an environment, making decisions, and receiving rewards or penalties as feedback.

The algorithm’s task: Maximize cumulative reward over time.

### 🛠️ Examples

- Game-playing AI (Chess, Go, Atari).  
- Self-driving cars (navigation).  
- Robotics (teaching robots to walk, pick objects).

### 📌 Key Points

- No direct supervision, only feedback (reward/penalty).  
- Useful for sequential decision-making problems.  
- Can combine with deep learning → Deep Reinforcement Learning (DRL).

---

## 🧠 Comparison Table

| Type | Data Used | Goal | Example |
| --- | --- | --- | --- |
| Supervised Learning | Labeled data (X, Y) | Learn mapping from input → output | Spam detection, price prediction |
| Unsupervised Learning | Unlabeled data (X) | Find hidden patterns, clusters | Customer segmentation |
| Reinforcement Learning | Feedback (reward/penalty) | Learn best actions through trial & error | Self-driving cars, game AI |

---

## 🚀 Workflow Summary

- **Supervised Learning** → Learn with teacher (labeled data).  
- **Unsupervised Learning** → Learn without teacher (discover hidden structure).  
- **Reinforcement Learning** → Learn by doing (trial, error, rewards).

---

## 📘 Key Takeaways

- **Supervised Learning** → Prediction from labeled data.  
- **Unsupervised Learning** → Pattern discovery in unlabeled data.  
- **Reinforcement Learning** → Learning strategies through trial and reward.

👉 These three pillars form the foundation of ML techniques you’ll explore in the rest of the course.
