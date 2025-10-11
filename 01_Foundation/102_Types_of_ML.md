
# 102. Types of Machine Learning

## 📖 Introduction

Machine Learning is not a single technique — it includes several **types of learning approaches**, depending on the nature of data and the task to be performed.

The four main types are:
1. **Supervised Learning**
2. **Unsupervised Learning**
3. **Semi-Supervised Learning**
4. **Reinforcement Learning**

---

## 🎯 1. Supervised Learning

### 📌 Definition
Supervised learning is when we train a model using **labeled data**, meaning each training example has an **input** and the correct **output** (target).  
The goal is to learn a mapping function from inputs (X) to outputs (Y).

> 🧩 Example: Given past data of house prices (features like area, location) and their prices (labels), predict the price of a new house.

### 📊 Applications
- Spam Email Detection  
- Predicting House Prices  
- Sentiment Analysis  
- Disease Diagnosis

### 🔢 Algorithms
- Linear Regression  
- Logistic Regression  
- Decision Trees  
- Support Vector Machines (SVM)  
- Random Forest  
- Neural Networks

---

## 🔍 2. Unsupervised Learning

### 📌 Definition
Unsupervised learning is when we train a model on **unlabeled data** — data without any target outputs.  
The model tries to **find hidden patterns, groupings, or structures** in the data.

> 🧩 Example: Grouping customers into segments based on their purchase behavior.

### 📊 Applications
- Customer Segmentation  
- Market Basket Analysis  
- Topic Modeling  
- Dimensionality Reduction

### 🔢 Algorithms
- K-Means Clustering  
- Hierarchical Clustering  
- DBSCAN  
- Principal Component Analysis (PCA)

---

## ⚖️ 3. Semi-Supervised Learning

### 📌 Definition
Semi-supervised learning is a **hybrid approach** — it uses a small amount of **labeled data** and a large amount of **unlabeled data**.  
It’s helpful when labeling data is expensive or time-consuming.

> 🧩 Example: A few labeled medical images and many unlabeled ones can be used together to train a model.

### 📊 Applications
- Web Content Classification  
- Speech Recognition  
- Medical Image Analysis

### 🔢 Common Techniques
- Self-training  
- Label Propagation  
- Graph-based Semi-Supervised Learning

---

## 🕹️ 4. Reinforcement Learning

### 📌 Definition
Reinforcement Learning (RL) is a learning approach where an **agent** learns by **interacting with an environment**.  
It receives **rewards** or **penalties** based on its actions and aims to **maximize the total reward**.

> 🧩 Example: A robot learning to walk or a game AI learning to play chess.

### 📊 Key Components
- **Agent:** The learner or decision-maker  
- **Environment:** Where the agent acts  
- **Action:** What the agent can do  
- **Reward:** Feedback from the environment

### 📊 Applications
- Game Playing (Chess, Go, Atari)  
- Robotics Control  
- Self-Driving Cars  
- Recommendation Systems

### 🔢 Algorithms
- Q-Learning  
- Deep Q-Networks (DQN)  
- Policy Gradient Methods  
- Actor-Critic Models

---

## 🧩 Comparison Table

| Learning Type | Data Type | Labeled? | Goal | Example |
|:--|:--|:--|:--|:--|
| **Supervised** | Input + Output | ✅ Yes | Predict output for new data | Predict house price |
| **Unsupervised** | Input only | ❌ No | Discover hidden patterns | Customer segmentation |
| **Semi-Supervised** | Input + Partial Output | ⚙️ Partial | Combine labeled & unlabeled learning | Medical image classification |
| **Reinforcement** | Sequential Data | 🚀 Reward-based | Learn through trial and error | Training a robot |

---

## 🧠 Key Takeaways

- **Supervised Learning:** Learns from labeled data to predict outcomes.  
- **Unsupervised Learning:** Finds hidden patterns in unlabeled data.  
- **Semi-Supervised Learning:** Combines small labeled data with large unlabeled data.  
- **Reinforcement Learning:** Learns through interaction and feedback.  

> ✅ **Next Topic:** "Traditional Programming vs. Machine Learning" — Understanding how ML differs from rule-based programming.

---

### 📚 Recommended Readings
- [Google Developers: ML Types Overview](https://developers.google.com/machine-learning)  
- [DeepMind: Reinforcement Learning Explained](https://deepmind.com/learning-resources)  
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* – Aurélien Géron (Book)
