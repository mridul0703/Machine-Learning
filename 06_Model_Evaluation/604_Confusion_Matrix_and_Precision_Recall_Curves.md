# 604. Confusion Matrix & Precision–Recall Curves

---

## 🧩 1. Introduction

Accuracy alone doesn’t always tell the full story of a model’s performance — especially in **imbalanced datasets** (e.g., fraud detection, disease diagnosis).

For example:
- A model that predicts “No Fraud” for all transactions may achieve 99% accuracy if fraud is rare.
- But it completely fails to identify the fraudulent cases.

To gain a **deeper understanding**, we use:
- **Confusion Matrix** – Summarizes prediction results.
- **Precision–Recall Curves** – Illustrate the trade-off between precision and recall.

---

## 🔹 2. Confusion Matrix

A **Confusion Matrix** is a table that describes the performance of a classification model.

|               | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Example
If we test 100 patients:
- 70 are healthy  
- 30 have a disease  

Our model predicts:
- 25 of 30 sick people correctly → **TP = 25**
- 5 sick people missed → **FN = 5**
- 10 healthy people misclassified → **FP = 10**
- 60 healthy correctly predicted → **TN = 60**

|               | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | 25 | 5 |
| **Actual Negative** | 10 | 60 |

---

## 🔹 3. Derived Metrics from Confusion Matrix

| Metric | Formula | Description |
|---------|----------|-------------|
| **Accuracy** | (TP + TN) / (TP + FP + TN + FN) | Overall correctness |
| **Precision (PPV)** | TP / (TP + FP) | How many predicted positives are actually correct |
| **Recall (Sensitivity)** | TP / (TP + FN) | How many actual positives are captured |
| **Specificity** | TN / (TN + FP) | How many actual negatives are correctly predicted |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balance between Precision and Recall |
| **False Positive Rate (FPR)** | FP / (FP + TN) | Proportion of negatives incorrectly classified as positives |

---

## 🔹 4. Visual Intuition
```scss
           ┌──────────────────────────────────────────┐
           │                 Confusion Matrix          │
           ├──────────────────────┬────────────────────┤
           │      TP (Correct +)  │  FN (Missed +)     │
           ├──────────────────────┼────────────────────┤
           │      FP (Wrong +)    │  TN (Correct -)    │
           └──────────────────────┴────────────────────┘
```

A **good model** has high TP and TN values and low FP, FN counts.

---

## 🔹 5. Precision–Recall Relationship

Precision and Recall often have an **inverse relationship**:

- Increasing the **decision threshold** makes the model **more precise** but **less sensitive**.
- Lowering the threshold increases **recall** but may reduce **precision**.

| Scenario | Precision | Recall | Comment |
|-----------|------------|--------|----------|
| Strict threshold (0.9) | High | Low | Model is cautious |
| Loose threshold (0.3) | Low | High | Model captures more positives but with more false alarms |

---

## 🔹 6. Precision–Recall Curve

The **Precision–Recall (PR) Curve** plots:
- **Precision** on the Y-axis  
- **Recall** on the X-axis  
for different classification thresholds.

### Interpretation:
- Each point represents a threshold.  
- A model that maintains **high precision and high recall** simultaneously performs best.  
- The **area under the PR curve (AUC-PR)** summarizes performance.

---

### 🔸 PR Curve vs. ROC Curve

| Aspect | PR Curve | ROC Curve |
|---------|-----------|-----------|
| Focus | Positive class performance | Both classes |
| X-axis | Recall | False Positive Rate |
| Useful when | Dataset is imbalanced | Dataset is balanced |
| Metric | AUC-PR | AUC-ROC |

✅ For **imbalanced datasets**, **PR Curve** is a more reliable metric.

---

## 🔹 7. Average Precision (AP)

Average Precision (AP) summarizes the **shape of the precision–recall curve** by averaging precision values over recall thresholds.

```math
AP = \sum (R_n - R_{n-1}) \times P_n
```

- Often used in **object detection** and **ranking tasks**.
- **Higher AP = better model**.

---

## 🔹 8. Use Cases

| Application | Metric Focus |
|--------------|---------------|
| Medical Diagnostics | High Recall (avoid false negatives) |
| Spam Detection | High Precision (avoid false positives) |
| Fraud Detection | High Recall (detect all fraud cases) |
| Search Engines | High Precision (show only relevant results) |
| Object Detection (e.g., YOLO, SSD) | Average Precision (AP) used for evaluation |

---

## 🧾 9. Summary

| Concept | Key Takeaway |
|----------|---------------|
| **Confusion Matrix** | Foundation for all classification metrics |
| **Precision & Recall** | Measure accuracy and completeness of predictions |
| **F1-Score** | Balances precision and recall |
| **PR Curve** | Visualizes trade-offs |
| **Average Precision (AP)** | Quantifies PR curve in one number |

---

## 🧩 10. Exercises

1. Compute and visualize a **confusion matrix** for a binary classification problem.  
2. Derive **Precision, Recall, and F1-score** from your confusion matrix.  
3. Plot a **Precision–Recall curve** using scikit-learn and interpret it.  
4. Compare **ROC-AUC vs. PR-AUC** on an imbalanced dataset.  
5. Discuss which metric you’d prioritize for:
   - Cancer detection  
   - Email spam filtering  
   - Credit card fraud detection  

---

✅ **Next Topic:**  
📘 *605. ROC Curve and AUC (Receiver Operating Characteristic)*

