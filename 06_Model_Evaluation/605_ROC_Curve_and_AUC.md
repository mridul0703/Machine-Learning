# 605. ROC Curve and AUC (Receiver Operating Characteristic)

---

## 🧩 1. Introduction

The **ROC Curve** and **AUC (Area Under Curve)** are crucial tools for evaluating **classification models**, especially when dealing with **imbalanced datasets**.

While the **Confusion Matrix** shows prediction counts, the ROC–AUC provides a **threshold-independent** view of how well the classifier separates classes.

---

## 🔹 2. What is the ROC Curve?

**ROC (Receiver Operating Characteristic)** curve plots:

- **True Positive Rate (TPR)** on the **Y-axis**  
- **False Positive Rate (FPR)** on the **X-axis**

Each point on the curve represents a **different decision threshold** used to classify probabilities into positive/negative classes.

### Formulas:
TPR = TP / (TP + FN)  
FPR = FP / (FP + TN)

| Term | Meaning |
|------|----------|
| **TPR (Recall)** | Fraction of actual positives correctly predicted |
| **FPR** | Fraction of actual negatives incorrectly predicted as positives |

---

## 🔹 3. Intuitive Example

| Threshold | TPR (Recall) | FPR |
|------------|---------------|-----|
| 0.9 | 0.60 | 0.05 |
| 0.7 | 0.75 | 0.10 |
| 0.5 | 0.85 | 0.20 |
| 0.3 | 0.95 | 0.40 |

---

## 🔹 4. ROC Curve Visualization

```
TPR ↑
1.0 |                             • (Ideal model)
    |                           •
    |                        •
    |                     •
0.5 |--------------------•
    |              •
    |         •
    |    •
0.0 |_________________________________________→ FPR
     0.0        0.5                1.0
```

---

## 🔹 5. Area Under the ROC Curve (AUC)

AUC (Area Under the Curve) measures the **total area** under the ROC curve.

0 ≤ AUC ≤ 1

| AUC Value | Interpretation |
|------------|----------------|
| 0.9 – 1.0 | Excellent model |
| 0.8 – 0.9 | Good |
| 0.7 – 0.8 | Fair |
| 0.6 – 0.7 | Poor |
| 0.5 | No discrimination (random guessing) |
| < 0.5 | Worse than random |

---

## 🔹 6. Why ROC–AUC Matters

✅ **Threshold-independent** – evaluates all thresholds  
✅ **Visual trade-off** – between true and false positives  
✅ **Comparison tool** – compare multiple models on the same plot  
✅ **Robustness metric** – not affected by class imbalance as much as accuracy  

---

## 🔹 7. ROC–AUC vs. Precision–Recall (PR) Curve

| Aspect | ROC Curve | PR Curve |
|---------|------------|----------|
| Focus | Both positive and negative classes | Positive class only |
| X-axis | False Positive Rate | Recall |
| Useful when | Classes are balanced | Dataset is imbalanced |
| Metric | AUC (Area under ROC) | AUC-PR (Average Precision) |
| Sensitivity to imbalance | Lower | Higher |

---

## 🔹 8. ROC Curve in Practice

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

---

## 🔹 9. Multi-Class ROC–AUC

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
```

---

## 🔹 10. Example Comparison

| Model | AUC | Comment |
|--------|-----|----------|
| Logistic Regression | 0.87 | Good balance |
| Random Forest | 0.92 | Better separation |
| k-NN | 0.76 | Moderate |
| SVM | 0.90 | Excellent margin classifier |

---

## 🔹 11. Interpretation Summary

| Point | Meaning |
|--------|----------|
| (0,0) | Classify everything as negative |
| (1,1) | Classify everything as positive |
| (0,1) | Perfect classifier (ideal) |
| Diagonal line | Random predictions |
| Higher AUC | Better class separability |

---

## 🧩 12. Exercises

1. Compute ROC curve for a binary classifier.  
2. Calculate AUC score using `roc_auc_score()` and interpret it.  
3. Plot ROC curves of multiple models on the same graph.  
4. Compare ROC–AUC vs. PR–AUC for imbalanced data.  
5. Discuss why a model may have high accuracy but low AUC.

---

## 🧾 13. Summary

| Concept | Description |
|----------|--------------|
| **ROC Curve** | Shows trade-off between TPR and FPR |
| **AUC** | Area under ROC curve – overall model ability |
| **Good ROC curve** | Close to top-left corner |
| **Use ROC–AUC** | For comparing classifiers |
| **PR Curve** | Better for imbalanced datasets |

---

✅ **Next Topic:**  
📘 *606. Model Evaluation: Precision, Recall, F1, ROC–AUC Integration*
