#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:12:56 2025

@author: fernandoduarte
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load the results from the CSV file
df = pd.read_csv("results_MODEL_PREDICTION.csv")

# Create confusion matrix
true_labels = df["True Class"].values
pred_labels = df["Predicted Class"].values
cm = confusion_matrix(true_labels, pred_labels, labels=["non-melanoma", "melanoma"])

# Calculate metrics
tn, fp, fn, tp = cm.ravel()  # tn: true negatives, fp: false positives, fn: false negatives, tp: true positives

# Calculate percentages
total = tp + tn + fp + fn
accuracy = (tp + tn) / total * 100
sensitivity = tp / (tp + fn) * 100  # True Positive Rate
specificity = tn / (tn + fp) * 100  # True Negative Rate
precision = tp / (tp + fp) * 100  # Positive Predictive Value
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

# Print metrics
print("Confusion Matrix:")
print(cm)
print("\nMetrics:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Sensitivity (TPR): {sensitivity:.2f}%")
print(f"Specificity (TNR): {specificity:.2f}%")
print(f"Precision (PPV): {precision:.2f}%")
print(f"F1 Score: {f1_score:.2f}")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Melanoma", "Melanoma"], yticklabels=["Non-Melanoma", "Melanoma"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Matriz de confusão")
plt.show()
