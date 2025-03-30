#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fernandoduarte
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Define test data directory
path = os.getcwd()
test_data_dir = os.path.join(path, 'tests/val/metadata.csv')

# Load the results from the CSV file
df_results = pd.read_csv("results_MODEL_PREDICTION.csv")
df_results_sorted = df_results.sort_values(by="isic_id")

df_fitzpatrick = pd.read_csv(test_data_dir)  # Load the Fitzpatrick data

df_results_sorted['isic_id'] = df_results_sorted['isic_id'].astype(str).str.replace('.jpg', '', regex=False)

# Merge the dataframes on 'isic_id'
df = pd.merge(df_results_sorted, df_fitzpatrick[['isic_id', 'fitzpatrick_skin_type']], on='isic_id', how='left')
df.to_csv("merged_results_with_fitzpatrick.csv", index=False)

# Initialize dictionaries for metrics
metrics = {"skin_type": [], "accuracy": [], "sensitivity": [], "specificity": [], "precision": [], "f1_score": []}
fairness_metrics = {"skin_type": [], "disparate_impact": [], "equal_opportunity": [], "predictive_parity": []}
selection_rates = {}
confusion_matrices = []

# Unique Fitzpatrick skin types sorted in ascending order
skin_types = np.sort(df['fitzpatrick_skin_type'].astype(str).unique())

for skin_type in skin_types:
    # Filter results for the current skin type
    skin_df = df[df['fitzpatrick_skin_type'] == skin_type]
    
    # Create confusion matrix
    true_labels = skin_df["True Class"].values
    pred_labels = skin_df["Predicted Class"].values
    
    if len(true_labels) == 0:  # Avoid errors if no samples
        continue
    
    cm = confusion_matrix(true_labels, pred_labels, labels=["non-melanoma", "melanoma"])
    confusion_matrices.append((skin_type, cm))

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # Selection rate for disparate impact
    selection_rate = (tp + fp) / total if total > 0 else 0
    selection_rates[skin_type] = selection_rate

    # Store metrics
    metrics["skin_type"].append(skin_type)
    metrics["accuracy"].append(accuracy *100)
    metrics["sensitivity"].append(sensitivity*100)
    metrics["specificity"].append(specificity*100)
    metrics["precision"].append(precision*100)
    metrics["f1_score"].append(f1_score*100)

    # Store fairness metrics (excluding disparate impact for now)
    fairness_metrics["skin_type"].append(skin_type)
    fairness_metrics["equal_opportunity"].append(sensitivity)
    fairness_metrics["predictive_parity"].append(precision)

# Calculate Disparate Impact Ratio (DIR) after collecting all selection rates
reference_rate = max(selection_rates.values()) if selection_rates else 0
for skin_type, rate in selection_rates.items():
    disparate_impact = rate / reference_rate if reference_rate > 0 else 0
    fairness_metrics["disparate_impact"].append(disparate_impact)
    print(f"Skin Type {skin_type}: Selection Rate = {rate:.3f}, DIR = {disparate_impact:.3f}")

# Convert metrics to DataFrames
metrics_df = pd.DataFrame(metrics)
fairness_df = pd.DataFrame(fairness_metrics)

# Save metrics to CSV files
metrics_df.to_csv("skin_type_metrics.csv", index=False)
fairness_df.to_csv("skin_type_fairness_metrics.csv", index=False)

# Plotting the results (unchanged)
plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
sns.barplot(x='skin_type', y='accuracy', data=metrics_df)
plt.title('Accuracy per Fitzpatrick Skin Type')
plt.subplot(2, 3, 2)
sns.barplot(x='skin_type', y='sensitivity', data=metrics_df)
plt.title('Sensitivity per Fitzpatrick Skin Type')
plt.subplot(2, 3, 3)
sns.barplot(x='skin_type', y='specificity', data=metrics_df)
plt.title('Specificity per Fitzpatrick Skin Type')
plt.subplot(2, 3, 4)
sns.barplot(x='skin_type', y='precision', data=metrics_df)
plt.title('Precision per Fitzpatrick Skin Type')
plt.subplot(2, 3, 5)
sns.barplot(x='skin_type', y='f1_score', data=metrics_df)
plt.title('F1 Score per Fitzpatrick Skin Type')
plt.tight_layout()
plt.show()

# Plotting fairness metrics
plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
sns.barplot(x='skin_type', y='disparate_impact', data=fairness_df)
plt.title('Disparate Impact per Skin Type')
plt.subplot(2, 3, 2)
sns.barplot(x='skin_type', y='equal_opportunity', data=fairness_df)
plt.title('Equal Opportunity per Skin Type')
plt.subplot(2, 3, 3)
sns.barplot(x='skin_type', y='predictive_parity', data=fairness_df)
plt.title('Predictive Parity per Skin Type')
plt.tight_layout()
plt.show()

# Save and plot confusion matrices (unchanged)
for skin_type, cm in confusion_matrices:
    cm_df = pd.DataFrame(cm, index=["True Non-Melanoma", "True Melanoma"], columns=["Predicted Non-Melanoma", "Predicted Melanoma"])
    cm_df.to_csv(f"confusion_matrix_{skin_type}.csv", index=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Predicted Non-Melanoma", "Predicted Melanoma"],
                yticklabels=["True Non-Melanoma", "True Melanoma"],
                annot_kws={"size": 16})  # Increase font size of numbers
    plt.title(f"Confusion Matrix for Fitzpatrick Skin Type: {skin_type}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"confusion_matrix_plot_{skin_type}.png")
    plt.show()