#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fernandoduarte
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the trained model
model = load_model("Models/melanoma_resnet50_model_v4.h5")

# Preprocess an input image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# Define test data directory
path = os.getcwd()
print(path)
test_data_dir = os.path.join(path, "/tests/val/")
print(test_data_dir)
test_data_dir = path + test_data_dir
# Class labels
class_labels = ["non-melanoma", "melanoma"]

# Lists to store results
true_labels = []
pred_labels = []
confidence_scores = []
image_names = []

# Loop through test images in both subfolders
for category in ["non-melanoma", "melanoma"]:
    category_path = os.path.join(test_data_dir, category)
    true_label = class_labels.index(category)
    
    for filename in os.listdir(category_path):
        image_path = os.path.join(category_path, filename)
        
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image
            processed_image = preprocess_image(image_path)
            prediction = model.predict(processed_image)

            # Get predicted class
            pred_label = int(prediction[0][0] > 0.5)
            predicted_class = class_labels[pred_label]
            confidence = round(float(prediction[0][0]) * 100, 2)

            # Store results
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            confidence_scores.append(confidence)
            image_names.append(filename)

            print(f"Image: {filename} | True Class: {category.capitalize()} | Predicted: {predicted_class} ({confidence}%)")

# Save results to CSV
df = pd.DataFrame({
    "Image Name": image_names,
    "True Class": [class_labels[i] for i in true_labels],
    "Predicted Class": [class_labels[i] for i in pred_labels],
    "Confidence (%)": confidence_scores
})
df.to_csv("results_MODEL_PREDICTION.csv", index=False)
print("\nResults saved to results.csv")

# Generate and display confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
