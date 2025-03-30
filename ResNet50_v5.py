#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fernandoduarte
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam  # Use legacy Adam if needed
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

path = os.getcwd()

# Directory paths
train_dir = path + "/data/ISIC-images/train"
val_dir = path + "/data/ISIC-images/val"
img_size = (224, 224)
batch_size = 32
num_classes = 2

from tensorflow.keras.callbacks import CSVLogger

# Create a CSV logger to save training logs
log_file = "training_log.csv"
csv_logger = CSVLogger(log_file, append=True)  # Set append=True to keep previous logs

train_datagen = ImageDataGenerator(
    brightness_range=[0.7, 1.3],  # Vary brightness
    rescale=1./255,
    rotation_range=40,  # Reduce from 45° to 30°
    width_shift_range=0.3,  # Reduce from 30% to 20%
    height_shift_range=0.3,
    shear_range=0.3,  # Reduce from 30% to 20%
    zoom_range=0.3,  # Reduce from 30% to 20%
    horizontal_flip=True,
    vertical_flip=False,  # Disable vertical flip
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Load Pretrained ResNet50 Model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers initially

# Ensure proper connection between layers
x = base_model.output  # Get the output of ResNet50
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)  # Ensure correct connection

# Define the model
model = Model(inputs=base_model.input, outputs=output_layer)  # Properly define inputs and outputs
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0003),  # Reduce initial LR
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=10,  # Start with 10, increase if needed
                    verbose=1,
                    callbacks=[csv_logger])

# Unfreeze base model for fine-tuning
base_model.trainable = True

for layer in base_model.layers[:-100]:  # Keep first 100 layers frozen
    layer.trainable = False

# Recompile after unfreezing layers
model.compile(optimizer=Adam(learning_rate=5e-6),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Continue training (fine-tuning)
history_fine = model.fit(train_generator,
                         validation_data=val_generator,
                         epochs=50,
                         verbose=1,
                         callbacks=[early_stopping, csv_logger, lr_scheduler])

# Save the trained model
model.save("melanoma_resnet50_model.h5")

# Evaluate the model
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc * 100:.2f}%")