
## Pneumonia Detection from Chest X-Ray Images using Deep Learning

## Project Overview

This project builds a Deep Learning model using Convolutional Neural Networks (CNN) to detect Pneumonia from Chest X-ray images.

The model classifies X-ray images into two categories:

Normal

Pneumonia

Early detection of pneumonia can help doctors provide faster treatment and reduce health risks.

## Business Problem

Manual diagnosis of pneumonia from chest X-rays can take time and requires expert radiologists.

This project aims to automatically detect pneumonia using AI, helping hospitals and doctors make faster decisions.

## Dataset

The dataset contains grayscale chest X-ray images divided into:

Train dataset

Validation dataset

Test dataset

Classes:

Normal

Pneumonia

## Dataset Source:
Chest X-Ray Dataset (Medical Images)

## Folder Structure:

chest_xray
│
├── train
│   ├── NORMAL
│   └── PNEUMONIA
│
├── val
│   ├── NORMAL
│   └── PNEUMONIA
│
├── test
│   ├── NORMAL
│   └── PNEUMONIA

## Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn

Google Colab

## Project Workflow
### 1 Data Loading

Load the chest X-ray dataset from Google Drive.

### 2 Data Preprocessing

Image resizing

Image normalization

Data augmentation using ImageDataGenerator

### 3 Model Building

A Convolutional Neural Network (CNN) is built using TensorFlow/Keras.

Typical layers used:

Convolution Layer

MaxPooling Layer

Flatten Layer

Dense Layer

Dropout

### 4 Model Training

The model is trained on the training dataset with validation monitoring.

Callbacks used:

EarlyStopping

ModelCheckpoint

ReduceLROnPlateau

### 5 Model Evaluation

The model performance is evaluated using:

Accuracy

Confusion Matrix

Classification Report

Model Output

The model predicts whether a chest X-ray image belongs to:

Normal

Pneumonia

This helps assist doctors in detecting pneumonia early.

## Example Libraries Used
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
## Future Improvements

Use Transfer Learning (ResNet, VGG16, EfficientNet)

Improve accuracy with larger datasets

Deploy as a web application

Integrate into hospital diagnostic systems

## Author

Rohini Bawake

LinkedIn
https://www.linkedin.com/in/rohini-bawake/

GitHub
https://github.com/RohiniBawake31