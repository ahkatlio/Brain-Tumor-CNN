###########################importing libraries################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
import cv2
import imutils
import keras 
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report


#################################Model Testing#################################
# Load the saved model
model = load_model('model.h5')
# Path to the test data directory
test_data_dir = 'E:\CNN for Brain Tumour\Brain Tumor Data Set\Brain Tumor Data Set'
# Get the list of all the images
test_images = os.listdir(test_data_dir)
# Get the list of all the images
test_data = []
test_labels = []
for i in ['Healthy', 'Brain Tumor']:
    path = os.path.join(test_data_dir,i)
    for img in os.listdir(path):
        try:
            image = cv2.imread(os.path.join(path,img))
            print(os.path.join(path,img))
            image = cv2.resize(image, (224,224))
            image = np.array(image)
            test_data.append(image)
            test_labels.append(i)
        except:
            print("Error loading image")
# Converting lists into numpy arrays
test_data = np.array(test_data)
test_labels = np.array(test_labels)
# Shuffle the data
test_data,test_labels = shuffle(test_data,test_labels, random_state=42)
# Normalize the data
test_data = test_data / 255.0
# Onehot encoding the labels
test_labels = pd.get_dummies(test_labels).values
# Evaluate the model on test data
model.evaluate(test_data, test_labels)
# Predict on test data
batch_size = 32
num_samples = test_data.shape[0]
num_batches = int(np.ceil(num_samples / batch_size))

preds = []
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_samples)
    batch_data = test_data[start_idx:end_idx]
    batch_preds = model.predict(batch_data)
    preds.append(batch_preds)

preds = np.concatenate(preds, axis=0)
# Converting predictions to label
preds = np.argmax(preds, axis=1)
preds = pd.Series(preds, name="Predictions")
# Converting one hot encoded test label to label
test_labels = np.argmax(test_labels, axis=1)
test_labels = pd.Series(test_labels, name="Actual")
# Plotting the confusion matrix
confusion_matrix = pd.crosstab(test_labels, preds, rownames=['Actual'], colnames=['Predictions'])
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix, annot=True)
plt.show()
# Classification report
print(classification_report(test_labels, preds))
# Plotting the test images with actual and predicted labels
plt.figure(figsize=(20,20))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(test_data[i])
    plt.title(f"Actual: {test_labels[i]}\nPredicted: {preds[i]}")
    plt.axis('off')
plt.show()
