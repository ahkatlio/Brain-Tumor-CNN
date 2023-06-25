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
#################################Data Preprocessing############################
# Path to the data directory
data_dir = 'E:\CNN for Brain Tumour\Brain Tumor Data Set\Brain Tumor Data Set'
# Get the list of all the images
images = os.listdir(data_dir)
# Get the list of all the images
data = []
labels = []
for i in ['Healthy', 'Brain Tumor']:
    path = os.path.join(data_dir,i)
    for img in os.listdir(path):
        try:
            image = cv2.imread(os.path.join(path,img))
            #print(os.path.join(path,img))
            image = cv2.resize(image, (224,224))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")
# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
# Shuffle the data
data,labels = shuffle(data,labels, random_state=42)
# Split the data into train and test set
train_data,test_data,train_labels,test_labels = train_test_split(data,labels,test_size=0.1,random_state=42)
# Normalize the data
train_data = train_data / 255.0
test_data = test_data / 255.0
# Onehot encoding the labels
train_labels = pd.get_dummies(train_labels).values
test_labels = pd.get_dummies(test_labels).values
# Data Augmentation
train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
train_datagen.fit(train_data)
#################################Model Building################################
# Model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
history = model.fit(train_datagen.flow(train_data, train_labels, batch_size=32), epochs=10, validation_data=(test_data, test_labels))
# Save the model
model.save('model.h5')
# Plot the training and validation accuracy and loss at each epoch
epochs = [i for i in range(10)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)
ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()