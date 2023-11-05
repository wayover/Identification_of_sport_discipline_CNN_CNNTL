import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# Used to build and deploy machine learning apps
import tensorflow as tf

# Deep Learning API for creating Neural Networks (Runs on TensorFlow)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import math
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, MaxPooling2D, AveragePooling2D,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix




physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_dir = "D:\Dane z Pulpitu\Zdjecia v3"

# Import data, turn it into batches of 32 with a size of 256x256
training_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    batch_size=32,
    image_size=(450, 300),
    seed=30
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    batch_size=32,
    image_size=(450, 300),
    seed=30
)

# Get class names
class_names = training_data.class_names
num_classes = len(class_names)
print(class_names)

def preprocess_label(image, label):
    label = tf.cast(label, tf.int32)
    label = tf.map_fn(lambda x: tf.one_hot(x, num_classes), label, dtype=tf.float32)
    return image, label

training_data_norm = training_data.shuffle(len(training_data)).map(preprocess_label)
validation_data_norm = validation_data.shuffle(len(validation_data)).map(preprocess_label)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomContrast(factor=0.3)
])

# Apply data augmentation to training data
def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # Normalizacja obrazu
    image = data_augmentation(image)  # Augmentacja danych
    return image, label

augmented_training_data = training_data_norm.map(preprocess_image)

model = tf.keras.models.Sequential([
    Conv2D(filters=96, kernel_size=(11), activation="relu", input_shape=(450, 300, 3)),
    MaxPool2D(pool_size=(3, 3)),
    Conv2D(256, 5, activation="relu"),
    MaxPool2D(pool_size=(3, 3)),
    Conv2D(384, 3, activation="relu"),
    Conv2D(384, 3, activation="relu"),
    Conv2D(256, 3, activation="relu"),
    MaxPool2D(pool_size=(3, 3)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=["accuracy"])

epoc = 30;

# Fit the model
history = model.fit(augmented_training_data,
                    epochs=epoc,
                    steps_per_epoch=len(training_data_norm),
                    validation_data=validation_data_norm,
                    validation_steps=len(validation_data_norm))


model.save('img450-300-lr0001seed30epoc10.h5')