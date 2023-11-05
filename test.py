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
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, MaxPooling2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_dir = "D:\Dane z Pulpitu\zdjecia v2\zdtest"

model = tf.keras.models.load_model('Test1-CNN-TL.h5')

# Utworzenie obiektu ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

test_generator = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    batch_size=32,
    image_size=(300, 200),
    seed=35
)


# Get class names
class_names = test_generator.class_names
num_classes = len(class_names)
print(class_names)

def preprocess_label(image, label):
    label = tf.cast(label, tf.int32)
    label = tf.map_fn(lambda x: tf.one_hot(x, num_classes), label, dtype=tf.float32)

    return image, label

test_generator_norm = test_generator.shuffle(len(test_generator)).map(preprocess_label)

# Pobieranie danych walidacyjnych
validation_data = []
validation_labels = []
for images, labels in test_generator:
    validation_data.append(images)
    validation_labels.append(labels)

validation_data = np.concatenate(validation_data)
validation_labels = np.concatenate(validation_labels)

# Przewidywanie etykiet na danych walidacyjnych
predicted_labels = np.argmax(model.predict(validation_data), axis=-1)

# Przekształcenie validation_labels do postaci dwuwymiarowej
validation_labels = np.expand_dims(validation_labels, axis=1)

cm = confusion_matrix(validation_labels, predicted_labels)


plt.figure(figsize=(8, 6))
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float)
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
sns.heatmap(cm_perc, annot=True, fmt='.2%', cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)

plt.ylabel('Prawdziwa klasa')
plt.xlabel('Przewidywana klasa')

total_correct = np.sum(np.diag(cm))
total_predictions = np.sum(cm)
overall_accuracy = (total_correct / total_predictions) * 100
plt.title('Confusion matrix')
plt.text(x=0.5, y=1.05, s='Accuracy: {:.2f}%'.format(overall_accuracy),
         fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes)
plt.show()
