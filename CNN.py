import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import math
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, MaxPooling2D, AveragePooling2D,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pathlib

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_dir = "D:\Dane z Pulpitu\Zdjecia v3"

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

class_names = training_data.class_names
num_classes = len(class_names)
print(class_names)

class_names = training_data.class_names

train_data_count = []
for class_name in class_names:
    count = 0
    for images, labels in training_data:
        count += tf.math.count_nonzero(labels == class_names.index(class_name))
    train_data_count.append((class_name, count))

val_data_count = []
for class_name in class_names:
    count = 0
    for images, labels in validation_data:
        count += tf.math.count_nonzero(labels == class_names.index(class_name))
    val_data_count.append((class_name, count))

print("Liczba zdjęć w training_data:")
for class_name, count in train_data_count:
    print(f"{class_name}: {count}")

print("\nLiczba zdjęć w validation_data:")
for class_name, count in val_data_count:
    print(f"{class_name}: {count}")


def preprocess_label(image, label):
    label = tf.cast(label, tf.int32)
    label = tf.map_fn(lambda x: tf.one_hot(x, num_classes), label, dtype=tf.float32)

    return image, label


training_data_norm = training_data.shuffle(len(training_data)).map(preprocess_label)
validation_data_norm = validation_data.shuffle(len(validation_data)).map(preprocess_label)

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomContrast(factor=0.3)
])

def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = data_augmentation(image)
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


model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=["accuracy"])

epoc = 10;
history = model.fit(augmented_training_data,
                    epochs=epoc,
                    steps_per_epoch=len(training_data_norm),
                    validation_data=validation_data_norm,
                    validation_steps=len(validation_data_norm))


model.save('Test8-CNN-TL.h5')

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epoc)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Dokładność trenowania')
plt.plot(epochs_range, val_accuracy, label='Dokładność walidacji')
plt.legend(loc='lower right')
plt.title('Dokładność trenowania i walidacji')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Strata trenowania')
plt.plot(epochs_range, val_loss, label='Strata walidacji')
plt.legend(loc='upper right')
plt.title('Strata trenowania i walidacji')
plt.show()

validation_images = []
validation_labels = []


for images, labels in validation_data_norm:
    validation_images.append(images)
    validation_labels.append(labels)

validation_images = np.concatenate(validation_images)
validation_labels = np.concatenate(validation_labels)

predicted_labels = np.argmax(model.predict(validation_images), axis=-1)

cm = confusion_matrix(np.argmax(validation_labels, axis=-1), predicted_labels)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

