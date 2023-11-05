import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_dir = "D:\Dane z Pulpitu\Zdjecia v3"

data = []
labels = []
class_names = []
label_counter = 0
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(300, 200))
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            data.append(image_array)
            labels.append(label_counter)
        class_names.append(class_name)
        label_counter += 1

data = np.array(data, dtype="float32")
labels = np.array(labels)

data /= 255.0

model = Sequential([
    Conv2D(filters=96, kernel_size=(11), activation="relu", input_shape=(300, 200, 3)),
    MaxPool2D(pool_size=(3, 3)),
    Conv2D(256, 5, activation="relu"),
    MaxPool2D(pool_size=(3, 3)),
    Conv2D(384, 3, activation="relu"),
    Conv2D(384, 3, activation="relu"),
    Conv2D(256, 3, activation="relu"),
    MaxPool2D(pool_size=(3, 3)),
    Flatten(),
    Dense(2048, activation="relu"),
    Dense(1024, activation="relu"),
    Dense(label_counter, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=["accuracy"])

datagen = ImageDataGenerator(
    rotation_range=0.3,
    zoom_range=0.2,
    fill_mode="nearest",
    horizontal_flip=True
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

all_histories = []
conf_matrices = []

for train, test in skf.split(data, labels):
    train_data, validation_data, train_labels, validation_labels = train_test_split(data[train], labels[train], test_size=0.2, random_state=42)

    history = model.fit(datagen.flow(train_data, train_labels), epochs=6, validation_data=(validation_data, validation_labels))

    all_histories.append(history.history)

    predictions = model.predict(data[test], batch_size=32)
    cm = confusion_matrix(labels[test], np.argmax(predictions, axis=1))
    conf_matrices.append(cm)


model.save('Test8-CNN.h5')

concatenated_histories = {}
for key in all_histories[0].keys():
    concatenated_histories[key] = np.concatenate([x[key] for x in all_histories])

plt.figure(figsize=(18, 4))
plt.subplot(1, 3, 1)
plt.plot(concatenated_histories['accuracy'])
plt.plot(concatenated_histories['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoki')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 3, 2)
plt.plot(concatenated_histories['loss'])
plt.plot(concatenated_histories['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoki')
plt.legend(['Train', 'Validation'], loc='upper left')

mean_conf_matrix = np.mean(conf_matrices, axis=0)
mean_conf_matrix = mean_conf_matrix / mean_conf_matrix.sum(axis=1)[:, np.newaxis]
plt.subplot(1, 3, 3)
sns.heatmap(mean_conf_matrix, annot=True, fmt='.2%', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion matrix')
plt.ylabel('Prawdziwa klasa')
plt.xlabel('Przewidywana klasa')

plt.tight_layout()
plt.show()
