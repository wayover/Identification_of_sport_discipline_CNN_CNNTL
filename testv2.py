import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import os

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_dir = "D:\Dane z Pulpitu\zdjecia v2\zdtest"

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

model = tf.keras.models.load_model('Test8-CNN-TL.h5')

predictions = model.predict(data, batch_size=32)

cm = confusion_matrix(labels, np.argmax(predictions, axis=1))

plt.figure(figsize=(8, 6))
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float)
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
