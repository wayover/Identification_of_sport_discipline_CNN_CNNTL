import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries


def load_and_prep_image(filename):
    img = tf.keras.preprocessing.image.load_img(filename, target_size=(300, 200))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    return img

model = tf.keras.models.load_model('Test1-CNN-TL.h5')

image_path = "D:\Dane z Pulpitu\zdjecia v2\zdtest\\badminton\\fabien-delrue-and-william-villeger-of-france-compete-in-the-mens-doubles-second-round-match2.jpg"
image = load_and_prep_image(image_path)

image = np.expand_dims(image, axis=0)

explainer = lime_image.LimeImageExplainer()

segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=50, ratio=0.1)

explanation = explainer.explain_instance(image[0],
                                         model.predict,
                                         top_labels=4,
                                         hide_color=0,
                                         num_samples=1000,
                                         segmentation_fn=segmentation_fn)

fig, axes = plt.subplots(1, 4, figsize=(20,5))
class_indices = {0: 'badminton', 1: 'squash', 2: 'tenis stołowy', 3: 'tenis ziemny'}

for i in range(4):
    label = explanation.top_labels[i]
    class_name = class_indices[label]

    temp, mask = explanation.get_image_and_mask(label, positive_only=False, num_features=10, hide_rest=False)

    axes[i].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    axes[i].axis('off')
    axes[i].set_title(class_name)

plt.show()