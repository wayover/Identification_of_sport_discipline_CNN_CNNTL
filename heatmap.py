import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2

# Functions for heatmap generation
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap_on_image(img, heatmap):
    heatmap = np.uint8(255 * heatmap)
    greens = plt.get_cmap("Greens")
    green_colors = greens(np.arange(256))[:, :3]
    green_heatmap = green_colors[heatmap]
    green_heatmap = tf.keras.preprocessing.image.array_to_img(green_heatmap)
    green_heatmap = green_heatmap.resize((img.shape[1], img.shape[0]))
    green_heatmap = tf.keras.preprocessing.image.img_to_array(green_heatmap)
    superimposed_img = green_heatmap * 1.6 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img



# Function to load, preprocess and plot image with heatmap
def pred_and_plot(model, filename, class_names):
    img = tf.keras.preprocessing.image.load_img(filename, target_size=(300, 200))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred[0])]
    last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap_img = overlay_heatmap_on_image(img_array[0]*255, heatmap)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img_array[0]))
    plt.title(f'Oryginalne zdjęcie: {pred_class}')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(heatmap_img)
    plt.title(f"Heatmap {pred_class}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Model prediction: {pred_class}")


# Load model
model = tf.keras.models.load_model('Test1-CNN-TL.h5')

# Define class names
class_names = ['badminton', 'squash', 'tenis stolowy', 'tenis ziemny']

pred_and_plot(model,"D:\Dane z Pulpitu\zdjecia v2\zdtest\\badminton\\kento-momota-of-japan-competes-in-the-mens-singles-second-round-match-against-prannoy-h-s-of2.jpg",class_names)
