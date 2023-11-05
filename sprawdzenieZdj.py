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



# Scikit-learn : machine learning library
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Used for accessing files and file names
import pathlib
import os

class_names = ['badminton', 'squash', 'tenis stolowy', 'tenis ziemny']

def load_and_prep_image(filename, img_shape=256):
  img = tf.io.read_file(filename)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.resize(img, size = [450, 300])
  img = img/255.
  return img


def pred_and_plot(model, filename, class_names):
  img = load_and_prep_image(filename)


  pred = model.predict(tf.expand_dims(img, axis=0))
  print(pred)

  pred_class = class_names[int(tf.round(pred)[0][0])]
  print(class_names)
  print(int(tf.round(pred)[0][0]))
  # Plot the image and predicted class
  plt.imshow(img)
  plt.axis(False);
  plt.show()

model = tf.keras.models.load_model('img450-300-lr0001seed35.h5')

pred_and_plot(model, "D:\Dane z Pulpitu\zdjecia v2\zdtest\\tenis stolowy\\wang-chuqin-of-china-plays-a-shot-against-lin-yun-ju-and-cheng-i-ching-of-chinese-taipei.jpg",class_names)