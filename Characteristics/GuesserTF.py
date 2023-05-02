import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

checkpoint_path = r"C:\Users\jimmy\Desktop\Coding\Python\FishML\Characteristics\ckpt/model.ckpt"



data_dir = r'C:\Users\jimmy\Desktop\Coding\Python\FishML\Characteristics\Images'
# try:
# 	tf.keras.utils.image_dataset_from_directory(data_dir)
# except ValueError:
# 	print("Could not load images - failing silently")
data_dir = pathlib.Path(data_dir)

batch_size = 16
img_height = 200
img_width = 200

try:
	val_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="validation",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size)
except ValueError:
	print("Could no load image dataset - failing silently")
AUTOTUNE = tf.data.AUTOTUNE
try:
	train_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="training",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size)
	class_names = train_ds.class_names
	num_classes = len(class_names)
except ValueError:
	class_names = os.listdir(data_dir)
	num_classes = len(class_names)
	print("Could not load dataset - using listDir")

model = Sequential([
	layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
	layers.Conv2D(16, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(64, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Dropout(0.2),
	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(num_classes)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.summary()

try:
	model.load_weights(checkpoint_path)
except:
	print("Could not load checkpoint - defaulting")

def guess(img):
	# EnergyTest_path = r'C:\Users\jimmy\Desktop\Coding\Python\ML\Images\Unfiled\0.png'

	# img = tf.keras.utils.load_img(EnergyTest_path, target_size=(img_height, img_width))
	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])
	# print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
	return(class_names[np.argmax(score)], np.max(score))