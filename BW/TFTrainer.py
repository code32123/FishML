import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

if __name__ == '__main__':
	epochs = input("Epochs (10): ")
	epochs = 10 if epochs == "" else int(epochs)
	Resume = False if input("Resume? (y)").lower() == 'n' else True
else:
	epochs = 5
	Resume = True

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

def QuickTest(Text):
	EnergyTest_path = r'C:\Users\jimmy\Desktop\Coding\Python\FishML\BW\Images'
	EnergyTests = os.listdir(EnergyTest_path)
	for i in EnergyTests:
		img = tf.keras.utils.load_img(EnergyTest_path + '\\' + i, target_size=(img_height, img_width))
		img_array = tf.keras.utils.img_to_array(img)
		img_array = tf.expand_dims(img_array, 0) # Create a batch
		predictions = model.predict(img_array)
		score = tf.nn.softmax(predictions[0])
		print(Text + ": {} {}, {:.2f} percent confidence.".format(i[:-4], class_names[np.argmax(score)], 100 * np.max(score)))

def train(Resume=True, epochs=5):
	checkpoint_path = r"C:\Users\jimmy\Desktop\Coding\Python\FishML\BW\ckpt/model.ckpt"
	data_dir = r'C:\Users\jimmy\Desktop\Coding\Python\FishML\BW\Images'
	tf.keras.utils.image_dataset_from_directory(data_dir)
	data_dir = pathlib.Path(data_dir)

	image_count = len(list(data_dir.glob('*/*.png')))
	print(image_count)

	batch_size = 16
	img_height = 200
	img_width = 200

	train_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="training",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size)

	val_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="validation",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size)

	class_names = train_ds.class_names
	print(class_names)

	# plt.figure(figsize=(10, 10))
	# for images, labels in train_ds.take(1):
	# 	for i in range(9):
	# 		ax = plt.subplot(3, 3, i + 1)
	# 		plt.imshow(images[i].numpy().astype("uint8"))
	# 		plt.title(class_names[labels[i]])
	# 		plt.axis("off")

	AUTOTUNE = tf.data.AUTOTUNE

	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

	normalization_layer = layers.Rescaling(1./255)

	normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
	image_batch, labels_batch = next(iter(normalized_ds))
	first_image = image_batch[0]
	# Notice the pixel values are now in `[0,1]`.
	# print(np.min(first_image), np.max(first_image))

	num_classes = len(class_names)

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

	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
	                                                 save_weights_only=True,
	                                                 verbose=1)

	if Resume:
		model.load_weights(checkpoint_path)


	# QuickTest("Pretrain")

	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs,
		callbacks=[cp_callback],
	)

	# QuickTest("Postrain")

	# acc = history.history['accuracy']
	# val_acc = history.history['val_accuracy']

	# loss = history.history['loss']
	# val_loss = history.history['val_loss']

	# epochs_range = range(epochs)

	# plt.figure(figsize=(8, 8))
	# plt.subplot(1, 2, 1)
	# plt.plot(epochs_range, acc, label='Training Accuracy')
	# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	# plt.legend(loc='lower right')
	# plt.title('Training and Validation Accuracy')

	# plt.subplot(1, 2, 2)
	# plt.plot(epochs_range, loss, label='Training Loss')
	# plt.plot(epochs_range, val_loss, label='Validation Loss')
	# plt.legend(loc='upper right')
	# plt.title('Training and Validation Loss')
	# plt.show()

if __name__ == '__main__':
	train(True)