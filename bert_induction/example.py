from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
keras_mobilenet_v2 = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3), include_top=False)

estimator_model = tf.keras.Sequential([
    keras_mobilenet_v2,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='softmax')
])

# Compile the model
estimator_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

est_mobilenet_v2 = tf.keras.estimator.model_to_estimator(keras_model=estimator_model)

IMG_SIZE = 160  # All images will be resized to 160x160

def preprocess(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

def train_input_fn(batch_size):
  data = tfds.load('cats_vs_dogs', as_supervised=True)
  train_data = data['train']
  train_data = train_data.map(preprocess).shuffle(500).batch(batch_size)
  return train_data

est_mobilenet_v2.train(input_fn=lambda: train_input_fn(32), steps=500)

est_mobilenet_v2.evaluate(input_fn=lambda: train_input_fn(32), steps=10)