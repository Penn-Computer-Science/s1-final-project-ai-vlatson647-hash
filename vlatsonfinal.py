# imports
import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# loads data set
image_dir = "data"
img_height = 224
img_width = 224

input_shape = (224, 224, 3)

dataset = tf.keras.utils.image_dataset_from_directory(
    image_dir,
    image_size=(img_height, img_width),
    batch_size=32
)

# Normalize images
dataset = dataset.map(lambda x, y: (x/255.0, y))

# Show one example image
for images, labels in dataset.take(1):
    idx = random.randint(0, images.shape[0] - 1)
    plt.imshow(images[idx].numpy())
    plt.title(f"Label: {labels[idx].numpy()}")
    plt.axis("off")
    plt.show()



batch_size = 128
num_classes = 8
epochs = 10

# builds model
# Model 1
model = tf.keras.models.Sequential(
    [
    tf.keras.layers.Conv2D(64, (1,1), padding='same', activation='relu',input_shape=input_shape),  # only FIRST layer
    tf.keras.layers.Conv2D(64, (1,1), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Model 2
# model = tf.keras.models.Sequential(
#     [
#         tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu', input_shape=input_shape),
#         tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
#         # tf.keras.layers.MaxPool2D(),
#         tf.keras.layers.Dropout(0.25), 
#         tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
#         tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu', input_shape=input_shape),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(num_classes, activation='softmax'),
#     ]
# )

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

dataset = tf.keras.utils.image_dataset_from_directory(
    image_dir,
    image_size=(img_height, img_width),
    batch_size=32,
    validation_split=0.2,
    subset="both",
    seed=123
)

train_ds = dataset[0]
val_ds   = dataset[1]

train_ds = train_ds.map(lambda x, y: (x/255., y))
val_ds   = val_ds.map(lambda x, y: (x/255., y))

history = model.fit(
    train_ds,
    epochs= 5,
    validation_data=val_ds
)


# plot out training and validation accuracy and loss
fig, ax = plt.subplots(2,1)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# ---- LOSS ----
ax[0].plot(history.history['loss'], color='b', label='Training Loss')
ax[0].plot(history.history['val_loss'], color='r', label='Validation Loss')
ax[0].legend(loc='best', shadow=True)
ax[0].set_title('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')

# ---- ACCURACY ----
ax[1].plot(history.history['accuracy'], color='b', label='Training Accuracy')
ax[1].plot(history.history['val_accuracy'], color='r', label='Validation Accuracy')
ax[1].legend(loc='best', shadow=True)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')

plt.show()


plt.tight_layout()
plt.show()
model.summary()