# imports
import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# loads data set
from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# sns.countplot(x=y_train)
# plt.show()

# check values to be sure there are no values that are not numbers
print("Any NaN Training:", np.isnan(x_train).any())
print("Any NaN Testing:", np.isnan(x_test).any())

# tell model what shape to expect
# 1 - grayscale, 3 - RGB
input_shape = (32, 32, 3) 

# reshape training and testing data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# convert our labels to be one-hot, not sparse
from keras.utils import to_categorical
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

# show an example image from MNIST
plt.imshow(x_train[random.randint(0, 5999)][:,:,0])
plt.show()



batch_size = 128
num_classes = 100
epochs = 5

# builds model
# Model 1
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.30), 
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ]
)
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

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train,epochs=5,validation_data=(x_test, y_test))


# plot out training and validation accuracy and loss
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color = 'b', label='Training Loss')
ax[0].plot(history.history['val_loss'], color = 'r', label='Validation Loss')
legend = ax[0].legend(loc='best', shadow=True)
ax[0].set_title('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')

ax[1].plot(history.history['acc'], color = 'b', label='Training Accuracy')
ax[1].plot(history.history['val_acc'], color = 'r', label='Validation Accuracy')
legend = ax[1].legend(loc='best', shadow=True)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')

plt.tight_layout()
plt.show()
model.summary()