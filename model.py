import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data preprocessing
# transfer the image data into the format that CNN needs.
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# one hot encoding
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# model
model = Sequential()
model.add(Conv2D(48, (3, 3), padding = 'same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2), padding = 'same'))
model.add(Dropout(0.25))
model.add(Conv2D(96, (3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D((2, 2), padding = 'same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model compile & train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size = 50, epochs = 50, validation_data = (x_test, y_test)) 
model.save('mnist.h5')

# plot accuracy vs. epoch
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


# plot loss vs. epoch
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.tight_layout()
plt.show()
