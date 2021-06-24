# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:27:26 2021

@author: Teresa
"""

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#replace with output from get_fibermap.py
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#replace with correct class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = models.Sequential()
#change input_shape to be (40, 40, 3)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding = "same"))
# model.add(layers.MaxPooling2D((2, 2), padding='same'))
# model.add(layers.Dropout(0.025))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding = "same"))
# model.add(layers.MaxPooling2D((2, 2), padding = "same"))
# model.add(layers.Dropout(0.025))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = "same"))
#use . A max pooling layer of size 2 and a dropout layer of 2.5 are used to prevent overfitting
model.add(layers.MaxPooling2D((2, 2), padding = "same"))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = "same"))
# model.add(layers.MaxPooling2D((2, 2), padding = "same"))
# model.add(layers.Dropout(0.025))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = "same"))
# model.add(layers.MaxPooling2D((2, 2), padding = "same"))
model.add(layers.Dropout(0.025))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = "same"))
# model.add(layers.MaxPooling2D((2, 2), padding = "same"))
# model.add(layers.Dropout(0.025))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding = "same"))
# model.add(layers.MaxPooling2D((2, 2), padding = "same"))
# model.add(layers.Dropout(0.025))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding = "same"))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dropout(.02))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(.02))
model.add(layers.Dense(512, activation='relu'))
#model.add(layers.Dropout(.02))
#change to be 35
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#change epochs = 25
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

#vector of size of classes
#convert to probability instead of one-hot encoding
plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)