# %% [markdown]
# # VGG16

# %%
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# %%
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# %% [markdown]
# VGG16 is a convolutional neural network that is 16 layers deep. 
# 
# You can load a pretrained version of the network trained on more than a million images from the ImageNet database. 
# 
# The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.
# 
# ## Important part of VGG16
# 
# 1. 3x3 Convolutional layer
# 2. 2x2 Max Pooling layer
# 3. Fully connected layer
# 4. Softmax layer

# %%
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# %% [markdown]
# <center>
#     <img src = "image/vgg16.jpeg">
# </center>

# %%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

vgg16 = model.fit(train_images, train_labels, epochs=100
                    , validation_data=(test_images, test_labels))

# %%
plt.plot(vgg16.history['accuracy'], label='accuracy')
plt.plot(vgg16.history['val_accuracy'], label = 'val_accuracy')
plt.title('VGG-16')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('VGG16.png')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


