# %%
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# %%
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Convert pixel values to floats in the range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Subtract the mean pixel value from each image
mean_pixel = x_train.mean(axis=(0,1,2), keepdims=True)
x_train -= mean_pixel
x_test -= mean_pixel

# Convert labels to one-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# %%
# RESNET50 model using Keras
model = ResNet50(weights=None, input_shape=x_train.shape[1:], classes=10)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# summary of the model
model.summary()

# %% [markdown]
# <center>
#     <img src = "image/resnet50.jpg">
# </center>

# %%
# Train the model
resnet = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

# %%
plt.plot(resnet.history['accuracy'], label='accuracy')
plt.plot(resnet.history['val_accuracy'], label = 'val_accuracy')
plt.title('ResNet')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('image/resnet50.jpg')

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)


