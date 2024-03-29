import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

t_images, v_images = train_images[:40000], train_images[40000:]
t_labels, v_labels = train_labels[:40000], train_labels[40000:]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# RESNET50 model using Keras
model = ResNet50(weights=None, input_shape=train_images.shape[1:], classes=10)

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.9
)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# summary of the model
model.summary()

# Train the model
resnet = model.fit(t_images, t_labels, epochs=20, batch_size=32,
validation_data=(v_images, v_labels))

plt.plot(resnet.history['accuracy'], label='accuracy')
plt.plot(resnet.history['val_accuracy'], label = 'val_accuracy')
plt.title('ResNet')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('ResNet50.png')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test Accuracy: {}, Test Loss: {}".format(test_acc, test_loss))