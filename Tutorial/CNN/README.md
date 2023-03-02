# CNN model comparison

## Dataset

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Preprocessing

- Train: `50000`
- Test: `10000`

## Models

- Custom CNN
- VGG16
- ResNet50

## Results

- Epochs: `10`
- lr: `0.00001`
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- batch_size: default (32)

| Model | Accuracy |
| --- | --- |
| Custom CNN | 0.4583 |
| VGG16 | 0.6258 |
| ResNet50 | 0.4628 |