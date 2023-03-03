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

<table>
    <tr>
        <th>Model</th>
        <th>Test Accuracy</th>
        <th>Graph</th>
    </tr>
    <tr>
        <td>Custom CNN</td>
        <td>0.4583</td>
        <td><img src="./image/customCNN10.png" width="300"></td>
    </tr>
    <tr>
        <td>VGG16</td>
        <td>0.6258</td>
        <td><img src="./image/vgg16_10.png" width="300"></td>
    </tr>
    <tr>
        <td>ResNet50</td>
        <td>0.4628</td>
        <td><img src="./image/resnet50_10.png" width="300"></td>
</table>

- Epochs: `100`
- lr: `0.00001`
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- batch_size: default (32)

<table>
    <tr>
        <th>Model</th>
        <th>Test Accuracy</th>
        <th>Graph</th>
    </tr>
    <tr>
        <td>Custom CNN</td>
        <td>0.6239</td>
        <td><img src="./image/customCNN100.png" width="300"></td>
    </tr>
    <tr>
        <td>VGG16</td>
        <td>0.6941</td>
        <td><img src="./image/vgg16_100.png" width="300"></td>
    </tr>
    <tr>
        <td>ResNet50</td>
        <td>0.5234</td>
        <td><img src="./image/resnet50_100.jpg" width="300"></td>
</table>