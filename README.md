# Fashion MNIST Image Classification using TensorFlow

## Overview

This project implements an image classification model using TensorFlow and Keras to classify clothing items from the Fashion MNIST dataset. The model is trained on grayscale images of fashion products and predicts the corresponding clothing category.

## Dataset

The project uses the Fashion MNIST dataset provided by TensorFlow.

* Training Images: 60,000
* Test Images: 10,000
* Image Size: 28 × 28 pixels
* Number of Classes: 10

### Classes

| Label | Category    |
| ----- | ----------- |
| 0     | T-shirt/Top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

## Technologies Used

* Python
* TensorFlow
* Keras
* Matplotlib
* NumPy

## Project Workflow

### 1. Load Dataset

The Fashion MNIST dataset is loaded using TensorFlow.

```python
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()
```

### 2. Explore Dataset

The number of training and testing images is displayed and sample images are visualized using Matplotlib.

### 3. Data Preprocessing

Pixel values are normalized from the range:

```text
0 - 255
```

to

```text
0 - 1
```

using:

```python
training_images = training_images / 255.0
test_images = test_images / 255.0
```

### 4. Build Neural Network

The model consists of:

* Flatten Layer (28×28 → 784)
* Dense Layer with 128 neurons and ReLU activation
* Output Layer with 10 neurons and Softmax activation

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### 5. Compile Model

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 6. Train Model

The model is trained for 50 epochs.

```python
model.fit(training_images, training_labels, epochs=50)
```

### 7. Evaluate Model

Model performance is evaluated on the test dataset.

```python
model.evaluate(test_images, test_labels)
```

### 8. Generate Predictions

Predictions are generated for unseen test images.

```python
model.predict(test_images)
```

## Results

The trained model achieves high classification accuracy on the Fashion MNIST test dataset and successfully identifies different categories of clothing items.

## Learning Outcomes

Through this project, the following concepts were explored:

* Image Classification
* Neural Networks
* TensorFlow and Keras
* Data Normalization
* Model Training and Evaluation
* Multi-class Classification
* Deep Learning Fundamentals

## Future Improvements

* Add Dropout layers to reduce overfitting.
* Use Convolutional Neural Networks (CNNs) for higher accuracy.
* Implement Early Stopping.
* Save and load trained models.
* Build a GUI/Web interface for image prediction.

## Author

Sindhu Nandipamula

B.Tech Information Technology
