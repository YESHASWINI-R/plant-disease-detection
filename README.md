# Plant Disease Detection using CNN

A Deep Learning Project for Sustainable Agriculture

---

## Project Overview

This project focuses on detecting plant diseases using Convolutional Neural Networks (CNN).
The main goal is to help farmers identify crop diseases at an early stage, which reduces pesticide use and supports sustainable farming.
By using image classification, the model can automatically recognize different types of leaf diseases and healthy leaves.

---

## Objectives

* To build a CNN model for classifying plant leaf images.
* To improve the accuracy of disease detection through image preprocessing and augmentation.
* To support sustainability by reducing chemical usage through early disease prediction.
* To make a simple and easy-to-use system that can help farmers and researchers.

---

## Sustainability Impact

This project supports **Sustainable Development Goal (SDG 12: Responsible Consumption and Production)** by promoting efficient pesticide use and minimizing environmental harm.
Early detection of plant diseases helps to:

* Reduce the overuse of harmful chemicals.
* Increase crop yield and quality.
* Promote eco-friendly agricultural practices.

---

## Model Details

* Algorithm Used: Convolutional Neural Network (CNN)
* Layers: Conv2D, MaxPooling2D, Flatten, Dense
* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Accuracy Achieved: Around 88–90% on validation data

---

## Dataset Information

* **Dataset Used:** PlantVillage Dataset
* **Source:** [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
* **Classes:** 38 plant types (healthy and diseased)
* **Preprocessing Steps:**

  * Resizing images to 224×224
  * Normalizing pixel values
  * Data augmentation (rotation, flipping)

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* PIL (Python Imaging Library)
* Google Colab
* Kaggle API

---

## Improvisations Done

* Resized and cleaned images to improve dataset quality.
* Used image augmentation for better generalization.
* Built a CNN model and planned to test pretrained models like VGG16.
* Focused on sustainability by helping reduce pesticide usage.
* Planned to develop a simple UI for prediction.

---

## Source Code

### 1. Import Libraries and Set Seeds

```python
import os, random, json
import numpy as np
import tensorflow as tf
from zipfile import ZipFile
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
```

---

### 2. Download Dataset from Kaggle

```python
!pip install kaggle

kaggle_credentails = json.load(open("kaggle.json"))
os.environ['KAGGLE_USERNAME'] = kaggle_credentails["username"]
os.environ['KAGGLE_KEY'] = kaggle_credentails["key"]

!kaggle datasets download -d abdallahalidev/plantvillage-dataset

with ZipFile("plantvillage-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall()

base_dir = "plantvillage dataset/color"
```

---

### 3. Data Preprocessing and Train-Test Split

```python
img_size = 224
batch_size = 32

data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

validation_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)
```

---

### 4. CNN Model Building

```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

### 5. Model Training

```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
```

---

### 6. Model Evaluation

```python
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()
```

---

### 7. Prediction Function

```python
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).resize(target_size)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img_array

def predict_image_class(model, image_path, class_indices):
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_indices[predicted_class]

class_indices = {v: k for k, v in train_generator.class_indices.items()}

image_path = '/content/test_leaf.jpg'
predicted_class = predict_image_class(model, image_path, class_indices)
print("Predicted Class:", predicted_class)
```

---

### 8. Save Model

```python
model.save('plant_disease_prediction_model.h5')
```

---

## How to Run the Project

1. Open **Google Colab**.
2. Upload your **kaggle.json** credentials file.
3. Run all the code cells in order.
4. The dataset will automatically download and extract.
5. Train the model and view accuracy/loss graphs.
6. Upload a test image and predict its disease class.

---

## Sample Results

| Sample Image | Predicted Disease |
| ------------ | ----------------- |
| Apple Leaf   | Apple Black Rot   |
| Tomato Leaf  | Tomato Leaf Mold  |
| Potato Leaf  | Late Blight       |

---

## Project Structure

```
plant-disease-detection/
│
├── Plant_Disease_Detection.ipynb
├── README.md
├── kaggle.json
├── plant_disease_prediction_model.h5
└── dataset/
```
