# 🌿 Plant Disease Detection using CNN

## 📘 Project Overview

The **Plant Disease Detection** project uses **Deep Learning (CNN)** to automatically identify various plant leaf diseases from images.
By analyzing leaf images, the model predicts whether the plant is healthy or diseased — and if diseased, which specific disease it has.
This approach supports **precision agriculture**, helping farmers take quick and accurate action to prevent crop loss.

---

## 🌍 Introduction

Plant diseases are one of the major causes of reduced crop yield worldwide. Traditionally, identifying diseases requires expert knowledge, which is time-consuming and expensive.
This project provides an **AI-based automated system** that uses **Convolutional Neural Networks (CNNs)** to detect plant diseases from images of leaves.

By using deep learning, farmers can quickly detect diseases, minimize pesticide misuse, and adopt sustainable farming techniques — reducing costs and promoting environmental safety.

---

## 🧠 Objectives

* Detect whether a plant leaf is **healthy** or **diseased**.
* Classify the disease type accurately using **CNN-based image classification**.
* Reduce dependency on manual inspection and expert diagnosis.
* Promote **eco-friendly and sustainable agriculture**.
* Contribute to **UN Sustainable Development Goals (SDG 12 & SDG 15)**.

---

## ⚙️ Tools and Technologies

| Component      | Description                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------- |
| **Language**   | Python                                                                                      |
| **Libraries**  | TensorFlow, Keras, NumPy, Matplotlib, Seaborn, Scikit-learn                                 |
| **Dataset**    | [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| **Platform**   | Google Colab                                                                                |
| **Model Type** | Convolutional Neural Network (CNN)                                                          |

---

## 🧩 Workflow

1. Data collection and preprocessing using **Kaggle dataset**.
2. Image augmentation for better generalization.
3. CNN model design and training using Keras.
4. Model evaluation using metrics such as accuracy, precision, recall, and F1-score.
5. Prediction on unseen leaf images.
6. Visualization of results with confidence score.

---

## ⚙️ Source Code

### **1️⃣ Install & Import Libraries**

```python
!pip install tensorflow matplotlib

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import random
```

---

### **2️⃣ Upload & Extract Dataset**

```python
from google.colab import drive
drive.mount('/content/drive')

import os, json
from zipfile import ZipFile

kaggle_dict = json.load(open("kaggle.json"))
os.environ["KAGGLE_USERNAME"] = kaggle_dict["username"]
os.environ["KAGGLE_KEY"] = kaggle_dict["key"]

!kaggle datasets download -d abdallahalidev/plantvillage-dataset

with ZipFile("plantvillage-dataset.zip", "r") as zip_ref:
    zip_ref.extractall("/content/plantvillage_dataset")

print("Dataset extracted successfully!")
print(os.listdir("/content/plantvillage_dataset"))
```

---

### **3️⃣ Set Random Seeds**

```python
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
```

---

### **4️⃣ Data Preparation and Augmentation**

```python
base_dir = "/content/plantvillage_dataset/plantvillage dataset/color"
img_size = 128
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=seed_value
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=seed_value
)
```

---

### **5️⃣ CNN Model Architecture**

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_data.class_indices), activation='softmax')
])

model.summary()
```

---

### **6️⃣ Compile and Train the Model**

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)
```

---

### **7️⃣ Visualize Model Performance**

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

### **8️⃣ Evaluate Model Performance**

```python
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")
```

---

### **9️⃣ Classification Report & Confusion Matrix**

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

true_labels = val_data.classes
predictions = model.predict(val_data)
predicted_labels = np.argmax(predictions, axis=1)
class_names = list(val_data.class_indices.keys())

print(classification_report(true_labels, predicted_labels, target_names=class_names))

cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
```

---

### **🔟 Save Model**

```python
model.save("plant_disease_cnn.h5")
print("Model saved successfully!")
```

---

### **🧪 Predict Disease from New Images**

```python
from tensorflow.keras.preprocessing import image
from google.colab import files

def predict_plant_disease(model, train_data):
    uploaded = files.upload()
    for img_name in uploaded.keys():
        img_path = img_name
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        pred = model.predict(img_array)
        predicted_class = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred) * 100
        class_labels = list(train_data.class_indices.keys())

        plt.figure(figsize=(4, 4))
        plt.imshow(image.load_img(img_path))
        plt.axis('off')
        plt.title(f"Prediction: {class_labels[predicted_class]}\nConfidence: {confidence:.2f}%", fontsize=12)
        plt.show()

        print(f"Predicted Class: {class_labels[predicted_class]}")
        print(f"Model Confidence: {confidence:.2f}%\n")

predict_plant_disease(model, train_data)
```

---

## 📊 Evaluation Results

| Metric                  |  Score |
| :---------------------- | :----: |
| **Validation Accuracy** | 89.88% |
| **Precision**           |  0.03  |
| **Recall**              |  0.03  |
| **F1 Score**            |  0.03  |

---

## 🌱 Sustainability Impact

This project directly aligns with the **United Nations Sustainable Development Goals (SDGs)**, especially:

* **SDG 12: Responsible Consumption and Production**

  * Reduces excessive use of pesticides by enabling early and accurate disease detection.
  * Promotes efficient resource utilization and minimizes crop wastage.

* **SDG 15: Life on Land**

  * Encourages healthier agricultural ecosystems.
  * Helps maintain biodiversity by preventing the spread of harmful crop diseases.

**Overall Environmental Impact:**

* Minimizes the use of harmful chemicals in agriculture.
* Improves food security by increasing yield and quality.
* Promotes **sustainable, tech-driven, and environment-friendly farming practices**.

---

## 🚀 Future Enhancements

* Integrate **mobile app** for real-time field predictions.
* Expand dataset with **more crops and diseases**.
* Connect with **IoT-based leaf monitoring sensors**.
* Deploy model on **cloud platforms** like AWS or Firebase for remote access.

---
