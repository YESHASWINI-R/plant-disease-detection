# 🌿 Plant Disease Detection using CNN

## 📘 Project Overview

This project detects plant leaf diseases using a **Convolutional Neural Network (CNN)** model trained on the **PlantVillage Dataset**.
It helps farmers and researchers identify plant diseases early, supporting sustainable farming and reducing pesticide overuse.

---

## 🧠 Objectives

* Identify whether a plant leaf is **healthy or diseased**.
* Improve crop quality and yield using AI-based disease prediction.
* Promote **eco-friendly agricultural practices**.

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

# Load Kaggle credentials
kaggle_dict = json.load(open("kaggle.json"))
os.environ["KAGGLE_USERNAME"] = kaggle_dict["username"]
os.environ["KAGGLE_KEY"] = kaggle_dict["key"]

# Download dataset
!kaggle datasets download -d abdallahalidev/plantvillage-dataset

# Extract dataset
with ZipFile("plantvillage-dataset.zip", "r") as zip_ref:
    zip_ref.extractall("/content/plantvillage_dataset")

print("Dataset extracted successfully!")
print(os.listdir("/content/plantvillage_dataset"))
```

**Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---

### **3️⃣ Set Random Seeds for Reproducibility**

```python
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
```

---

### **4️⃣ Prepare Data Generators with Augmentation**

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

print("Number of Classes:", len(train_data.class_indices))
```

---

### **5️⃣ Build CNN Model**

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

### **6️⃣ Train the Model**

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)
```

---

### **7️⃣ Plot Training and Validation Accuracy**

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

### **9️⃣ Generate Classification Report & Confusion Matrix**

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

true_labels = val_data.classes
predictions = model.predict(val_data)
predicted_labels = np.argmax(predictions, axis=1)
class_names = list(val_data.class_indices.keys())

# Classification Report
print("Classification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Overall Metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
```

---

### **🔟 Save Model**

```python
model.save("plant_disease_cnn.h5")
print("Model saved successfully!")
```

---

### **🧪 Test Model with New Images**

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

# Call the function
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

This project contributes to **UN Sustainable Development Goal (SDG 12: Responsible Consumption and Production)**.
By enabling early detection of plant diseases, it helps to:

* Reduce unnecessary pesticide use.
* Increase crop yield and product quality.
* Encourage sustainable and environment-friendly farming practices.

---

## 🚀 Future Enhancements

* Deploy the model as a **mobile or web app** for real-time predictions.
* Include **more crop species and disease types**.
* Integrate **IoT and drone-based image collection** for live monitoring.

---
