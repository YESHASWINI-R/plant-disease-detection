# plant-disease-detection
A CNN-based image classification project for detecting plant diseases.
Of course 👍 Here’s a **clean, professional, emoji-free version** of your README.md — perfectly formatted for direct copy-paste into GitHub:

---

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

This project supports Sustainable Development Goal (SDG 12: Responsible Consumption and Production) by promoting efficient pesticide use and minimizing environmental harm.
Early detection of plant diseases helps to:

* Reduce the overuse of harmful chemicals.
* Increase crop yield and quality.
* Promote eco-friendly agricultural practices.

---

## Model Details

* Algorithm Used: Convolutional Neural Network (CNN)
* Layers: Conv2D, Batch Normalization, MaxPooling, Dropout, and Dense layers
* Optimizer: Adam (learning rate = 0.0005)
* Loss Function: Categorical Crossentropy
* Accuracy Achieved: Around 95% on validation data

---

## Dataset Information

* Dataset Used: PlantVillage Dataset
* Source: Kaggle ([https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset))
* Classes: Multiple plant species with healthy and diseased leaves
* Image Type: Colored leaf images
* Preprocessing Steps:

  * Resizing images to 224x224
  * Normalizing pixel values (0–1 range)
  * Applying data augmentation (rotation, flipping, zooming)

---

## Technologies Used

* Programming Language: Python
* Libraries and Tools:

  * TensorFlow / Keras
  * NumPy
  * Matplotlib
  * PIL (Python Imaging Library)
  * Google Colab
  * Kaggle Datasets

---

## Improvisations Done

* Resized and cleaned images to improve dataset quality.
* Used image augmentation (rotation, flipping) for better accuracy.
* Built a CNN model and planned to experiment with pretrained models like VGG16.
* Focused on sustainability by helping reduce pesticide usage through early detection.
* Planning to create a simple user interface for easy use in future.

---

## How to Run the Project

1. Open Google Colab.
2. Upload the `Plant_Disease_Detection.ipynb` notebook file.
3. Upload your `kaggle.json` file to access the dataset.
4. Run all code cells one by one.
5. The model will train and show accuracy and loss graphs.
6. You can upload a leaf image to get disease prediction results.

---

## Sample Results

| Sample Image | Predicted Disease   |
| ------------ | ------------------- |
| Apple Leaf   | Apple Rust          |
| Tomato Leaf  | Tomato Mosaic Virus |
| Healthy Leaf | Healthy             |

---

## Project Structure

```
plant-disease-detection/
│
├── Plant_Disease_Detection.ipynb
├── README.md
├── requirements.txt
└── dataset_link.txt
```

---

## Future Work

* Build a web or mobile interface for real-time prediction.
* Use transfer learning with pretrained models (VGG16, ResNet50).
* Deploy the model using Flask or Streamlit for easy accessibility.

---

Would you like me to write a short **requirements.txt** file next (listing only the libraries needed for this project)? You can upload that to your repo too.
