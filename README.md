---
title: Vehicle Type Recognition
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🚗 Vehicle Type Recognition System

A deep learning-based system that classifies vehicle types such as **Car, Bus, Truck, and Motorcycle** using an EfficientNet-based multimodal model.

## 🔥 Project Overview

This project uses **EfficientNetB0** with **multimodal features (image + metadata)** for vehicle classification.

It includes a modern interactive UI and a Flask backend API for real-time predictions.

## 🚀 Live Demo

Try Vehicle Type Recognition online:

👉 [**Project Live Demo**](https://huggingface.co/spaces/chillx123/vehicle-type-recognition)

## 🎯 Features

* 🚗 Vehicle classification: Car, Bus, Truck, Motorcycle
* 🧠 EfficientNetB0-based deep learning model
* 🔀 Multimodal input using image and metadata
* 🔬 Fine-grained vehicle classification
* 🌐 Flask REST API for predictions
* 🎨 Professional interactive UI

  * Dark/Light mode toggle
  * Animated background
  * Image preview
  * Confidence progress bar
  * Class-based visual highlighting
* ⚡ Real-time prediction results

## 🛠️ Tech Stack

### Machine Learning

* TensorFlow / Keras
* EfficientNetB0
* NumPy
* OpenCV
* Pillow

### Backend

* Flask
* Flask-CORS
* Gunicorn

### Frontend

* HTML
* CSS
* JavaScript

### Deployment

* Docker
* Hugging Face Spaces
* Git / Git LFS

## 📁 Project Structure

```text
vehicle-type-recognition/
│
├── app.py
├── index.html
├── vehicle_weights.h5
├── class_names.json
├── requirements.txt
├── Dockerfile
├── README.md
└── sample_images/
```

## 🚀 How It Works

1. User uploads a vehicle image.
2. The image is sent to the Flask backend.
3. The backend preprocesses the image and extracts metadata.
4. The EfficientNet-based model performs the prediction.
5. The prediction result is returned through the API.
6. The UI displays the predicted vehicle type and confidence score.

## 📊 Output

The application provides:

* Predicted vehicle type
* Confidence percentage
* Visual prediction feedback

## 🌐 Deployment

The application is deployed as a Docker-based Hugging Face Space.

The trained model weights and project images are managed using **Git LFS**.
