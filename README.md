# 🚗 Vehicle Type Recognition System

A deep learning-based system that classifies vehicle types (Car, Bus, Truck, Motorcycle) using an EfficientNet-based multimodal model.

---

## 🔥 Project Overview

This project leverages **EfficientNet(B0)**,**multimodal features (image + metadata)** along with **fine-grained classification** to accurately classify vehicle types.
It also includes a **modern interactive UI** and a **Flask backend API** for real-time predictions.

---

## 🎯 Features

* 🚗 Vehicle classification (Car, Bus, Truck, Motorcycle)
* 🧠 EfficientNet-based deep learning model
* 🔀 Multimodal input (image + brightness + dimensions)
* Fine-grained classification
* 🌐 Flask REST API for predictions
* 🎨 Professional UI with:

  * Dark/Light mode toggle
  * Animated background
  * Image preview
  * Confidence progress bar
  * Class-based color highlighting
* ⚡ Real-time prediction results

---

## 🛠️ Tech Stack

### 🔹 Machine Learning

* TensorFlow / Keras
* EfficientNetB0
* NumPy
* OpenCV

### 🔹 Backend

* Flask
* Flask-CORS

### 🔹 Frontend

* HTML
* CSS (Glassmorphism UI)
* JavaScript

---

## 📁 Project Structure

```
vehicle-type-recognition/
│
├── app.py                # Flask backend
├── index.html            # Frontend UI
├── vehicle_weights.h5    # Trained model weights
├── class_names.json      # Class labels
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── sample_images/        # Test images
```

## 🚀 How It Works

1. User uploads an image
2. Image is sent to Flask backend
3. Preprocessing + metadata extraction
4. Model predicts vehicle type
5. Result is sent back to UI
6. UI displays:

   * Prediction label
   * Confidence score
   * Progress bar visualization

---

## 📊 Output

* Predicted vehicle type
* Confidence percentage
* Visual feedback with UI enhancements

---

