from flask_cors import CORS
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json

app = Flask(__name__)
CORS(app)

IMG_SIZE = (224,224)

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

# 🔥 Rebuild model (same as notebook)
image_input = tf.keras.Input(shape=(224, 224, 3))

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights=None,
    input_tensor=image_input
)

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="swish")(x)

meta_input = tf.keras.Input(shape=(3,))
m = tf.keras.layers.Dense(64, activation="relu")(meta_input)
m = tf.keras.layers.Dense(32, activation="relu")(m)

combined = tf.keras.layers.Concatenate()([x, m])
combined = tf.keras.layers.Dense(256, activation="swish")(combined)
combined = tf.keras.layers.Dropout(0.4)(combined)

output = tf.keras.layers.Dense(len(class_names), activation="softmax")(combined)

model = tf.keras.Model(inputs=[image_input, meta_input], outputs=output)

# Load weights
model.load_weights("vehicle_weights.h5")
print("✅ Model ready!")

# Metadata function
def extract_metadata(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)/255.0
    h,w,_ = img.shape
    return np.array([brightness, h/1000, w/1000],dtype=np.float32)

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    img = Image.open(file).convert("RGB").resize(IMG_SIZE)
    img_arr = np.array(img)

    img_arr = tf.keras.applications.efficientnet.preprocess_input(img_arr)
    img_arr = np.expand_dims(img_arr,axis=0)

    meta = extract_metadata(np.array(img))
    meta = np.expand_dims(meta,axis=0)

    preds = model.predict([img_arr,meta])
    idx = np.argmax(preds)

    return jsonify({
        "prediction": class_names[idx],
        "confidence": float(preds[0][idx]*100)
    })

if __name__ == "__main__":
    app.run(debug=True)