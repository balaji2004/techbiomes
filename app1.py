from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Directory to store uploaded files and processed images
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load pre-trained AI model (replace with actual model)
model = tf.keras.models.load_model("ct_scan_model.h5")

# Function to convert DICOM to PNG
def dicom_to_png(dicom_path):
    """Convert DICOM to PNG format."""
    dicom_data = pydicom.dcmread(dicom_path)
    pixel_array = dicom_data.pixel_array

    # Normalize pixel values to 0-255
    pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
    pixel_array = np.uint8(pixel_array)

    # Convert to RGB format for visualization
    img = Image.fromarray(pixel_array).convert("RGB")

    # Save PNG file
    png_path = os.path.join(RESULTS_FOLDER, os.path.basename(dicom_path).replace(".dcm", ".png"))
    img.save(png_path)
    return png_path

# Function to preprocess image for AI model
def preprocess_image(image_path):
    """Resize and normalize image for model input."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.resize(img, (512, 512))  # Resize to 512x512
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to generate a heatmap using Grad-CAM
def generate_heatmap(image_path, model, layer_name="conv_last"):
    """Generate a Grad-CAM heatmap for explainability."""
    img_array = preprocess_image(image_path)

    # Get last convolutional layer
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    conv_output, predictions = grad_model(img_array)

    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch(conv_output)
        pred_class = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_class]

    grads = tape.gradient(pred_output, conv_output)[0]

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight layer activation map
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output[0]), axis=-1)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Overlay heatmap on original image
    img = cv2.imread(image_path)
    heatmap_resized = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)

    # Save heatmap
    heatmap_path = os.path.join(RESULTS_FOLDER, "heatmap_" + os.path.basename(image_path))
    cv2.imwrite(heatmap_path, overlay)
    return heatmap_path, float(predictions[0][pred_class])

# Flask API endpoint to handle uploads
@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle CT scan uploads, convert to PNG, and run AI model."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Ensure it's a DICOM file
    if not file.filename.lower().endswith(".dcm"):
        return jsonify({"error": "Invalid file format. Only DICOM (.dcm) allowed."}), 400

    # Save uploaded file
    dicom_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(dicom_path)

    # Convert DICOM to PNG
    png_path = dicom_to_png(dicom_path)

    # Preprocess image and make AI prediction
    preprocessed_img = preprocess_image(png_path)
    prediction = model.predict(preprocessed_img)
    predicted_label = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor"
    confidence = float(prediction[0][0]) * 100  # Convert to percentage

    # Generate heatmap
    heatmap_path, model_confidence = generate_heatmap(png_path, model)

    # Return AI result
    return jsonify({
        "prediction": predicted_label,
        "confidence": model_confidence,
        "heatmap_path": heatmap_path
    })

if __name__ == "__main__":
    app.run(debug=True)
