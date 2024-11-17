from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
import os

import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Initialize Flask app
app = Flask(__name__)


# # Desired image dimensions
img_width = 200
img_height = 50

# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4

# # Maximum length of any captcha in the dataset
max_length = 10

# Define Arabic and English numerals as strings
arabic_numerals = ['/', '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩']
english_numerals = ["/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Create StringLookup layer to map Arabic to indices
char_to_num = layers.StringLookup(vocabulary=arabic_numerals, oov_token="", mask_token=None)

# Create a StringLookup layer with an invert option to map indices back to English numerals
num_to_char = layers.StringLookup(vocabulary=english_numerals, invert=True, mask_token=None)



# Load the ONNX model# Create inference session with rt.InferenceSession
providers = ['CPUExecutionProvider']
onnx_session = ort.InferenceSession('pred_model.onnx', providers=providers)


# # Try different providers if available (e.g., 'CPUExecutionProvider', 'CUDAExecutionProvider')
# try:
#     onnx_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider']) 
# except Exception as e:
#     print(f"Error with CPUExecutionProvider: {e}")
#     try:
#         onnx_session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider'])
#     except Exception as e:
#         print(f"Error with CUDAExecutionProvider: {e}")
#         print("Consider upgrading onnxruntime or re-exporting your model with a lower opset version.")

# Get input and output names for the ONNX model
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# Helper function to preprocess the image
def preprocess_image(image_path):
    """
    Preprocess the input image to match the ONNX model's expected input shape.
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((50, 200))  # Resize to (50, 200)
    image = np.array(image).astype("float32") / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

# Define routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Save the uploaded file
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)  # Create the directory if it doesn't exist
        file.save(filepath)

        # Preprocess the image and make a prediction
        try:
            image = preprocess_image(filepath)
            # Run inference with the ONNX model
            predictions = onnx_session.run([output_name], {input_name: image})
            # preds = prediction_model.predict(batch_images)
            pred_texts = decode_batch_predictions(predictions[0])

            # label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            # predicted_class = np.argmax(predictions[0], axis=-1).tolist()  # Get predicted class
            return jsonify({"predicted_class": pred_texts})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.remove(filepath)  # Clean up the saved file

    # Render a simple HTML page for uploads
    return render_template("index.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)