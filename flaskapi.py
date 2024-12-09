from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.applications.resnet import preprocess_input

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("resnet_model")

# Define image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to model's input size
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return preprocess_input(img_array)  # Apply the ResNet preprocessing

# API endpoint to predict from an image
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Get the file from the request
    file = request.files["file"]

    try:
        # Save the file temporarily
        file_path = f"./temp_image2.jpg"
        file.save(file_path)

        # Preprocess the image
        processed_image = preprocess_image(file_path)

        # Predict
        prediction = model.predict(processed_image)[0][0]  # Extract single value
        result = True if prediction > 0.5 else False

        # Return the result as JSON
        return jsonify({"prediction": result, "confidence": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run( host="0.0.0.0", port=5000)
