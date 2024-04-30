import os
import logging
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms
import torch
import io
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# Configure the Flask app logger
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)

UPLOAD_FOLDER = '/workspaces/Model-Deployment/Images'  # Update with your desired upload folder path
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask endpoint to upload a photo using a file path from the query parameter or JSON parameter
@app.route("/upload-photos", methods=["POST"])
def upload_photos():
    try:
        if 'file' not in request.files:
            result = {"status": 400, "message": "No file provided"}
            return jsonify(result), 400

        file = request.files['file']
        # Check if the file is an image
        if file.filename == '':
            result = {"status": 400, "message": "No file selected"}
            return jsonify(result), 400
        if file and allowed_file(file.filename):
            # Save the uploaded image to a temporary file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the image and return the result
            upload_result = process_image(file_path)
            os.remove(file_path)  # Remove the temporary file

            if upload_result is None:
                result = {"status": 200, "message": "Photo uploaded and processed successfully"}
                return jsonify(result), 200
            else:
                return upload_result

    except Exception as e:
        app.logger.error(f"Unhandled error during photo upload: {str(e)}")
        result = {"status": 500, "message": "Internal Server Error", "error_details": str(e)}
        return jsonify(result), 500

def process_image(file_path):
    try:
        # Open the original image using PIL
        original_image = Image.open(file_path)

        # Convert RGBA to RGB if the image has an alpha channel
        if original_image.mode == 'RGBA':
            original_image = original_image.convert('RGB')

        # Apply the desired transformations (resize, normalize, etc.)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Apply the transformations to the original image
        transformed_image = transform(original_image)

        # Display the original and transformed images (optional)
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(original_image))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(transformed_image.permute(1, 2, 0)))  # Convert tensor to numpy array
        plt.title("Transformed Image")
        plt.axis('off')

        plt.show()

        # You can perform further processing or analysis here

        return None  # Success

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return {"status": 500, "message": "Error processing image", "error_details": str(e)}

# Custom error handler for 404 errors
@app.errorhandler(404)
def not_found_error(error):
    result = {"status": 404, "message": "Endpoint not found"}
    return jsonify(result), 404

# Custom error handler for generic errors
@app.errorhandler(Exception)
def handle_generic_error(error):
    app.logger.error(f"Unhandled error: {str(error)}")
    result = {"status": 500, "message": "Internal Server Error", "error_details": str(error)}
    return jsonify(result), 500

if __name__ == '__main__':
    app.run(debug=True)
