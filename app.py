import os
import logging
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms
import torch
import io
import jsonp
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# Configure the Flask app logger
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)

# Retrieve the connection string from the environment variable
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

# Check if the connection string is set
if connect_str:
    result = {'status': 'success', 'message': 'AZURE_STORAGE_CONNECTION_STRING is set successfully', 'value': connect_str}
else:
    # Log a warning and set an error message
    app.logger.warning('AZURE_STORAGE_CONNECTION_STRING is not set.')
    result = {'status': 'error', 'message': 'AZURE_STORAGE_CONNECTION_STRING is not set.'}

# Container name in which images will be stored in the storage account
container_name = "picturesstorage"

# Create a blob service client to interact with the storage account
blob_service_client = BlobServiceClient.from_connection_string(conn_str=connect_str)

try:
    # Get container client to interact with the container in which images will be stored
    container_client = blob_service_client.get_container_client(container=container_name)

except Exception as e:
    app.logger.error(f"Error accessing container: {str(e)}")
    app.logger.info("Creating container...")
    container_client = blob_service_client.create_container(container_name)


# Flask endpoint to upload a photo using a file path from the query parameter or JSON parameter
@app.route("/upload-photos", methods=["POST"])
def upload_photos():
    try:
        # Check if the request has JSON data
        if request.is_json:
            data = request.get_json()

            # Extract the file path from the JSON data
            original_file_path = data.get('file_path')
        else:
            # Extract the file path from the query parameter
            original_file_path = request.args.get('file_path')

        # Check if the file path exists
        if os.path.exists(original_file_path):
            # Generate a timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # Extract the file extension
            file_extension = os.path.splitext(original_file_path)[1]

            # Generate a new file name with timestamp appended
            new_file_name = f"{timestamp}_{os.path.basename(original_file_path)}"

            # Call the function to handle the upload with resizing and prediction
            upload_result = upload_photo(original_file_path, new_file_name)

            if upload_result is None:
                # Get the URL of the uploaded blob
                blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{new_file_name}"

                result = {"status": 200, "message": "Photo uploaded successfully", "new_file_path": blob_url}
                return jsonify(result), 200  # Set the status code to 200
            else:
                # Return the error response from the upload function
                return upload_result

        else:
            result = {"status": 404, "message": "File path does not exist"}
            return jsonify(result), 404  # Set the status code to 404

    except Exception as e:
        app.logger.error(f"Unhandled error during photo upload: {str(e)}")
        result = {"status": 500, "message": "Internal Server Error", "error_details": str(e)}
        return jsonify(result), 500  # Set the status code to 500

def upload_photo(original_file_path, new_file_name):
    try:
        # Open the original image using PIL
        original_image = Image.open(original_file_path)

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

        # Display the original and transformed images
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(original_image))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(transformed_image.permute(1, 2, 0)))  # Convert tensor to numpy array
        plt.title("Transformed Image")
        plt.axis('off')

        plt.show()

        # Save the transformed image to a temporary file in JPEG format
        temp_file_path = "temp_transformed_image.jpg"
        transformed_image_pil = transforms.ToPILImage()(transformed_image)
        transformed_image_pil.save(temp_file_path, format="JPEG")

        blob_name = os.path.basename(new_file_name)
   

        # Upload the transformed file to the container using the file path
        with open(temp_file_path, 'rb') as file:
            container_client.upload_blob(blob_name, file)

        # Remove the temporary file
        os.remove(temp_file_path)

        app.logger.info(f"Successfully uploaded photo: {blob_name}")

        return None

    except Exception as e:
        app.logger.error(f"Error uploading photo: {str(e)}")
        return str(e)

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

# The rest of your code goes here

if __name__ == '__main__':
    app.run(debug=True)
