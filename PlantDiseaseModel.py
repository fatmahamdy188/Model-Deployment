import os
import logging
from datetime import datetime
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms
import torch
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch.nn as nn


app = Flask(__name__)

# Configure the Flask app logger
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)

num_classes = 61


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        # Use RexNet-150 as the backbone
        self.backbone = timm.create_model("rexnet_150", pretrained=True)
        
        # Adjust the fully connected layer for your task
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        try:
            print("Before backbone:", x.shape)
            x = self.backbone(x)
            print("After backbone:", x.shape)
            print("Before fc:", x.shape)
            x = self.fc(x.view(x.size(0), -1))
            print("After fc:", x.shape)
            return x
        except Exception as e:
            app.logger.error(f"Error in forward pass: {str(e)}")
            raise e
# Instantiate the model
plant_disease_model = PlantDiseaseModel(num_classes)

# Print the architecture of the backbone model
#print(plant_disease_model.backbone)

model_weights_path = r'D:\Graduation Project\Project implementation\PlantifyApp.FlaskApis\disease_best_model.pth'


checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu'))

 # Load the state dictionary directly
plant_disease_model.eval()

# # Retrieve the connection string from the environment variable
# connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

# # Check if the connection string is set
# if connect_str:
#     result = {'status': 'success', 'message': 'AZURE_STORAGE_CONNECTION_STRING is set successfully', 'value': connect_str}
# else:
#     # Log a warning and set an error message
#     app.logger.warning('AZURE_STORAGE_CONNECTION_STRING is not set.')
#     result = {'status': 'error', 'message': 'AZURE_STORAGE_CONNECTION_STRING is not set.'}

# # Container name in which images will be stored in the storage account
# container_name = "picturesstorage"

# # Create a blob service client to interact with the storage account
# blob_service_client = BlobServiceClient.from_connection_string(conn_str=connect_str)

# try:
#     # Get container client to interact with the container in which images will be stored
#     container_client = blob_service_client.get_container_client(container=container_name)

# except Exception as e:
#     app.logger.error(f"Error accessing container: {str(e)}")
#     app.logger.info("Creating container...")
#     container_client = blob_service_client.create_container(container_name)

# Dictionary of plant disease classes
classes = {
    'Apple___alternaria_leaf_spot': 0,
    'Apple___black_rot': 1,
    'Apple___brown_spot': 2,
    'Apple___gray_spot': 3,
    'Apple___healthy': 4,
    'Apple___rust': 5,
    'Apple___scab': 6,
    'Bell_pepper___bacterial_spot': 7,
    'Bell_pepper___healthy': 8,
    'Blueberry___healthy': 9,
    'Cassava___bacterial_blight': 10,
    'Cassava___brown_streak_disease': 11,
    'Cassava___green_mottle': 12,
    'Cassava___healthy': 13,
    'Cassava___mosaic_disease': 14,
    'Cherry___healthy': 15,
    'Cherry___powdery_mildew': 16,
    'Corn___common_rust': 17,
    'Corn___gray_leaf_spot': 18,
    'Corn___healthy': 19,
    'Corn___northern_leaf_blight': 20,
    'Grape___black_measles': 21,
    'Grape___black_rot': 22,
    'Grape___healthy': 23,
    'Grape___isariopsis_leaf_spot': 24,
    'Grape_leaf_blight': 25,
    'Orange___citrus_greening': 26,
    'Peach___bacterial_spot': 27,
    'Peach___healthy': 28,
    'Potato___bacterial_wilt': 29,
    'Potato___early_blight': 30,
    'Potato___healthy': 31,
    'Potato___late_blight': 32,
    'Potato___nematode': 33,
    'Potato___pests': 34,
    'Potato___phytophthora': 35,
    'Potato___virus': 36,
    'Raspberry___healthy': 37,
    'Rice___bacterial_blight': 38,
    'Rice___blast': 39,
    'Rice___brown_spot': 40,
    'Rice___tungro': 41,
    'Soybean___healthy': 42,
    'Squash___powdery_mildew': 43,
    'Strawberry___healthy': 44,
    'Strawberry___leaf_scorch': 45,
    'Sugarcane___healthy': 46,
    'Sugarcane___mosaic': 47,
    'Sugarcane___red_rot': 48,
    'Sugarcane___rust': 49,
    'Sugarcane___yellow_leaf': 50,
    'Tomato___bacterial_spot': 51,
    'Tomato___early_blight': 52,
    'Tomato___healthy': 53,
    'Tomato___late_blight': 54,
    'Tomato___leaf_curl': 55,
    'Tomato___leaf_mold': 56,
    'Tomato___mosaic_virus': 57,
    'Tomato___septoria_leaf_spot': 58,
    'Tomato___spider_mites': 59,
    'Tomato___target_spot': 60
   
}

# Flask endpoint to upload a photo and make predictions
@app.route("/upload-photos-and-predict", methods=["POST"])
def upload_photos_and_predict():
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

            # Call the function to handle the upload and prediction
        prediction_result = upload_photo_and_predict(original_file_path, new_file_name)

        if prediction_result is not None:
            return jsonify(prediction_result)

        else:
            result = {"status": 404, "message": "File path does not exist"}
            return jsonify(result), 404  # Set the status code to 404

    except Exception as e:
        app.logger.error(f"Unhandled error during photo upload and prediction: {str(e)}")
        result = {"status": 500, "message": "Internal Server Error", "error_details": str(e)}
        return jsonify(result), 500  # Set the status code to 500


def upload_photo_and_predict(original_file_path, new_file_name):
    try:
        # Open the original image using PIL
        original_image = Image.open(original_file_path)

        # Apply the desired transformations (resize, normalize, etc.)
        transform = transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Convert RGBA to RGB if the image has an alpha channel
        if original_image.mode == 'RGBA':
            original_image = original_image.convert('RGB')

       
        # Apply the transformations to the original image
        transformed_image = transform(original_image)

      
        # Save the transformed image to a temporary file in JPEG format
        temp_file_path = "temp_transformed_image.jpg"
        transformed_image_pil = transforms.ToPILImage()(transformed_image)
        transformed_image_pil.save(temp_file_path, format="JPEG")

        # Upload the transformed file to the container using the file path
        # with open(temp_file_path, 'rb') as file:
        #     container_client.upload_blob(new_file_name, file)


        print(transformed_image.size())
        # Make predictions using the model
        prediction = predict_image(transformed_image)

        # Remove the temporary file
        os.remove(temp_file_path)

        app.logger.info(f"Successfully uploaded and predicted photo: {new_file_name}")

        return prediction

    except Exception as e:
        app.logger.error(f"Error during photo upload and prediction: {str(e)}")
        return {"status": 500, "message": "Internal Server Error", "error_details": str(e)}

def predict_image(transformed_image):
    try:
        # Ensure the model is in evaluation mode
        plant_disease_model.eval()

        # Print the shape of the input tensor
        print("Input tensor shape:", transformed_image.shape)

        print("Input tensor shape:", transformed_image.shape)
        # Add a batch dimension
        transformed_image = transformed_image.unsqueeze(0)
        print("Input tensor shape after unsqueeze:", transformed_image.shape)


        # Check if the input tensor has at least three dimensions
        if len(transformed_image.shape) < 3:
            raise ValueError("Input tensor should have at least three dimensions")

        print("Input tensor shape after unsqueeze:", transformed_image.shape)

        # Forward pass through the model
        with torch.no_grad():
            output = plant_disease_model(transformed_image)

        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=-1)[0]

        # Get the predicted class index
        predicted_class_index = torch.argmax(probabilities).item()

        # Directly access label using index
        predicted_class_label = list(classes.keys())[predicted_class_index]

        # Directly access label using index
        predicted_class = classes[predicted_class_label]

        # Create a JSON response
        result = {
            "status": 200,
            "message": "Prediction successful",
            "predicted_class_label": predicted_class_label,
            "predicted_class": predicted_class,
            "probabilities": probabilities.tolist(),
            "input_tensor_shape": transformed_image.shape[1:]  # Exclude batch dimension
        }

        return result

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return {"status": 500, "message": "Internal Server Error", "error_details": str(e)}


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