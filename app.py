from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import torch
import timm
import datetime
import logging
import os
from torchvision import transforms
import torch.nn as nn
import requests
from urllib.parse import urlparse
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import random
from PIL import Image
from io import BytesIO
import tempfile
import json

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = timm.create_model("rexnet_150", pretrained=True, num_classes=60).to(device)

# Transform code
mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
tfs = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

class SaveFeatures():
    """ Extract pretrained activations"""
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

def getCAM(conv_fs, linear_weights, class_idx):
    bs, chs, h, w = conv_fs.shape
    cam = linear_weights[class_idx].dot(conv_fs[0, :, :, ].reshape((chs, h * w)))
    cam = cam.reshape(h, w)

    return (cam - np.min(cam)) / np.max(cam)

def test_single_image(model, device, image, final_conv, fc_params, cls_names=None):
    weight = np.squeeze(fc_params[0].cpu().data.numpy())
    activated_features = SaveFeatures(final_conv)

    # Move image to device and make prediction
    image = image.to(device)
    pred_class = torch.argmax(model(image.unsqueeze(0)), dim=1).item()

    # Get CAM for the predicted class
    heatmap = getCAM(activated_features.features, weight, pred_class)
    # Clean up
    activated_features.remove()

    return pred_class, heatmap

# Preprocess the image
def preprocess_image(image_path, transform=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)  # Convert to PIL Image
    if transform:
      image = transform(image)
    return image

# Dictionary for class names
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

with open('config.json') as config_file:
    config = json.load(config_file)

model_weights_path = config['model_path']
# Example usage:
m.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
m.eval()
final_conv, fc_params = m.features[-1], list(m.head.fc.parameters())

# Invert the dictionary to get class names by index
class_idx_to_name = {v: k for k, v in classes.items()}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image_path' not in request.json:
        raise BadRequest("Image path not present in request")
    image_path = request.json['image_path']
    if not image_path:
        raise BadRequest("Image path is empty")

    try:
        if image_path.startswith('http://') or image_path.startswith('https://'):
            response = requests.get(image_path)
            if response.status_code != 200:
                raise BadRequest(f"Unable to fetch image from URL: {image_path}")
            img = Image.open(BytesIO(response.content))
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            img.save(temp_file.name)
            temp_file_path = temp_file.name
            temp_file.close()
        else:
            temp_file_path = image_path
        
        # Apply preprocessing
        img = preprocess_image(temp_file_path, transform=tfs)
        
        # Make prediction
        pred_class, heatmap = test_single_image(
            model=m.to(device), device=device, image=img, 
            final_conv=final_conv, fc_params=fc_params, 
            cls_names=class_idx_to_name
        )
        
        # Clean up temporary file if it was used
        if image_path.startswith('http://') or image_path.startswith('https://'):
            os.remove(temp_file_path)

        return jsonify({'class_index': pred_class, 'class_name': class_idx_to_name[pred_class]})
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Error occurred, please try again'})


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)