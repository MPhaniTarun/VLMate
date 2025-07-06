# main.py
from flask import Flask, request, render_template, jsonify
from collections import deque
import base64
import io
import os # For setting up a templates folder
from PIL import Image
from io import BytesIO

# # This is without Unsloth
# from model import IMAGE_TO_TEXT_MODEL

# This is with Unsloth
from old_model import IMAGE_TO_TEXT_MODEL

app = Flask(__name__)

# Configure the templates folder (optional, Flask usually finds it if named 'templates')
# app.template_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))

# Using a deque to store the latest 7 images
# Each element will be the raw bytes of an image
latest_images_deque = deque(maxlen=7)

@app.route('/')
def index():
    """Renders the main index.html page."""
    return render_template('index.html')

@app.route('/uploadLatest', methods=['POST'])
def upload_latest():
    """
    Receives image files, adds their bytes to the deque,
    keeping only the latest 7 images.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No selected files'}), 400

    # Clear the deque before adding new images if you strictly want ONLY the 7 from this specific upload
    # If you want to continuously add and let the deque manage the maxlen, don't clear here.
    # Given the request "latest 7 images every second", it implies a rolling window, so we'll just extend.
    
    # Process each uploaded file
    for file in files:
        if file.filename == '':
            continue # Skip empty file parts

        # Read the image data as bytes
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes))

        latest_images_deque.append(image)
        print(f"Received file: {file.filename}, size: {len(image_bytes)} bytes. Deque size: {len(latest_images_deque)}")
    print("Latest 7 files description are: ", len(latest_images_deque))
    return jsonify({'message': f'Successfully received {len(files)} images. Total latest images: {len(latest_images_deque)}'}), 200

@app.route('/whatsAround', methods=['GET'])
def whats_around():
    """
    Returns the filenames of all images currently in the deque.
    """
    print("Latest 7 files description are: ", len(latest_images_deque))
    if not latest_images_deque:
        return jsonify({'message': 'No images captured yet.', 'filenames': []}), 200

    # filenames = ["test filenames 1", "test filenames 2", "test filenames 3"]
    # print("Latest 3 files description are: ", len(filenames), filenames)
    # return jsonify({'message': 'Here is a description of the images:', 'description': filenames}), 200

    # print("Latest 3 files description are: ", len(latest_images_deque), latest_images_deque)

    images_description = get_description_about_image()
    return jsonify({'message': 'Here are the latest captured file names:', 'description': images_description}), 200

def get_description_about_image():
    instruction = "What are the images about?"

    inference_output = IMAGE_TO_TEXT_MODEL.inference(latest_images_deque, instruction)
    inference_output = IMAGE_TO_TEXT_MODEL.postprocess(inference_output)

    return inference_output

if __name__ == '__main__':
    # Ensure a 'templates' directory exists for Flask to find index.html
    if not os.path.exists('templates'):
        os.makedirs('templates')

    app.run(debug=True, port=5000) # Run in debug mode for development, on port 5000
