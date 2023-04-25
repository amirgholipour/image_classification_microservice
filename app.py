# app.py
import os
from flask import Flask, request, jsonify
import numpy as np
from inference import infer
from model import load_image, preprocess_image

app = Flask(__name__)

with open("ImageNetLabels.txt", "r") as f:
    class_labels = f.read().splitlines()[1:]  # Skip the first line, which is 'background'

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' in request.files:
        file = request.files['image']
    elif 'image_url' in request.json:
        file = request.json['image_url']
    else:
        return jsonify({"error": "No image file or URL provided"}), 400

    image, _ = load_image(file)
    probabilities = infer(image)
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_labels[predicted_class_index]
    # Print the list of existing files
    file_list = os.listdir('/tmp')
    print('List of existing files:')
    for f in file_list:
        print(f)




    return jsonify({"predicted_class_name": predicted_class_name})


    # Return the response as JSON
    # return jsonify(probabilities.tolist())
@app.route('/list_files')
def list_files():
    files = os.listdir('.')
    return {'files': files}
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
