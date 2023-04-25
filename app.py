import os
from flask import Flask, request, jsonify
import numpy as np
from inference import infer
from model import load_image, preprocess_image
from flask_cors import CORS
from heapq import nlargest
app = Flask(__name__)
CORS(app)
with open("ImageNetLabels.txt", "r") as f:
    class_labels = f.read().splitlines()[1:]

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

    #top_k = 5
    #top_indices = nlargest(top_k, range(len(probabilities)), probabilities.take)
    #top_probabilities = probabilities[top_indices]
    #top_5_predictions = [class_labels[i] for i in top_indices]
    #print(top_indices, top_probabilities, top_5_predictions)

    #return jsonify({"top_classes": top_class_names, "top_probabilities": top_probabilities.tolist()})


    #top_5 = 5  # Change this to display a different number of predictions
    #top_5_indices = np.argpartition(probabilities, -top_5)[-top_5:]
    #top_5_predictions = [(class_labels[index], float(probabilities[index])) for index in top_5_indices]

    probabilities  = np.squeeze(probabilities)
    top_5_indices = np.argsort(probabilities)[-5:]#[::-1]
    print(top_5_indices)
    probabilities = np.array(probabilities, dtype=np.float32)  # Add this line
    top_5_predictions = [(class_labels[index], float(probabilities[index])) for index in top_5_indices]

    return jsonify({"top_5_predictions": top_5_predictions})


@app.route('/list_files')
def list_files():
    files = os.listdir('.')
    return {'files': files}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
