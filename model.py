# model.py
import requests
from io import BytesIO
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

model_handle = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2"
def preprocess_image(image):
    image = np.array(image)
    img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
    image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
    return image

#def load_image(image_path, image_size=224):
#    image = Image.open(image_path)
#    image = image.resize((image_size, image_size))
#    return preprocess_image(image)



def load_image(image_path_or_stream, target_size=(224, 224)):
    if isinstance(image_path_or_stream, str) and (image_path_or_stream.startswith("http://") or image_path_or_stream.startswith("https://")):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
        }
        response = requests.get(image_path_or_stream, headers=headers)
        content_type = response.headers.get('content-type')

        if response.status_code != 200 or 'image' not in content_type:
            raise Exception(f"Failed to fetch image from URL, status code: {response.status_code}, content type: {content_type}")

        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_stream)
    
    image = image.resize(target_size)
    return np.asarray(image), image
    #return preprocess_image(image), image


def load_model():
    return hub.load(model_handle)




def classify_image(model, image):
    # Convert the image to float32 and normalize pixel values
    image = np.asarray(image, dtype=np.float32) / 255.0

    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)

    # Run the model on the input image
    logits = model(image)
    probabilities = tf.nn.softmax(logits)
    return probabilities.numpy()



model = load_model()
