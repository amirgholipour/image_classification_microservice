# model.py
import requests
from io import BytesIO
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import cv2

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


def visualize_segmentation(image, segmentation_map, alpha=0.7):
    # Create a color map
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.arange(cmap.N))

    # Map the segmentation to RGB colors
    colored_segmentation = colors[segmentation_map]

    # Blend the original image with the colored segmentation
    merged_image = cv2.addWeighted(image / 255.0, alpha, colored_segmentation[:, :, :3], 1 - alpha, 0)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(merged_image)
    plt.title("Merged Segmented Image")

    plt.show()
    return merged_image

# # Visualize the segmentation results
# merged_image = visualize_segmentation(image, segmentation_map)

# def segment_image(image):
#     image_tensor = tf.image.resize(tf.expand_dims(tf.keras.preprocessing.image.img_to_array(image), 0), (513, 513))
#     segmentation_map = tf.argmax(seg_model(image_tensor), axis=-1)[0].numpy()

#     segmented_image = Image.fromarray((segmentation_map* 255).astype(np.uint8), "P")
#     segmented_image.putpalette([
#         0, 0, 0,       # Black for the background
#         255, 0, 0,     # Red for the segmented object
#     ])
#     segmented_image = segmented_image.convert("RGBA")
#     pixels = segmented_image.load()

#     for i in range(segmented_image.size[0]):
#         for j in range(segmented_image.size[1]):
#             if pixels[i, j] == 0:
#                 pixels[i, j] = (0, 0, 0, 0)

#     buffer = BytesIO()
#     segmented_image.save(buffer, format="PNG")
#     buffer.seek(0)
#     # mask = Image.fromarray(segmented_image)

#     # # Apply the mask on the original image
#     # masked_image = Image.composite(image, ImageOps.colorize(mask, "#000", "#FFF"), mask)

#     # # Save the masked image as PNG
#     # buffer = BytesIO()
#     # masked_image.save(buffer, format="PNG")
#     # buffer.seek(0)

#     return buffer
#segmented_image is in "P" mode (palette-based image) and ImageOps.colorize expects an "L" mode (grayscale) image.
# model_url = "https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1"
# mask_rcnn_model = hub.load(model_url)
seg_model = hub.load('https://tfhub.dev/google/HRNet/coco-hrnetv2-w48/1')

def segment_image(image):
    # Convert the PIL image to a tensor
    input_size = (512, 512)

    image_tensor = tf.convert_to_tensor(image.astype(np.uint8))
    image_tensor = tf.expand_dims(image_tensor, 0)
    image_tensor = tf.image.resize(image_tensor, input_size)
    image_tensor = tf.cast(image_tensor, tf.float32)

    # # Run inference on the preprocessed image
    # image = tf.cast([image], tf.float32)/255.0

    # input_tensor = tf.expand_dims(image, 0) / 255.0
    output_tensor = seg_model(image_tensor)


    # Extract the segmentation map and convert it to a numpy array
    segmentation_map = tf.argmax(output_tensor, axis=-1)
    segmentation_map = segmentation_map[0].numpy()
    # image = np.array(image)
    image = cv2.resize(image, (segmentation_map.shape[1], segmentation_map.shape[0]))
    masked_image = visualize_segmentation(image, segmentation_map)

    masked_image = Image.fromarray(np.uint8(masked_image*255))
    # Save the masked image as PNG
    buffer = BytesIO()
    masked_image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer







model = load_model()
