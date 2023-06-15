# AI Developer Dockerized Application

This repository contains a Dockerfile and Python code for running an AI-based application that performs image classification and segmentation. The application is built using Flask, a lightweight web framework in Python.

## Prerequisites
- Docker installed on your machine

## Usage

1. Clone the repository:

   ```
   git clone <repository_url>
   ```

2. Build the Docker image:

   ```
   docker build -t ai-application .
   ```

3. Run the Docker container:

   ```
   docker run -p 5001:5000 -d ai-application
   ```

4. Access the application in your web browser by navigating to `http://localhost:5001`.

## Endpoints

The application provides the following endpoints:

- `POST /classify_and_segment`: This endpoint accepts an image file or URL as input and performs image classification and segmentation. It returns the top 5 predicted classes along with the segmented image encoded in base64 format.

- `GET /list_files`: This endpoint lists all the files in the current directory.

## Additional Files

- `ImageNetLabels.txt`: This file contains the labels for the ImageNet dataset used for image classification.

- `model.py`: This file contains code for loading the image, preprocessing it, and segmenting the image.

- `inference.py`: This file contains the code for performing inference on the loaded image.

- `data/`: This directory contains additional data files required by the application.

## Important Notes

- The application uses the `opencv-python-headless` library with version `4.5.3.56` for image processing. The specific version is installed to ensure compatibility.

- The application runs on port `5001` within the Docker container and is mapped to the same port on the host machine.

- The application assumes that the image classification model and other required files are present in the current directory during the Docker build process.

- Ensure that the necessary Python packages mentioned in `requirements.txt` are available for installation before running the Docker build command.

- The application is set to accept image files or URLs in the `image` field of a multipart/form-data POST request or in the `image_url` field of a JSON POST request.

- The segmented image is returned in base64 format for ease of transmission in JSON.

Feel free to modify the `app.py` file and other relevant files according to your requirements.

Please note that this README assumes familiarity with Docker and basic web development concepts. If you encounter any issues or have questions, feel free to reach out for assistance.