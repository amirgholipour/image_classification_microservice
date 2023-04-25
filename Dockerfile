# Use the official lightweight Python image as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of your application code into the container
# COPY . .
COPY data /app
COPY ImageNetLabels.txt /app/ImageNetLabels.txt
COPY model.py /app/model.py
COPY inference.py /app/inference.py
COPY app.py /app/app.py
# Expose the port the app runs on
EXPOSE 5000

# Start the application
CMD ["python", "app.py"]
