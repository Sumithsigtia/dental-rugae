import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import tempfile

# Load the model
model = tf.keras.models.load_model('keras_model.h5')

# Load class labels
with open("labels.txt", "r") as file:
    class_labels = [line.strip() for line in file.readlines()]

# Define the function for image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = image.convert('RGB')  # Ensure image is in RGB mode
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Define the function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]  # Get the first (and only) prediction
    return predictions

# Streamlit app
st.title("Dental Rugae Classification Test App")

# Add an option to choose between image upload and camera capture

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        predictions = predict(image)

        # Highlight the predicted class
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]
        confidence = predictions[predicted_class_index]
        st.write(f"\n**Predicted Class:** {predicted_class}")


