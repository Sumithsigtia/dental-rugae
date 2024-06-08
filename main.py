import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model without compiling
model = tf.keras.models.load_model('keras_model.h5', compile=False)

# Define a function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match model input size
    img = img.convert('RGB')  # Ensure image is in RGB mode
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define a function to make predictions
def predict(image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    confidence = np.max(predictions)
    # Update classes to match the Teachable Machine classes
    classes = ['Average', 'Horizontal', 'Vertical']
    predicted_class = classes[np.argmax(predictions)]
    return predicted_class, confidence * 100  # Convert confidence to percentage

# Streamlit app
st.title("Dental Rugae Classification")
st.write("Upload a dental rugae image to classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Make prediction
    predicted_class, confidence = predict(image)

    # Display the result
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")  # Display confidence as a percentage
