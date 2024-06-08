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
@tf.function(input_signature=[tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32)])
def predict(img_tensor):
    predictions = model(img_tensor, training=False)
    return predictions

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

    # Preprocess the image
    img_tensor = preprocess_image(image)

    # Make prediction
    predictions = predict(img_tensor)
    confidence = np.max(predictions)
    classes = ['Average', 'Horizontal', 'Vertical']
    predicted_class = classes[np.argmax(predictions)]

    # Display the result
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}") 
