import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Function to preprocess the image
def preprocess_image(image):
    # Resize and crop the image
    image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
    # Convert to numpy array and normalize
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    return data

# Streamlit app
st.title("Dental Rugae Classification")
st.write("Upload a dental rugae image to classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    data = preprocess_image(image)
    
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the result
    st.write(f"**Predicted Class:** {class_name[2:]}")
    st.write(f"**Confidence Score:** {confidence_score}")
