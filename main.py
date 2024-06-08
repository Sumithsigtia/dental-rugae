import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model


# Function to load and preprocess image
def preprocess_image(img_path, target_size):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Function to make prediction
def predict_image(img, model, categories):
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = categories[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100
    return predicted_class, confidence

# Load the trained model
model = load_model('keras_model.h5')

# Define categories
categories = ['average', 'horizontal', 'vertical']

# Streamlit app
st.title('Dental Image Classifier')
st.write('Upload a dental image and let us classify it for you!')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = preprocess_image(uploaded_file, (224,224))

    # Make prediction
    predicted_class, confidence = predict_image(img, model, categories)

    # Display prediction
    st.write('Predicted Class:', predicted_class)
    st.write('Confidence:', f'{confidence:.2f}%')
