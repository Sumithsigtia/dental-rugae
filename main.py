import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
model = tf.keras.models.load_model('keras_model.h5')

# Load class labels
with open("labels.txt", "r") as file:
    class_labels = [line.strip() for line in file.readlines()]

# Define conclusions for each class
class_conclusions = {
    "Average": "The average dental rugae pattern suggests a balanced face structure, typically characterized by well-proportioned facial features.",
    "Horizontal": "A horizontal dental rugae pattern indicates a broader face with horizontal prominence, often associated with wider cheekbones and a more pronounced jawline.",
    "Vertical": "A vertical dental rugae pattern corresponds to a longer face with vertical prominence, usually featuring a higher forehead and a longer chin."
}

def preprocess_image(image):
    """
    Preprocess the image for the model: resize, convert to RGB, normalize, and expand dimensions.
    """
    image = image.resize((224, 224))  # Resize to model input size
    image = image.convert('RGB')  # Ensure image is in RGB mode
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict(image):
    """
    Predict the class and confidence scores of the image using the loaded model.
    """
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]  # Get the first (and only) prediction
    return predictions

def display_predictions(predictions):
    """
    Display the confidence scores for all classes as progress bars and highlight the predicted class.
    """
    st.write("**Confidence Scores for All Classes:**")
    for class_label, score in zip(class_labels, predictions):
        percentage = int(score * 100)  # Convert to percentage
        st.progress(percentage, text=f"{class_label}: {score:.2%}")

    # Highlight the predicted class
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[predicted_class_index]
    st.write(f"\n**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2%}")

    # Display the conclusion for the predicted class
    conclusion = class_conclusions.get(predicted_class)
    st.write(f"**Conclusion:** {conclusion}")

# Streamlit app
st.title("Dental Rugae Classification Test App")

# File uploader for image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    predictions = predict(image)
    display_predictions(predictions)
