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
option = st.sidebar.selectbox("Select an option", ("Upload Image", "Capture from Camera"))

if option == "Upload Image":
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        predictions = predict(image)

        # Display all class confidence scores
        st.write("**Confidence Scores for All Classes:**")
        for class_label, score in zip(class_labels, predictions):
            st.write(f"{class_label}: {score:.2%}")

        # Highlight the predicted class
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]
        confidence = predictions[predicted_class_index]
        st.write(f"\n**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2%}")

elif option == "Capture from Camera":
    st.write("Camera Capture Mode")

    # Capture image from camera
    cap = cv2.VideoCapture(0)  # 0 indicates the default camera

    if not cap.isOpened():
        st.write("Error: Could not open camera.")
        st.stop()

    capture_button_key = "capture_button"
    stop_button_key = "stop_camera_button"

    placeholder = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Error: Could not read frame from camera.")
            break

        # Display the captured frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(frame_rgb, channels="RGB", use_column_width=True, caption="Camera Capture")

        # Check for user input to capture image or stop camera
        if st.button("Capture Image", key=capture_button_key):
            # Save the captured frame to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                cv2.imwrite(tmpfile.name, frame)
                image = Image.open(tmpfile.name)

            # Display the captured image
            placeholder.image(image, caption="Captured Image", use_column_width=True)

            predictions = predict(image)

            # Display all class confidence scores
            st.write("**Confidence Scores for All Classes:**")
            for class_label, score in zip(class_labels, predictions):
                st.write(f"{class_label}: {score:.2%}")

            # Highlight the predicted class
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_index]
            confidence = predictions[predicted_class_index]
            st.write(f"\n**Predicted Class:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2%}")

            # Option to continue or stop camera
            if st.button("Stop Camera", key=stop_button_key):
                break

        # Option to stop camera without capturing
        if st.button("Stop Camera", key=f"{stop_button_key}_alt"):
            break

    # Release the camera
    cap.release()
    placeholder.empty()
