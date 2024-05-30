import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Load the model
model = load_model('fruits2_classifier.h5')

# Define class mapping
with open("C:\\Arti fici\\ML_NeuralNet\\Fruit-Classifier-app\\class_mapping.json") as f:
    class_mapping = json.load(f)

# Function to preprocess image
def load_and_preprocess_image(img_array, target_size=(150, 150)):
    img = Image.fromarray(img_array)
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)  # Convert to float32
    img_array /= 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match the model's input shape
    return img_array

# Function to predict image
def predict_image(model, img_array):
    img_array = load_and_preprocess_image(img_array)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the index of the max value
    predicted_class_name = class_mapping[predicted_class_index]  # Map the index to the class name
    return predicted_class_name, prediction

def main():
    st.title("Fruit Classifier")

    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose input method", ("Upload an Image", "Use Camera"))

    if option == "Upload an Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            img_array = np.array(img)
            predicted_class_name, prediction = predict_image(model, img_array)
            st.write(f"Predicted Class: {predicted_class_name}")

    elif option == "Use Camera":
        st.write("Using your camera to capture an image.")
        camera_capture = st.camera_input("Take a picture")
        if camera_capture is not None:
            img = Image.open(camera_capture)
            st.image(img, caption='Captured Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            img_array = np.array(img)
            predicted_class_name, prediction = predict_image(model, img_array)
            st.write(f"Predicted Class: {predicted_class_name}")

if __name__ == "__main__":
    main()
