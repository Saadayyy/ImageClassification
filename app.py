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
model = load_model('fruits_classifier.h5')

# Define class mapping
with open("C:\\Arti fici\\ML_NeuralNet\\Fruit-Classifier-app\\class_mapping.json") as f:
    class_mapping = json.load(f)

# Function to load and preprocess an image
def load_and_preprocess_image(img_array, target_size=(150, 150)):
    img = Image.fromarray(img_array)
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)  # Convert to float32
    img_array /= 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match the model's input shape
    return img_array

# Function to make predictions on a single image and return the class name
def predict_image(model, img_array):
    img_array = load_and_preprocess_image(img_array)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the index of the max value
    predicted_class_name = class_mapping[str(predicted_class_index)]  # Map the index to the class name
    return predicted_class_name, prediction

# Main Streamlit app
def main():
    st.title("Fruit Classifier App")

    # Option to upload image or capture from camera
    option = st.radio("Choose an option:", ("Upload Image", "Capture from Camera"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            img_array = np.array(img)
            predicted_class_name, prediction = predict_image(model, img_array)
            st.write("Predicted Class:", predicted_class_name)

    elif option == "Capture from Camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access camera.")
            return
        st.warning("Press 'Capture' to take a picture.")
        if st.button("Capture"):
            ret, frame = cap.read()
            if ret:
                cap.release()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels="RGB", use_column_width=True, caption="Captured Image")
                predicted_class_name, prediction = predict_image(model, frame_rgb)
                st.write("Predicted Class:", predicted_class_name)

if __name__ == "__main__":
    main()
