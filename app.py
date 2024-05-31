import streamlit as st
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('fruits2_classifier.h5')

# Class mapping
class_mapping = {
    0: 'apple_6', 1: 'apple_braeburn_1', 2: 'apple_crimson_snow_1', 3: 'apple_golden_1',
    4: 'apple_golden_2', 5: 'apple_golden_3', 6: 'apple_granny_smith_1', 7: 'apple_hit_1',
    8: 'apple_pink_lady_1', 9: 'apple_red_1', 10: 'apple_red_2', 11: 'apple_red_3',
    12: 'apple_red_delicios_1', 13: 'apple_red_yellow_1', 14: 'apple_rotten_1',
    15: 'cabbage_white_1', 16: 'carrot_1', 17: 'cucumber_1', 18: 'cucumber_3',
    19: 'eggplant_violet_1', 20: 'pear_1', 21: 'pear_3', 22: 'zucchini_1', 23: 'zucchini_dark_1'
}

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
    predicted_class_name = class_mapping[predicted_class_index]  # Map the index to the class name
    return predicted_class_name, prediction

# Main Streamlit app
def main():
    st.title("Image Classifier")

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
