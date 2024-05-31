import streamlit as st
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('fruits_classifier2.h5')

class_mapping = {
    "0": "Apple Braeburn", "1": "Apple Crimson Snow", "2": "Apple Golden 1",
    "3": "Apple Golden 2", "4": "Apple Golden 3", "5": "Apple Granny Smith",
    "6": "Apple Pink Lady", "7": "Apple Red 1", "8": "Apple Red 2",
    "9": "Apple Red 3", "10": "Apple Red Delicious", "11": "Apple Red Yellow 1",
    "12": "Apple Red Yellow 2", "13": "Apricot", "14": "Avocado",
    "15": "Avocado ripe", "16": "Banana", "17": "Banana Lady Finger",
    "18": "Banana Red", "19": "Beetroot", "20": "Blueberry",
    "21": "Cactus fruit", "22": "Cantaloupe 1", "23": "Cantaloupe 2",
    "24": "Carambula", "25": "Cauliflower", "26": "Cherry 1",
    "27": "Cherry 2", "28": "Cherry Rainier", "29": "Cherry Wax Black",
    "30": "Cherry Wax Red", "31": "Cherry Wax Yellow", "32": "Chestnut",
    "33": "Clementine", "34": "Cocos", "35": "Corn", "36": "Corn Husk",
    "37": "Cucumber Ripe", "38": "Cucumber Ripe 2", "39": "Dates",
    "40": "Eggplant", "41": "Fig", "42": "Ginger Root", "43": "Granadilla",
    "44": "Grape Blue", "45": "Grape Pink", "46": "Grape White",
    "47": "Grape White 2", "48": "Grape White 3", "49": "Grape White 4",
    "50": "Grapefruit Pink", "51": "Grapefruit White", "52": "Guava",
    "53": "Hazelnut", "54": "Huckleberry", "55": "Kaki", "56": "Kiwi",
    "57": "Kohlrabi", "58": "Kumquats", "59": "Lemon", "60": "Lemon Meyer",
    "61": "Limes", "62": "Lychee", "63": "Mandarine", "64": "Mango",
    "65": "Mango Red", "66": "Mangostan", "67": "Maracuja",
    "68": "Melon Piel de Sapo", "69": "Mulberry", "70": "Nectarine",
    "71": "Nectarine Flat", "72": "Nut Forest", "73": "Nut Pecan",
    "74": "Onion Red", "75": "Onion Red Peeled", "76": "Onion White",
    "77": "Orange", "78": "Papaya", "79": "Passion Fruit", "80": "Peach",
    "81": "Peach 2", "82": "Peach Flat", "83": "Pear", "84": "Pear 2",
    "85": "Pear Abate", "86": "Pear Forelle", "87": "Pear Kaiser",
    "88": "Pear Monster", "89": "Pear Red", "90": "Pear Stone",
    "91": "Pear Williams", "92": "Pepino", "93": "Pepper Green",
    "94": "Pepper Orange", "95": "Pepper Red", "96": "Pepper Yellow",
    "97": "Physalis", "98": "Physalis with Husk", "99": "Pineapple",
    "100": "Pineapple Mini", "101": "Pitahaya Red", "102": "Plum",
    "103": "Plum 2", "104": "Plum 3", "105": "Pomegranate",
    "106": "Pomelo Sweetie", "107": "Potato Red", "108": "Potato Red Washed",
    "109": "Potato Sweet", "110": "Potato White", "111": "Quince",
    "112": "Rambutan", "113": "Raspberry", "114": "Redcurrant",
    "115": "Salak", "116": "Strawberry", "117": "Strawberry Wedge",
    "118": "Tamarillo", "119": "Tangelo", "120": "Tomato 1",
    "121": "Tomato 2", "122": "Tomato 3", "123": "Tomato 4",
    "124": "Tomato Cherry Red", "125": "Tomato Heart", "126": "Tomato Maroon",
    "127": "Tomato Yellow", "128": "Tomato not Ripened", "129": "Walnut",
    "130": "Watermelon"
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
    predicted_class_name = class_mapping[str(predicted_class_index)]  # Convert index to string and map to class name
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
