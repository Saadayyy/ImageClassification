# ImageClassification
This repository contains a Streamlit web application that allows users to classify images of fruits using a pre-trained convolutional neural network (CNN) model. The app supports uploading photos from the user's device or capturing images using their laptop or phone camera.

## Features

- Upload images for classification
- Capture images using a laptop or phone camera
- View the predicted class and probability score

## Model Information

The model used in this app is a CNN trained on a dataset of various fruit images. The model can classify different types of fruits.

## Installation

1. **Clone the Repository**

    ```sh
    git clone https://github.com/Saadayyy/ImageClassification.git
    cd ImageClassification/Fruit-Classifier-app
    ```

2. **Set Up the Virtual Environment**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```sh
    pip install -r requirements.txt
    ```

4. **Run the App**

    ```sh
    streamlit run app.py
    ```

## Usage

1. Open your browser and go to `http://localhost:8501`.
2. Use the "Browse files" button to upload an image or click the "Capture photo" button to take a picture using your camera.
3. The app will display the uploaded image and show the predicted class along with the probability score.

## Deployment

To deploy this app, follow these steps:

1. **Initialize Git in the Project Directory**

    ```sh
    git init
    git add .
    git commit -m "Initial commit"
    ```

2. **Add Remote Repository**

    ```sh
    git remote add origin https://github.com/Saadayyy/ImageClassification.git
    ```

3. **Pull Remote Changes and Resolve Any Conflicts**

    ```sh
    git pull origin main --allow-unrelated-histories
    ```

4. **Push Local Changes to GitHub**

    ```sh
    git push -u origin main
    ```

## Contributing

Feel free to open issues or submit pull requests if you have any improvements or bug fixes.

## License

This project is licensed under the MIT License.

