import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and class names once when the app starts
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()


def predict_image(image: Image.Image):
    """
    Preprocess the image, run prediction, and return the class name and confidence.
    """
    # Ensure image is in RGB mode and resize/crop it to 224x224
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert image to numpy array and normalize
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create array of shape (1, 224, 224, 3) for prediction
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction and process results
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:].strip(), confidence_score


def main():
    st.title("Eiffel Tower vs Tower of Pisa Classifier")
    st.write(
        "Upload an image below and click 'Predict' to check if it's an image of the Eiffel Tower or the Tower of Pisa.")

    # File uploader widget for image files
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # When the user clicks the Predict button, process the image
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                prediction, confidence = predict_image(image)
            st.success("Prediction complete!")
            st.write(f"**Prediction:** {prediction}")
            st.write(f"**Confidence Score:** {confidence:.2f}")


if __name__ == '__main__':
    main()
