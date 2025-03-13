import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels once when the app starts
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()


def predict_image(image: Image.Image):
    """Preprocess the image and predict using the loaded model."""
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:].strip(), confidence_score


# Create a VideoTransformer class to capture webcam frames
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img  # store the current frame for snapshot use
        return img


def main():
    st.title("Eiffel Tower vs Tower of Pisa Classifier")
    st.write("Upload an image or use your webcam to check if it's of the Eiffel Tower or the Tower of Pisa.")

    input_method = st.radio("Select input method:", ("Upload Image", "Webcam Capture"))

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Predict"):
                with st.spinner("Analyzing..."):
                    prediction, confidence = predict_image(image)
                st.success("Prediction complete!")
                st.write(f"**Prediction:** {prediction}")
                st.write(f"**Confidence Score:** {confidence:.2f}")

    elif input_method == "Webcam Capture":
        st.write("Your webcam feed will appear below. Click **Capture Snapshot** to take a picture.")
        ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
        if st.button("Capture Snapshot"):
            if ctx.video_transformer and ctx.video_transformer.frame is not None:
                # Convert the captured frame (BGR) to RGB, then to a PIL Image
                frame = cv2.cvtColor(ctx.video_transformer.frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                st.image(pil_image, caption="Captured Snapshot", use_column_width=True)
                with st.spinner("Analyzing..."):
                    prediction, confidence = predict_image(pil_image)
                st.success("Prediction complete!")
                st.write(f"**Prediction:** {prediction}")
                st.write(f"**Confidence Score:** {confidence:.2f}")
            else:
                st.warning("No frame available. Please ensure your webcam is active.")


if __name__ == "__main__":
    main()
