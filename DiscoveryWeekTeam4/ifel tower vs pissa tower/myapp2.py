import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from keras.models import load_model
import streamlit_webrtc as webrtc
import av

# Load the model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Custom CSS for Arctic theme
def apply_arctic_theme():
    st.markdown(
        """
        <style>
        body {
            background-image: url('https://your-url-to-background-image.jpg');
            background-size: cover;
            color: white;
        }
        .main {
            background: rgba(0, 0, 0, 0.5);  /* Dark overlay */
            padding: 20px;
            border-radius: 15px;
        }
        h1 {
            color: #cce7ff;  /* Light blue color */
        }
        .icon {
            width: 50px;
            height: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the arctic theme
apply_arctic_theme()

# Title and Arctic-themed header
st.markdown("<h1>Arctic Animal Identification</h1>", unsafe_allow_html=True)

# Arctic animal icon
st.image('C:\\Users\\vchuk\\Downloads\\Arctic\\Icons\\seal.png', width=100)

# Create the array for the AI model
def prepare_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# File upload button
uploaded_file = st.file_uploader("Upload an image of an Arctic animal...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Arctic Animal', use_column_width=True)

    # Process and predict
    data = prepare_image(image)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    st.write(f"Prediction: {class_name}")
    st.write(f"Confidence Score: {confidence_score:.2f}")

# WebRTC live stream camera feed
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    # Process frame and predict
    data = prepare_image(img_pil)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Display the prediction on the live video feed
    cv2.putText(img, f"{class_name} ({confidence_score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA)
    return av.VideoFrame.from_ndarray(img, format="rgb24")

# WebRTC Streamer
webrtc_ctx = webrtc.webrtc_streamer(
    key="arctic-animal-camera",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

# Arctic-themed footer
st.markdown("<footer><p style='text-align: center;'>Made with ❄️ in the Arctic</p></footer>", unsafe_allow_html=True)
