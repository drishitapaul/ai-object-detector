import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("🧠 YOLOv8 Object Detection App")
st.markdown("Upload an image to detect objects using the YOLOv8 model.")

# Load YOLOv8 model (nano version for speed)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # 'n' = nano (fastest), use 's'/'m' for more accuracy

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    with st.spinner("Detecting..."):
        results = model(temp_path)[0]
        result_image = results.plot()
        result_image_rgb = Image.fromarray(result_image[..., ::-1])  # Convert BGR to RGB

    st.image(result_image_rgb, caption="Detected Image", use_column_width=True)

    os.remove(temp_path)
