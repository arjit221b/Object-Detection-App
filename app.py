import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import base64
import io
from streamlit.components.v1 import html

# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")
model.to(device)

# App title
st.title("Object Detection (using YOLOv8)")

# JavaScript for webcam capture
html_code = """
<script>
    let video;
    let canvas;
    let ctx;
    let videoStream;
    let videoWidth = 640;
    let videoHeight = 480;

    async function startVideo() {
        video = document.createElement('video');
        canvas = document.createElement('canvas');
        ctx = canvas.getContext('2d');
        
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: videoWidth, height: videoHeight }
        });

        video.srcObject = videoStream;
        video.play();
        
        // Set up canvas size
        canvas.width = videoWidth;
        canvas.height = videoHeight;

        // Start sending video frames every 100 ms
        setInterval(() => {
            ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
            let dataUrl = canvas.toDataURL('image/png'); 
            window.parent.postMessage(dataUrl, "*");
        }, 100);
    }

    startVideo();
</script>
"""

# Display the webcam stream using custom HTML
html(html_code, height=500)

# Create a placeholder for displaying frames
video_placeholder = st.empty()

# Capture and process frames in Python
if st.button("Start Detection"):
    # Get the base64 image data from JavaScript
    data_url = st.text_input("Paste Image Data URL", "")

    if data_url:
        # Convert base64 data to OpenCV image
        img_data = base64.b64decode(data_url.split(',')[1])
        np_img = np.asarray(bytearray(img_data), dtype=np.uint8)
        frame = cv2.imdecode(np_img, 1)

        # YOLO inference on the frame
        results = model(frame, conf=0.5, device=device)

        # Annotate detections
        annotated_frame = results[0].plot()

        # Display the annotated frame
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
