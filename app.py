import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO model
model = YOLO("yolov8n.pt")
model.to(device)

# App title
st.title("Object Detection (using YOLOv8)")

# Define video processor class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.model.to(device)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize image
        img_resized = cv2.resize(img, (640, 480))

        # YOLO inference
        results = self.model(img_resized, conf=0.5, device=device)

        # Annotate detections
        annotated_frame = results[0].plot()

        return annotated_frame

# Start video stream with WebRTC
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
