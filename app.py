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

# Persistent state for detection
if "running" not in st.session_state:
    st.session_state.running = False

# Create a button to start detection
start_detection = st.button("Start")
if start_detection:
    st.session_state.running = True

# Create a class to process the video frames from the webcam
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to BGR (OpenCV format)
        
        if st.session_state.running:
            # Resize frame for YOLO inference
            frame_resized = cv2.resize(img, (640, 480))

            # Perform YOLO inference
            results = model(frame_resized, conf=0.5, device=device)

            # Annotate detections on the frame
            annotated_frame = results[0].plot()

            # Convert to RGB for Streamlit display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            return annotated_frame_rgb
        return img  # return the original frame if not running detection

# WebRTC streamer to capture webcam feed and process it using VideoProcessor
webrtc_streamer(key="yolo_object_detection", video_processor_factory=VideoProcessor)
