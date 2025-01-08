import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import torch

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

start_detection = st.button("Start")
if start_detection:
    st.session_state.running = True

if st.session_state.running:
    cap = cv2.VideoCapture(0)  # Adjust '0' for external cameras

    if not cap.isOpened():
        st.error("Unable to access the camera.")
        st.session_state.running = False
    else:
        st.write("Press 'Stop' to end detection.")
        stop_detection = st.button("Stop")

        video_placeholder = st.empty()

        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab a frame.")
                st.session_state.running = False
                break

            # Resize for consistency
            frame_resized = cv2.resize(frame, (640, 480))

            # YOLO inference
            results = model(frame_resized, conf=0.5, device=device)

            # Annotate detections
            annotated_frame = results[0].plot()

            # Convert to RGB for Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display the frame
            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

            # Stop button action
            if stop_detection:
                st.session_state.running = False
                st.write("Detection stopped.")
                break

        cap.release()
