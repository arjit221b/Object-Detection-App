import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n.pt")  # Change this to yolov8n.pt or yolov8m.pt as per requirement
model.to(device)

st.title("Live Object Detection with YOLOv8")
st.write("This app uses your camera to detect objects in real-time. Allow camera permissions below.")

start_detection = st.button("Start Camera")

if start_detection:
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Unable to access the camera.")
    else:
        st.write("Press 'q' in the camera window to stop the detection.")

        video_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab a frame.")
                break

            frame_resized = cv2.resize(frame, (640, 480))

            results = model(frame_resized, conf=0.5, device=device)  # Explicitly send to correct device
            
            annotated_frame = results[0].plot()

            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
