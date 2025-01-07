import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import torch
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.state import SessionState

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n.pt")
model.to(device)

st.title("Object Detection with YOLOv8")

start_detection = st.button("Start")

if start_detection:
    cap = cv2.VideoCapture(0)  # 0 may need adjustment for external cameras

    if not cap.isOpened():
        st.error("Unable to access the camera.")
    else:
        st.write("Press 'Stop' to end detection.")
        stop_detection = st.button("Stop")

        video_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab a frame.")
                break

            frame_resized = cv2.resize(frame, (640, 480))

            results = model(frame_resized, conf=0.5, device=device)

            annotated_frame = results[0].plot()

            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

            if stop_detection:
                st.write("Detection stopped.")
                break

        cap.release()
