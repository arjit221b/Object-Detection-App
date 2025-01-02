import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use YOLOv8n for better speed and performance

# Streamlit UI
st.title("Object Detection (using YOLOv8)")
# st.write("This app uses your camera to detect objects in real-time. Allow camera permissions below.")

# Camera feed start button
start_detection = st.button("Start Camera")

# Stream video when the button is clicked
if start_detection:
    # Open the webcam (camera index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Unable to access the camera.")
    else:
        st.write("Press 'q' in the camera window to stop the detection.")
        
        # Create a placeholder for the video frames
        video_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab a frame.")
                break

            # Run YOLOv8 inference on the frame
            results = model(frame, conf=0.5)
            
            # Annotate the frame with detection results
            annotated_frame = results[0].plot()
            
            # Convert the frame to RGB (for Streamlit compatibility)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame in the placeholder
            video_placeholder.image(annotated_frame_rgb, channels="RGB")

            # Check for the 'q' key to stop the stream
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
