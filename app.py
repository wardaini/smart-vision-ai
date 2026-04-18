import cv2
import streamlit as st
from ultralytics import YOLO
import time


st.title("Object Detection AI (camera)")

run = st.checkbox("Start camera")

FRAME_WINDOW = st.image([])

model = YOLO("yolov8s.pt", verbose=False)
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)
while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image")
        break

    results = model(frame, conf=0.6)
    annotated_frame = results[0].plot()
    FRAME_WINDOW.image(annotated_frame, channels="BGR")

cap.release()