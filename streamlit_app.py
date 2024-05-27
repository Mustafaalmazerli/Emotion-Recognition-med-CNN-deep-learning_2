import os
import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Streamlit configuration
st.set_page_config(layout="wide")

# Load pre-trained face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the saved models for age and emotion detection
emotion_model_path = 'modelmustafa.h5'
age_proto_path = 'age_deploy.prototxt'
age_net_path = 'age_net.caffemodel'

# Load age detection model
age_net = cv2.dnn.readNetFromCaffe(age_proto_path, age_net_path)

# Load emotion detection model
if os.path.exists(emotion_model_path):
    emotion_classifier = load_model(emotion_model_path)
    st.write("Emotion detection model loaded successfully")
else:
    st.error("Emotion model file not found. Please check the path and try again.")
    st.stop()

# Age and emotion labels
AGE_RANGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Could not open video device")
    st.stop()

st.title("Age and Emotion Detector")

# Streamlit placeholder for video feed
frame_placeholder = st.empty()

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Age detection
        age_blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(age_blob)
        age_predictions = age_net.forward()
        age_index = age_predictions[0].argmax()
        age_label = AGE_RANGES[age_index]
        age_label_position = (x, y - 10)
        cv2.putText(frame, age_label, age_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Emotion detection
        emotion_roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([emotion_roi_gray]) != 0:
            emotion_roi = emotion_roi_gray.astype('float') / 255.0
            emotion_roi = img_to_array(emotion_roi)
            emotion_roi = np.expand_dims(emotion_roi, axis=0)

            # Predict the class for the ROI
            emotion_prediction = emotion_classifier.predict(emotion_roi)[0]
            emotion_label = emotion_labels[emotion_prediction.argmax()]
            emotion_label_position = (x, y + h + 30)
            cv2.putText(frame, emotion_label, emotion_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
