from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

app = Flask(__name__)
socketio = SocketIO(app)

# Load pre-trained face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the saved models
emotion_model_path = 'modelmustafa.h5'
age_model_path = 'saved_model.h5'
if os.path.exists(emotion_model_path) and os.path.exists(age_model_path):
    emotion_classifier = load_model(emotion_model_path)
    age_classifier = load_model(age_model_path)
else:
    raise FileNotFoundError("One or both model files not found. Please check the paths and try again.")

# Labels for emotion and age prediction
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
age_labels = ['YOUNG', 'MIDDLE', 'OLD']

def preprocess_image(image, target_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    roi = gray_resized.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    roi = np.repeat(roi, 3, axis=-1)
    return roi

def capture_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video device")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi_emotion = preprocess_image(roi_gray, (48, 48))
                roi_age = preprocess_image(roi_gray, (64, 64))

                # Predict emotion
                emotion_prediction = emotion_classifier.predict(roi_emotion)[0]
                emotion_label = emotion_labels[emotion_prediction.argmax()]

                # Predict age
                age_prediction = age_classifier.predict(roi_age)[0]
                age_label = age_labels[age_prediction.argmax()]

                label = f"{emotion_label}, {age_label}"
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
