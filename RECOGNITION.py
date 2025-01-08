import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from os import listdir
from os.path import isfile, join
import joblib
import time

# Load the FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load the face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_classifier.empty():
    print("Error loading Haar Cascade XML file.")

# Load the trained classifier and label encoder
classifier = joblib.load('face_classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Apply histogram equalization
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    print(f"Number of faces detected: {len(faces)}")

    if len(faces) == 0:
        return img, None, None, None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (160, 160))
    return img, roi, x, y

cap = cv2.VideoCapture(0)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    image, face, x, y = face_detector(frame)

    try:
        if face is not None:
            face = np.transpose(face, (2, 0, 1))  # Convert to (C, H, W)
            face = torch.tensor(face, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            face = (face - 127.5) / 128.0  # Normalize to [-1, 1]

            with torch.no_grad():
                embedding = model(face).numpy().flatten()

            prediction = classifier.predict([embedding])
            confidence = classifier.predict_proba([embedding]).max()
            name = label_encoder.inverse_transform(prediction)[0]

            if confidence > 0.5:  # Adjust confidence threshold as needed
                cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                print(f"Recognized as: {name} with confidence {confidence:.2f}")
            else:
                cv2.putText(image, f"{name} (Low confidence)", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                print(f"Face recognized but confidence is low. Recognized as: {name} with confidence {confidence:.2f}")
        else:
            cv2.putText(image, "Face Not Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            print("No face detected.")
    except Exception as e:
        cv2.putText(image, "Face Not Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        print(f"Error: {e}")

    cv2.imshow('Face Cropper', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

    # Stop the loop after 5 seconds
    if time.time() - start_time > 5:
        print("Stopping the loop after 5 seconds.")
        break

cap.release()
cv2.destroyAllWindows()