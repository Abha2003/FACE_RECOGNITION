import cv2
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# Load the FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

data_path = r'D:\a\pics'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

embeddings = []
labels = []

for file in onlyfiles:
    image_path = join(data_path, file)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (160, 160))
    img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    img = (img - 127.5) / 128.0  # Normalize to [-1, 1]

    with torch.no_grad():
        embedding = model(img).numpy().flatten()
    embeddings.append(embedding)
    labels.append(file.split('_')[0])

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Train an SVM classifier
classifier = SVC(kernel='linear', probability=True)
classifier.fit(embeddings, labels)

# Save the classifier and label encoder
joblib.dump(classifier, 'face_classifier.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model training complete!")