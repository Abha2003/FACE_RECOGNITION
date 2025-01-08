import cv2
import numpy as np
from os import listdir
from os.path import isfile, join, isdir

# Load Haar Cascade Classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face

def augment_image(image):
    augmented_images = [image]
    # Rotate the image
    for angle in [15, -15]:
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        augmented_images.append(rotated)
    # Flip the image
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    return augmented_images

# Directory containing the images
data_path = r'D:\a\pics'
subdirs = [d for d in listdir(data_path) if isdir(join(data_path, d))]

for subdir in subdirs:
    person_path = join(data_path, subdir)
    onlyfiles = [f for f in listdir(person_path) if isfile(join(person_path, f))]
    
    for file in onlyfiles:
        image_path = join(person_path, file)
        frame = cv2.imread(image_path)
        if face_extractor(frame) is not None:
            face = face_extractor(frame)
            face = cv2.resize(face, (200, 200))
            augmented_faces = augment_image(face)
            for i, aug_face in enumerate(augmented_faces):
                file_name_path = f'D:\\a\\pics\\{subdir}_{file.split(".")[0]}_aug_{i}.jpeg'
                cv2.imwrite(file_name_path, aug_face)
                cv2.putText(aug_face, f'{subdir}_{i}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', aug_face)
                cv2.waitKey(500)  # Display each image for 500ms
        else:
            print("Face not found in image:", file)

cv2.destroyAllWindows()
print('Samples Collection Completed')