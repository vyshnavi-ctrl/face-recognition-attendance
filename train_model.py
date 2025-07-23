import os
import cv2
import face_recognition
import pickle

# Path to the images folder
images_path = 'images'

# Lists to store encodings and names
known_encodings = []
known_names = []

# Loop through each image in the folder
for filename in os.listdir(images_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Extract student name from filename (remove extension)
        name = os.path.splitext(filename)[0]

        # Load the image
        img_path = os.path.join(images_path, filename)
        image = cv2.imread(img_path)

        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations
        boxes = face_recognition.face_locations(rgb, model='hog')  # or 'cnn' if GPU supported

        # Encode the face(s) found
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

print("[INFO] Training complete. Saving encodings...")

# Save encodings and names to a pickle file
data = {"encodings": known_encodings, "names": known_names}

# Create trained_model folder if it doesn't exist
if not os.path.exists("trained_model"):
    os.makedirs("trained_model")

with open("trained_model/encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("[INFO] Encodings saved to trained_model/encodings.pickle")
