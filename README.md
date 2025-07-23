# Face Recognition Attendance System

This is a desktop-based **Face Recognition Attendance System** project built using Python. It uses OpenCV and face_recognition libraries to detect and recognise faces and mark attendance automatically.

## 🚀 **Features**
- Face detection and recognition using webcam
- Attendance marking with timestamp
- Uses trained encodings for faster recognition

## 🛠️ **Technologies Used**
- Python
- OpenCV
- face_recognition
- pickle (for saving encodings)

## 📂 **Project Structure**
- **images/** – Contains student images for training
- **trained_model/** – Stores encodings.pickle file
- **attendance/** – Stores attendance records (to be implemented)
- **data_collection.py** – For data/image collection (future)
- **train_model.py** – Encodes faces from images folder
- **main.py** – Runs webcam recognition and marks attendance
- **README.md** – Project description

## 💻 **How To Run**
1. Clone the repo:
   ```bash
   git clone https://github.com/vyshnavi-ctrl/face-recognition-attendance.git
