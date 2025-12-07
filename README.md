# Syntecxhub_project_Face-Detection-And-Recognition
Face Detection &amp; Recognition system using Python, OpenCV, Haarcascade, and LBPH. Supports dataset creation, model training, and real-time recognition. A complete SyntecxHub project for beginners in AI &amp; Computer Vision.
# Face Detection and Recognition using OpenCV & LBPH  
### SyntecxHub Project

This project implements a complete **Face Detection and Recognition system** using **OpenCV**, **Haarcascade classifiers**, and the **LBPH (Local Binary Patterns Histogram)** algorithm.  
It detects faces in real time from a webcam, trains a face recognizer on a dataset of images, and identifies known persons.

---

## 🧠 About the Project

This project is part of **SyntecxHub Projects**, designed to help beginners understand:

- How face detection works using **Haarcascade classifiers**
- How face recognition works using **LBPH algorithm**
- Dataset creation, model training, and recognition pipeline
- Real-time image processing using webcam frames

---

## ✨ Features

- ✔️ Real-time **face detection**  
- ✔️ Face **dataset creation**  
- ✔️ Model **training using LBPH**  
- ✔️ Real-time **face recognition**  
- ✔️ Automatic **labels.pickle** and **trainer.yml** generation  
- ✔️ Simple and easy-to-understand codebase  

---

## 📂 Project Structure

```bash
FaceDetectionRecognition/
├── README.md
├── create_dataset.py        # Create images for each person
├── train_model.py           # Train LBPH recognizer
├── recognize.py             # Run real-time recognition
├── haarcascade_frontalface_default.xml
├── dataset/                 # Folder for storing face images
│   ├── person1/
│   ├── person2/
│   └── ...
├── trainer/
│   ├── labels.pickle
│   └── trainer.yml
└── screenshots/
    └── demo.png

