# Micro Expression Face Recognition
A real-time Micro Expression Recognition (MER) system developed using Python, OpenCV, and LibreFace for facial feature extraction and emotion analysis.
This project detects facial expressions from webcam/video input and performs emotion classification in real time using machine learning and computer vision techniques.

## Features
* Real-time facial expression detection
* Webcam-based emotion recognition
* Facial landmark and feature extraction using LibreFace
* CSV-based dataset support
* Real-time prediction and visualization
* Modular Python implementation
* Easy-to-run scripts for testing and inference

## Tech Stack
### Programming Language
* Python
### Libraries & Frameworks
* OpenCV
* NumPy
* Pandas
* LibreFace
* Scikit-learn
* TensorFlow / Keras (if used)

## Project Structure
Micro_Expression_Face_Recognition/
│
├── gallery/                     # Sample images and outputs
├── outputs/                     # Prediction results
├── tmp/                         # Temporary files
├── weights_libreface/           # LibreFace model weights
│
├── Final_MER.py                 # Main final execution script
├── MER_Webcam (with CSV).py     # Webcam prediction using CSV data
├── MER_Webcam (without CSV).py  # Webcam prediction without CSV
├── run_libreface.py             # LibreFace execution script
├── main.py                      # Main application entry
├── check.py                     # Utility/testing script
├── test.py                      # Model testing
├── try.py                       # Experimental script
└── README.md

## Installation
### 1. Clone the Repository
git clone https://github.com/your-username/Micro_Expression_Face_Recognition.git
cd Micro_Expression_Face_Recognition

### 2. Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
macOS/Linux
python3 -m venv venv
source venv/bin/activate

## Install Dependencies
pip install -r requirements.txt
If requirements.txt is not available:
pip install opencv-python numpy pandas scikit-learn tensorflow

## Running the Project
Run Main Application
python Final_MER.py
or
python main.py

## Webcam Emotion Detection
With CSV Dataset
python "MER_Webcam (with CSV).py"
Without CSV Dataset
python "MER_Webcam (without CSV).py"

## How It Works
1. Captures real-time video using webcam.
2. Detects faces using OpenCV.
3. Extracts facial landmarks and features using LibreFace.
4. Processes facial micro-expressions.
5. Uses machine learning model for emotion classification.
6. Displays detected emotion in real time.

## Supported Emotions
The model can classify emotions such as:
* Happy
* Sad
* Angry
* Surprise
* Fear
* Neutral
* Disgust
(Depends on the dataset and trained model.)

## Applications
* Human Emotion Analysis
* Mental Health Monitoring
* Smart Surveillance Systems
* Human-Computer Interaction
* Online Interview Analysis
* AI-Based Behavioral Research

## Future Improvements
* Deep learning-based micro-expression recognition
* Better real-time accuracy
* GUI integration using Tkinter or PyQt
* Cloud deployment support
* Multi-face emotion detection
* Explainable AI using Grad-CAM
* Multilingual support
* Mobile application integration

## Machine Learning Workflow
Input Webcam Feed
        ↓
Face Detection
        ↓
Feature Extraction (LibreFace)
        ↓
Preprocessing
        ↓
Emotion Classification
        ↓
Real-Time Output Display

## Requirements
* Python 3.8+
* Webcam
* Internet connection (for downloading dependencies)
* Minimum 4GB RAM recommended

## Author
Aditya Ingle
* Python Developer
* AI & Machine Learning Enthusiast
* Computer Vision Developer
