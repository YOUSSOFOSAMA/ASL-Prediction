# 🤟 AI-Powered ASL Predictor

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-green?style=for-the-badge&logo=opencv)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange?style=for-the-badge&logo=scikit-learn)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-blue?style=for-the-badge)

A real-time American Sign Language (ASL) alphabet predictor built with Python. This project utilizes Google's **MediaPipe** for advanced hand-landmark extraction and a **Support Vector Machine (SVM)** classifier trained via **Scikit-Learn** to accurately translate hand poses into text. 

The application features a sleek, dark-mode web interface built with **Streamlit** that allows users to capture poses, predict letters, and build entire words in real-time.

---

## 🌟 Live Demo
*https://asl-prediction-rmb9yscdovv9bhe9cxruv5.streamlit.app/#camera-input*

---

## 🚀 Features
- **Real-Time Landmark Detection**: Uses MediaPipe to extract 63 complex 3D hand coordinates.
- **Custom Trained ML Model**: Uses a custom-trained SVC model with `StandardScaler` for highly accurate letter prediction.
- **Interactive Word Builder**: An interactive GUI allowing users to string together letters into words, featuring Undo and Clear functionalities.
- **Cloud-Ready GUI**: Fully responsive and deployable via Streamlit Community Cloud.

## 📁 Repository Structure
- `app.py`: The main Streamlit web application.
- `Predictionpy.py`: Original OpenCV desktop script for real-time webcam inference.
- `Classification.py` / `mediapipeee.py`: The scripts used to extract landmarks from the dataset and train the Machine Learning model.
- `asl_classifier.pkl` / `scaler.pkl`: The serialized, pre-trained Scikit-Learn models.
- `requirements.txt`: Python dependencies required for cloud deployment.

## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/asl-predictor.git
   cd asl-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   python -m streamlit run app.py
   ```
   *The app will automatically open in your default web browser at `http://localhost:8501`.*

## 🧠 Model Training Details
The model was trained on a comprehensive dataset of ASL alphabet images. Hand landmarks were extracted, normalized (translation and scale invariant based on the wrist and bounding box), and flattened into a 63-feature array before being scaled and passed into the SVM classifier.
