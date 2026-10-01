# SignBridge AI - Hand Gestures to Sign Language Communication System

A real-time Computer Vision and Machine Learning application that recognizes hand gestures through a webcam, translates them into text, forms continuous sentences, and converts them into spoken voice (Text-to-Speech).

---

## 🌟 Key Features

1. **Real-Time Landmark Tracking**:
   * Uses **Google MediaPipe Hands** to extract 21 high-precision 3D hand landmarks at 60+ FPS on Apple Silicon.
2. **Invariant Feature Normalization**:
   * Preprocessing removes translation and scale variance. Hand position in frame and distance from camera do not degrade classification.
3. **Smart Sentence Builder**:
   * Includes temporal hold and debounce logic: signs are appended only when held stably above 75% confidence for ~0.8 seconds.
4. **Asynchronous Text-to-Speech**:
   * Native Apple speech synthesis on macOS with non-blocking audio threading.
5. **Modern Interactive Web Frontend**:
   * Glassmorphism dark-mode UI accessible at `http://localhost:5000`.
   * Live camera monitor with landmark skeleton toggle.
   * Dynamic confidence gauge and hold-to-type progress meter.
   * Built-in **Data Studio** to record custom gestures directly with a 3-second countdown.
   * 1-Click **Model Retraining & Hot-Reloading** without restarting the server.
6. **Dual Running Modes**:
   * Web App dashboard (`python app.py`)
   * Standalone desktop OpenCV HUD (`python recognize.py`)

---

## 📁 Project Structure

```
Sign language/
├── .venv/                      # Python virtual environment
├── requirements.txt            # Dependencies (OpenCV, MediaPipe, Scikit-learn, Flask)
├── dataset/
│   ├── raw_landmarks.npz       # Coordinate landmarks & labels
│   └── gesture_labels.json     # Class mappings
├── models/
│   ├── gesture_model.pkl       # Trained Random Forest classifier
│   └── model_metadata.json     # Accuracy and evaluation metrics
├── preprocess.py               # Scale & translation invariant normalization engine
├── text_to_speech.py           # Thread-safe asynchronous speech engine
├── sample_dataset_generator.py # Generates calibrated baseline poses (A-Z, Hello, Yes, etc.)
├── train_model.py              # Model trainer with train/test metrics & evaluation
├── collect_data.py             # CLI/GUI interactive data collector with visual feedback
├── recognize.py                # Standalone real-time OpenCV desktop recognizer
├── app.py                      # Full Flask Web Application & Streaming Server
├── templates/
│   └── index.html              # Sleek Web Dashboard UI
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphism dark theme styles
│   └── js/
│       └── main.js             # Real-time state polling & audio controls
└── README.md                   # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Activate the Virtual Environment
```bash
source .venv/bin/activate
```

### 2. Launch the Web Application (Recommended)
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://localhost:5000`**

From the browser you can:
* Watch the live vision feed with hand skeleton overlay.
* See detected signs and real-time confidence scores.
* Form sentences and click **Speak Sentence** or **Copy**.
* Switch to the **Data Studio** tab to record new custom signs.
* Switch to the **Model & Analytics** tab to see accuracy or retrain.

---

### 3. Alternative: Run the Desktop OpenCV HUD
If you prefer a direct OpenCV desktop window:
```bash
python recognize.py
```

**Desktop Keyboard Controls**:
* `[ENTER]` : Speak the accumulated sentence out loud
* `[SPACE]` : Insert a space
* `[BACKSPACE]` : Delete the last signed word/letter
* `[C]` : Clear the entire sentence
* `[Q]` or `[ESC]` : Quit

---

## 📸 Collecting Custom Gestures

### Option A: From the Web UI
1. Navigate to the **Data Studio (Record)** tab at `http://localhost:5000`.
2. Type your gesture name (e.g. `Water`, `Coffee`, `Z`).
3. Click **Start Recording Gesture**.
4. A 3-second countdown will start—position your hand in view.
5. Once captured, click **Retrain & Hot-Reload Model** on the Model tab!

### Option B: From the Terminal
```bash
python collect_data.py --gesture "Hello" --samples 100
```
* Press `[SPACE]` to start the 3-second countdown.
* Hold your hand in front of the camera while the progress bar fills up.
* Press `[S]` to save into the dataset.

---

## 🧠 Model Training

To retrain the model from terminal:
```bash
python train_model.py --model rf
```
You can also train a neural network (Multi-Layer Perceptron):
```bash
python train_model.py --model mlp
```

---

## ⚙️ macOS Camera Permissions
If the webcam feed displays a connection notice:
1. Open **System Settings** on your Mac.
2. Go to **Privacy & Security** $\rightarrow$ **Camera**.
3. Ensure your Terminal or IDE application is granted Camera access.
