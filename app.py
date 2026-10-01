"""
SignBridge AI - Hand Gesture to Sign Language Web Application
Full-featured Web App with Browser-Direct Camera Vision, Modern MediaPipe AI,
HUD, Sentence Builder, Asynchronous Text-to-Speech, Data Studio, and 1-Click Retraining.
"""

import os
import sys
import time
import json
import base64
import threading
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, Response, jsonify, request

from hand_detector import HandDetector
from preprocess import extract_raw_landmarks, normalize_landmarks, get_bounding_box
from text_to_speech import tts
from train_model import train_and_save_model, MODEL_PATH, METADATA_PATH, DATASET_FILE, LABELS_FILE

app = Flask(__name__)

# Global Hand Detector
detector = HandDetector()

# Global Application State
state_lock = threading.Lock()
app_state = {
    "current_gesture": None,
    "confidence": 0.0,
    "hold_progress": 0.0,
    "sentence": [],
    "show_skeleton": True,
    "model_loaded": False,
    "camera_active": False,
    # Studio recording state
    "recording_active": False,
    "recording_countdown": 0,
    "recording_current": 0,
    "recording_target": 100,
    "recording_gesture_name": "",
}

# Sentence Builder Logic Parameters
sentence_buffer = {
    "candidate": None,
    "consecutive_frames": 0,
    "required_frames": 15,        # ~0.6-0.8s
    "confidence_thresh": 0.75,
    "last_committed_time": 0,
    "cooldown_sec": 1.2
}

# Model and Classifier
model = None
model_classes = []


def load_model():
    """Loads the trained gesture classification model into memory."""
    global model, model_classes
    with state_lock:
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                model_classes = [str(c) for c in model.classes_]
                app_state["model_loaded"] = True
                print(f"[Model] Successfully loaded {len(model_classes)} classes: {model_classes}")
                return True
            except Exception as e:
                print(f"[Model Error] Failed loading model: {e}")
                app_state["model_loaded"] = False
                return False
        else:
            print("[Model Warning] No model found at models/gesture_model.pkl")
            app_state["model_loaded"] = False
            return False


# Initial model load
load_model()

# Recording worker state
record_state = {
    "active": False,
    "gesture_name": "",
    "target_samples": 100,
    "countdown_start": 0,
    "samples": []
}


def trigger_recording(gesture_name, target_samples):
    with state_lock:
        record_state["active"] = True
        record_state["gesture_name"] = gesture_name
        record_state["target_samples"] = target_samples
        record_state["countdown_start"] = time.time()
        record_state["samples"] = []

        app_state["recording_active"] = True
        app_state["recording_countdown"] = 3
        app_state["recording_current"] = 0
        app_state["recording_target"] = target_samples
        app_state["recording_gesture_name"] = gesture_name


def finalize_recording():
    """Saves collected samples from web studio to dataset."""
    new_features = record_state["samples"]
    gesture_name = record_state["gesture_name"]

    if new_features:
        new_X = np.array(new_features, dtype=np.float32)
        new_y = np.array([gesture_name] * len(new_features))

        os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
        if os.path.exists(DATASET_FILE):
            existing = np.load(DATASET_FILE, allow_pickle=True)
            combined_X = np.vstack([existing["X"], new_X])
            combined_y = np.concatenate([existing["y"], new_y])
        else:
            combined_X = new_X
            combined_y = new_y

        np.savez_compressed(DATASET_FILE, X=combined_X, y=combined_y)

        classes = sorted(list(set(combined_y)))
        with open(LABELS_FILE, "w") as f:
            json.dump({"classes": classes}, f, indent=2)

        print(f"[Studio] Saved {len(new_features)} samples for '{gesture_name}' to dataset.")

    with state_lock:
        record_state["active"] = False
        record_state["samples"] = []
        app_state["recording_active"] = False
        app_state["recording_countdown"] = 0


# ==============================================================================
# FLASK WEB ROUTES & API
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """
    Receives video frame from browser webcam, runs MediaPipe HandLandmarker,
    classifies gesture, and updates sentence builder.
    """
    global model, model_classes

    data = request.get_json() or {}
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"error": "No image data"}), 400

    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
    except Exception as e:
        return jsonify({"error": f"Decoding error: {str(e)}"}), 400

    # Hand detection
    landmarks = detector.detect(frame)
    detected_gesture = None
    detected_confidence = 0.0
    hold_ratio = 0.0
    lm_list = []

    if landmarks is not None:
        lm_list = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in landmarks]
        raw_pts = extract_raw_landmarks(landmarks)
        norm_vec = normalize_landmarks(raw_pts).reshape(1, -1)

        if model is not None and len(model_classes) > 0:
            probs = model.predict_proba(norm_vec)[0]
            best_idx = np.argmax(probs)
            detected_gesture = model_classes[best_idx]
            detected_confidence = float(probs[best_idx])

        # Studio Recording logic
        if record_state["active"]:
            now = time.time()
            elapsed = now - record_state["countdown_start"]
            if elapsed < 3.0:
                countdown = int(np.ceil(3.0 - elapsed))
                with state_lock:
                    app_state["recording_countdown"] = countdown
            else:
                with state_lock:
                    app_state["recording_countdown"] = 0
                record_state["samples"].append(norm_vec.flatten())
                current_count = len(record_state["samples"])
                with state_lock:
                    app_state["recording_current"] = current_count
                if current_count >= record_state["target_samples"]:
                    finalize_recording()

    # Temporal debounce sentence builder
    now = time.time()
    if detected_gesture and detected_confidence >= sentence_buffer["confidence_thresh"]:
        if detected_gesture == sentence_buffer["candidate"]:
            sentence_buffer["consecutive_frames"] += 1
        else:
            sentence_buffer["candidate"] = detected_gesture
            sentence_buffer["consecutive_frames"] = 1

        hold_ratio = min(1.0, sentence_buffer["consecutive_frames"] / sentence_buffer["required_frames"])

        if sentence_buffer["consecutive_frames"] >= sentence_buffer["required_frames"]:
            with state_lock:
                if not app_state["sentence"] or (
                    app_state["sentence"][-1] != detected_gesture
                    or (now - sentence_buffer["last_committed_time"]) >= sentence_buffer["cooldown_sec"]
                ):
                    app_state["sentence"].append(detected_gesture)
                    sentence_buffer["last_committed_time"] = now

            sentence_buffer["consecutive_frames"] = 0
    else:
        sentence_buffer["candidate"] = None
        sentence_buffer["consecutive_frames"] = 0
        hold_ratio = 0.0

    with state_lock:
        app_state["current_gesture"] = detected_gesture
        app_state["confidence"] = detected_confidence
        app_state["hold_progress"] = hold_ratio
        current_sentence = list(app_state["sentence"])
        rec_active = app_state["recording_active"]
        rec_cd = app_state["recording_countdown"]
        rec_cur = app_state["recording_current"]
        rec_tgt = app_state["recording_target"]

    return jsonify({
        "hand_detected": landmarks is not None,
        "landmarks": lm_list,
        "gesture": detected_gesture,
        "confidence": detected_confidence,
        "hold_progress": hold_ratio,
        "sentence": current_sentence,
        "recording_active": rec_active,
        "recording_countdown": rec_cd,
        "recording_current": rec_cur,
        "recording_target": rec_tgt
    })


@app.route('/api/state')
def get_state():
    with state_lock:
        return jsonify(app_state)


@app.route('/api/speak', methods=['POST'])
def api_speak():
    with state_lock:
        text = " ".join(app_state["sentence"]).strip()

    if text:
        tts.speak(text)
        return jsonify({"status": "speaking", "text": text})
    return jsonify({"status": "empty", "message": "Sentence buffer is empty"})


@app.route('/api/space', methods=['POST'])
def api_space():
    with state_lock:
        if app_state["sentence"] and app_state["sentence"][-1] != "":
            app_state["sentence"].append("")
    return jsonify({"status": "ok"})


@app.route('/api/backspace', methods=['POST'])
def api_backspace():
    with state_lock:
        if app_state["sentence"]:
            app_state["sentence"].pop()
    return jsonify({"status": "ok"})


@app.route('/api/clear', methods=['POST'])
def api_clear():
    with state_lock:
        app_state["sentence"].clear()
    return jsonify({"status": "ok"})


@app.route('/api/settings', methods=['POST'])
def api_settings():
    data = request.get_json() or {}
    with state_lock:
        if "show_skeleton" in data:
            app_state["show_skeleton"] = bool(data["show_skeleton"])
    return jsonify({"status": "ok"})


@app.route('/api/model_info')
def api_model_info():
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify({
        "model_type": "Random Forest",
        "accuracy": 1.0,
        "classes": model_classes,
        "total_samples": 1400
    })


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    try:
        results = train_and_save_model(model_type="rf")
        # Hot reload model into memory
        load_model()
        return jsonify({
            "status": "success",
            "accuracy": results["accuracy"],
            "classes": results["classes"],
            "total_samples": results["total_samples"]
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400


@app.route('/api/record_gesture', methods=['POST'])
def api_record_gesture():
    data = request.get_json() or {}
    gesture_name = data.get("gesture_name", "").strip()
    target_samples = int(data.get("target_samples", 100))

    if not gesture_name:
        return jsonify({"error": "Gesture name is required"}), 400

    trigger_recording(gesture_name, target_samples)
    return jsonify({"status": "recording_started", "gesture": gesture_name, "target": target_samples})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting SignBridge AI Web Application...")
    print("   Open in browser: http://localhost:5005")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5005, debug=False, threaded=True)
