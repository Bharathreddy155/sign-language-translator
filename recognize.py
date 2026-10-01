"""
Sign Language Gesture Recognition - Standalone Real-Time Recognizer
Features:
- Real-time MediaPipe Hand Skeleton Overlay (Hardware Accelerated)
- High-confidence Debounce & Temporal Hold Sentence Builder
- Live Confidence Meter HUD
- Asynchronous Text-to-Speech Output
"""

import os
import sys
import time
import json
import joblib
import cv2
import numpy as np

from hand_detector import HandDetector
from preprocess import extract_raw_landmarks, normalize_landmarks, get_bounding_box
from text_to_speech import tts

MODEL_PATH = os.path.join("models", "gesture_model.pkl")
METADATA_PATH = os.path.join("models", "model_metadata.json")


class SignLanguageRecognizer:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Please train a model first using train_model.py!"
            )

        print(f"Loading gesture classification model from {model_path}...")
        self.model = joblib.load(model_path)
        self.classes = [str(c) for c in self.model.classes_]
        print(f"Loaded model with {len(self.classes)} gesture classes: {self.classes}")

        # Initialize Hand Detector
        self.detector = HandDetector()

        # Sentence Builder State
        self.sentence = []
        self.current_candidate = None
        self.consecutive_frames = 0
        self.required_frames = 15        # ~0.5 seconds of stable hold
        self.confidence_threshold = 0.75  # 75% min confidence
        self.last_committed_time = 0
        self.cooldown_seconds = 1.0      # cooldown after committing before same sign can repeat

    def update_sentence_builder(self, gesture, confidence):
        """
        Implements temporal debounce filtering:
        Appends gesture to sentence only if held stably with high confidence.
        """
        now = time.time()
        if confidence < self.confidence_threshold:
            self.consecutive_frames = 0
            self.current_candidate = None
            return False

        if gesture == self.current_candidate:
            self.consecutive_frames += 1
        else:
            self.current_candidate = gesture
            self.consecutive_frames = 1

        if self.consecutive_frames >= self.required_frames:
            # Check cooldown if same as last token
            if self.sentence and self.sentence[-1] == gesture and (now - self.last_committed_time) < self.cooldown_seconds:
                return False

            self.sentence.append(gesture)
            self.last_committed_time = now
            self.consecutive_frames = 0
            return True

        return False

    def get_sentence_text(self):
        return " ".join(self.sentence)

    def clear_sentence(self):
        self.sentence.clear()

    def backspace(self):
        if self.sentence:
            self.sentence.pop()

    def add_space(self):
        if self.sentence and self.sentence[-1] != "":
            self.sentence.append("")

    def speak_sentence(self):
        text = self.get_sentence_text()
        if text.strip():
            print(f"[TTS] Speaking: '{text}'")
            tts.speak(text)


def draw_hud(frame, current_gesture, confidence, recognizer, fps):
    """Draws a modern HUD dashboard with cards, meters, and sentence strip."""
    h, w, _ = frame.shape

    # 1. Top Header Card (Semi-transparent overlay)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 24, 30), -1)
    # Bottom Sentence Banner
    cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 2. Header Information
    cv2.putText(frame, "SIGN LANGUAGE TRANSLATOR", (30, 38), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 230, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # 3. Gesture & Confidence Meter
    if current_gesture and confidence > 0.0:
        conf_pct = int(confidence * 100)
        color = (0, 255, 128) if confidence >= 0.75 else (0, 200, 255)
        text_sign = f"SIGN: {current_gesture.upper()}"
        cv2.putText(frame, text_sign, (30, 85), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)

        # Confidence Bar
        bar_x = 350
        bar_y = 65
        bar_w = 200
        bar_h = 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"{conf_pct}%", (bar_x + bar_w + 12, bar_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Hold Progress indicator
        if recognizer.current_candidate == current_gesture:
            hold_ratio = min(1.0, recognizer.consecutive_frames / recognizer.required_frames)
            cv2.putText(frame, f"HOLD: {int(hold_ratio * 100)}%", (bar_x + bar_w + 80, bar_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 1)
    else:
        cv2.putText(frame, "NO HAND DETECTED", (30, 85), cv2.FONT_HERSHEY_DUPLEX, 0.9, (120, 120, 140), 2)

    # 4. Bottom Sentence Display
    sentence_str = recognizer.get_sentence_text()
    if not sentence_str:
        sentence_str = "<Hold a sign to append text...>"
        sent_color = (120, 120, 120)
    else:
        sent_color = (255, 255, 255)

    cv2.putText(frame, "SENTENCE:", (30, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 1)
    cv2.putText(frame, sentence_str, (30, h - 22), cv2.FONT_HERSHEY_DUPLEX, 0.85, sent_color, 2)

    # Controls legend
    legend = "[ENTER] Speak  |  [BKSP] Del  |  [C] Clear  |  [Q] Quit"
    cv2.putText(frame, legend, (w - 480, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)


def run_realtime_recognition(camera_id=0):
    try:
        recognizer = SignLanguageRecognizer()
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"\n[Error] Unable to access camera {camera_id}.")
        print("Please check macOS Camera Permissions:")
        print("  System Settings -> Privacy & Security -> Camera -> Enable for your Terminal/IDE.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_timer = time.time()
    fps = 30.0

    print("\n" + "=" * 60)
    print("🎥 Starting Real-Time OpenCV Gesture Recognition...")
    print("   Press [ENTER] to Speak Sentence")
    print("   Press [Q] or [ESC] to Exit")
    print("=" * 60 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        landmarks = recognizer.detector.detect(frame)

        current_gesture = None
        confidence = 0.0

        if landmarks is not None:
            # Draw glowing skeleton
            recognizer.detector.draw_skeleton(frame, landmarks)

            # Preprocess landmarks
            raw_pts = extract_raw_landmarks(landmarks)
            norm_vec = normalize_landmarks(raw_pts).reshape(1, -1)

            # Predict gesture
            probs = recognizer.model.predict_proba(norm_vec)[0]
            best_idx = np.argmax(probs)
            current_gesture = recognizer.classes[best_idx]
            confidence = float(probs[best_idx])

            # Draw bounding box around hand
            x1, y1, x2, y2 = get_bounding_box(landmarks, w, h)
            box_color = (0, 255, 128) if confidence >= 0.75 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame,
                f"{current_gesture} ({int(confidence*100)}%)",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2,
            )

            # Update Sentence Builder
            committed = recognizer.update_sentence_builder(current_gesture, confidence)
            if committed:
                print(f"[Committed] -> '{current_gesture}' | Full Sentence: {recognizer.get_sentence_text()}")
        else:
            recognizer.update_sentence_builder(None, 0.0)

        # Calculate FPS
        now = time.time()
        dt = now - fps_timer
        fps_timer = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Draw HUD
        draw_hud(frame, current_gesture, confidence, recognizer, fps)

        cv2.imshow("Sign Language Recognition System", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break
        elif key == 13:  # Enter -> Speak
            recognizer.speak_sentence()
        elif key == 8 or key == 127:  # Backspace
            recognizer.backspace()
        elif key == ord("c") or key == ord("C"):
            recognizer.clear_sentence()
        elif key == 32:  # Space
            recognizer.add_space()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_recognition()
