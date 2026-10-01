"""
Sign Language Gesture Recognition - Interactive Webcam Data Collector
Captures real-time hand landmarks via webcam, applies invariant normalization,
and saves samples to dataset/raw_landmarks.npz with class labels.
"""

import os
import sys
import time
import json
import argparse
import cv2
import numpy as np

from hand_detector import HandDetector
from preprocess import extract_raw_landmarks, normalize_landmarks, get_bounding_box

DATASET_DIR = "dataset"
DATASET_FILE = os.path.join(DATASET_DIR, "raw_landmarks.npz")
LABELS_FILE = os.path.join(DATASET_DIR, "gesture_labels.json")


def save_samples(new_features, gesture_name):
    """Appends new landmark samples to the dataset npz file."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    new_X = np.array(new_features, dtype=np.float32)
    new_y = np.array([gesture_name] * len(new_features))

    if os.path.exists(DATASET_FILE):
        existing = np.load(DATASET_FILE, allow_pickle=True)
        old_X = existing["X"]
        old_y = existing["y"]
        combined_X = np.vstack([old_X, new_X])
        combined_y = np.concatenate([old_y, new_y])
    else:
        combined_X = new_X
        combined_y = new_y

    np.savez_compressed(DATASET_FILE, X=combined_X, y=combined_y)

    classes = sorted(list(set(combined_y)))
    with open(LABELS_FILE, "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    print(f"\n[Saved] Appended {len(new_features)} samples for '{gesture_name}'.")
    print(f"        Total dataset size: {len(combined_X)} across {len(classes)} classes.")


def run_collector(gesture_name, target_samples=100, camera_id=0):
    detector = HandDetector()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {camera_id}.")
        print("Please check macOS Camera Permissions in System Settings -> Privacy & Security -> Camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 60)
    print(f"  COLLECTING DATA FOR GESTURE: '{gesture_name}'")
    print(f"  Target Samples: {target_samples}")
    print("  Controls:")
    print("    [SPACE] : Start 3-second countdown and begin recording")
    print("    [Q]     : Quit")
    print("=" * 60 + "\n")

    captured_features = []
    state = "READY"  # READY, COUNTDOWN, RECORDING, FINISHED
    countdown_start = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        landmarks = detector.detect(frame)
        hand_found = landmarks is not None
        raw_pts = None

        if hand_found:
            detector.draw_skeleton(frame, landmarks)
            x1, y1, x2, y2 = get_bounding_box(landmarks, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
            raw_pts = extract_raw_landmarks(landmarks)

        # State Machine
        if state == "COUNTDOWN":
            remaining = 3.0 - (time.time() - countdown_start)
            if remaining > 0:
                cv2.putText(
                    frame,
                    f"Get Ready: {int(np.ceil(remaining))}",
                    (w // 2 - 140, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 165, 255),
                    4,
                )
            else:
                state = "RECORDING"

        elif state == "RECORDING":
            if hand_found and raw_pts is not None:
                norm_vec = normalize_landmarks(raw_pts)
                captured_features.append(norm_vec)

            # Progress bar
            progress = len(captured_features) / target_samples
            bar_w = int(w * 0.6)
            bar_x = int(w * 0.2)
            bar_y = h - 60

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 24), (40, 40, 40), -1)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + int(bar_w * min(1.0, progress)), bar_y + 24),
                (0, 220, 100),
                -1,
            )
            cv2.putText(
                frame,
                f"Recording '{gesture_name}': {len(captured_features)} / {target_samples}",
                (bar_x, bar_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 128),
                2,
            )

            if len(captured_features) >= target_samples:
                state = "FINISHED"

        elif state == "READY":
            cv2.putText(
                frame,
                f"Gesture: '{gesture_name}' - Press [SPACE] to Record",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Hand detected: {'YES' if hand_found else 'NO (show hand in camera)'}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if hand_found else (0, 0, 255),
                2,
            )

        elif state == "FINISHED":
            cv2.putText(
                frame,
                f"SUCCESS! {len(captured_features)} samples captured.",
                (w // 2 - 240, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 100),
                3,
            )
            cv2.putText(
                frame,
                "Press [S] to Save & Exit, or [Q] to Discard",
                (w // 2 - 220, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Hand Gesture Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            print("Data collection cancelled by user.")
            break
        elif key == 32 and state == "READY":  # Space
            countdown_start = time.time()
            state = "COUNTDOWN"
        elif key == ord("s") and state == "FINISHED":
            save_samples(captured_features, gesture_name)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Hand Gesture Landmark Dataset")
    parser.add_argument("--gesture", type=str, default=None, help="Name of gesture (e.g. 'Hello', 'A')")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to capture (default 100)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    g_name = args.gesture
    if not g_name:
        try:
            g_name = input("Enter the gesture name to record (e.g., 'Hello', 'A', 'Thank You'): ").strip()
        except EOFError:
            g_name = "Sample_Gesture"

    if not g_name:
        print("Error: Gesture name cannot be empty.")
        sys.exit(1)

    run_collector(g_name, target_samples=args.samples, camera_id=args.camera)
