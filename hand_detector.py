"""
Sign Language Gesture Recognition - Universal Hand Detector
Uses Google MediaPipe Tasks API (modern, hardware-accelerated for Apple Silicon).
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

TASK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

# Hand connections (21 joints)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckle bridge
    (5, 9), (9, 13), (13, 17)
]

# Color map for finger bones (BGR format)
FINGER_COLORS = [
    (0, 220, 255),  # Thumb - Gold/Yellow
    (0, 255, 128),  # Index - Emerald Green
    (255, 200, 0),  # Middle - Cyan/Light Blue
    (255, 120, 0),  # Ring - Blue
    (200, 0, 255),  # Pinky - Magenta
]


class HandDetector:
    def __init__(self, model_path=TASK_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Download it using: curl -L -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr_image):
        """
        Runs detection on a BGR OpenCV image.
        
        Returns:
            list of landmarks for the first detected hand, or None if no hand detected.
        """
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self.landmarker.detect(mp_image)

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            return result.hand_landmarks[0]
        return None

    def draw_skeleton(self, bgr_image, landmarks, color=(0, 255, 128)):
        """
        Draws glowing connections and joint circles on the image.
        """
        if not landmarks:
            return bgr_image

        h, w, _ = bgr_image.shape
        points = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))

        # Draw bones / connections
        for idx, (p1, p2) in enumerate(HAND_CONNECTIONS):
            # Pick finger color based on connection index
            f_color = color
            if idx < 4:
                f_color = FINGER_COLORS[0]
            elif idx < 8:
                f_color = FINGER_COLORS[1]
            elif idx < 12:
                f_color = FINGER_COLORS[2]
            elif idx < 16:
                f_color = FINGER_COLORS[3]
            elif idx < 20:
                f_color = FINGER_COLORS[4]
            else:
                f_color = (180, 180, 180)

            pt1 = points[p1]
            pt2 = points[p2]
            cv2.line(bgr_image, pt1, pt2, f_color, 2, cv2.LINE_AA)

        # Draw joint nodes
        for i, (px, py) in enumerate(points):
            # Bigger circles on fingertips and wrist
            radius = 5 if i in [0, 4, 8, 12, 16, 20] else 3
            node_color = (255, 255, 255) if i in [4, 8, 12, 16, 20] else (0, 230, 255)
            cv2.circle(bgr_image, (px, py), radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(bgr_image, (px, py), radius, node_color, -1, cv2.LINE_AA)

        return bgr_image
