"""
Sign Language Gesture Recognition - Landmark Preprocessing
Provides position- and scale-invariant normalization for MediaPipe 21 Hand Landmarks.
Compatible with both MediaPipe Tasks API and legacy NormalizedLandmarkList.
"""

import numpy as np


def extract_raw_landmarks(hand_landmarks):
    """
    Extracts raw (x, y, z) coordinates from a hand landmarks container or list.
    
    Args:
        hand_landmarks: List of NormalizedLandmark or container with .landmark attribute.
        
    Returns:
        np.ndarray: Array of shape (21, 3) with (x, y, z) values.
    """
    lms = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    coords = []
    for lm in lms:
        coords.append([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(landmarks):
    """
    Normalizes 21 3D landmarks to be invariant to:
    1. Translation (hand position in frame) -> Wrist becomes origin (0, 0, 0)
    2. Scale (distance from camera) -> Hand span normalized to unit scale [0, 1]
    
    Args:
        landmarks: np.ndarray of shape (21, 3) or list of 21 (x, y, z) points.
        
    Returns:
        np.ndarray: Flattened 1D array of 63 normalized features.
    """
    pts = np.array(landmarks, dtype=np.float32)
    if pts.shape != (21, 3):
        raise ValueError(f"Expected landmarks of shape (21, 3), got {pts.shape}")

    # 1. Translation Invariance: Shift origin to wrist (landmark 0)
    wrist = pts[0].copy()
    shifted = pts - wrist

    # 2. Scale Invariance: Divide by max Euclidean distance from wrist
    distances = np.linalg.norm(shifted, axis=1)
    max_dist = np.max(distances)
    if max_dist < 1e-6:
        max_dist = 1.0

    normalized = shifted / max_dist
    return normalized.flatten()


def get_bounding_box(hand_landmarks, img_width, img_height, margin=25):
    """
    Computes a pixel bounding box around the hand with optional padding margin.
    
    Returns:
        (x_min, y_min, x_max, y_max)
    """
    lms = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    x_coords = [lm.x * img_width for lm in lms]
    y_coords = [lm.y * img_height for lm in lms]

    x_min = max(0, int(min(x_coords) - margin))
    y_min = max(0, int(min(y_coords) - margin))
    x_max = min(img_width, int(max(x_coords) + margin))
    y_max = min(img_height, int(max(y_coords) + margin))

    return x_min, y_min, x_max, y_max
