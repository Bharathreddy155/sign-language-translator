"""
Sign Language Gesture Recognition - Starter Dataset Generator
Generates anatomically calibrated baseline 21-landmark poses for standard signs
(A, B, C, D, L, O, V, Y, Hello, Thank You, Yes, No, Help, I Love You)
with realistic 3D jitter, rotation, and hand-size variations.
Allows running and testing the entire recognition pipeline immediately!
"""

import os
import json
import numpy as np
from preprocess import normalize_landmarks

DATASET_DIR = "dataset"
DATASET_FILE = os.path.join(DATASET_DIR, "raw_landmarks.npz")
LABELS_FILE = os.path.join(DATASET_DIR, "gesture_labels.json")


def make_hand(wrist, thumb, index, middle, ring, pinky):
    """Assembles a 21x3 hand landmark array."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = wrist
    # Thumb (1..4)
    pts[1:5] = thumb
    # Index (5..8)
    pts[5:9] = index
    # Middle (9..12)
    pts[9:13] = middle
    # Ring (13..16)
    pts[13:17] = ring
    # Pinky (17..20)
    pts[17:21] = pinky
    return pts


def generate_canonical_gestures():
    """Defines baseline 21-landmark 3D coordinate templates for core signs."""
    wrist = [0.5, 0.8, 0.0]

    # Knuckle base positions (MCP joints 5, 9, 13, 17)
    imcp = [0.42, 0.58, 0.0]
    mmcp = [0.49, 0.55, 0.0]
    rmcp = [0.56, 0.58, 0.0]
    pmcp = [0.63, 0.62, 0.0]
    tmcp = [0.42, 0.72, 0.0]

    def straight_finger(mcp, length, angle_x=0.0):
        # 3 joints: PIP, DIP, TIP extending upward
        pip = [mcp[0] + angle_x * 0.3, mcp[1] - length * 0.35, 0.0]
        dip = [mcp[0] + angle_x * 0.7, mcp[1] - length * 0.70, 0.0]
        tip = [mcp[0] + angle_x * 1.0, mcp[1] - length * 1.00, 0.0]
        return [pip, dip, tip]

    def folded_finger(mcp, depth=-0.05):
        # 3 joints curled inward toward palm
        pip = [mcp[0], mcp[1] - 0.07, depth]
        dip = [mcp[0], mcp[1] - 0.02, depth * 1.5]
        tip = [mcp[0], mcp[1] + 0.04, depth]
        return [pip, dip, tip]

    def curved_finger(mcp, rad=0.08):
        pip = [mcp[0] - 0.03, mcp[1] - rad * 0.6, -0.04]
        dip = [mcp[0] - 0.06, mcp[1] - rad * 0.3, -0.06]
        tip = [mcp[0] - 0.08, mcp[1] + 0.01, -0.05]
        return [pip, dip, tip]

    # Thumb templates
    thumb_folded = [
        tmcp,
        [0.44, 0.66, -0.03],
        [0.48, 0.62, -0.04],
        [0.52, 0.60, -0.04],
    ]
    thumb_straight_up = [
        tmcp,
        [0.38, 0.64, 0.0],
        [0.34, 0.56, 0.0],
        [0.30, 0.48, 0.0],
    ]
    thumb_out = [
        tmcp,
        [0.36, 0.68, 0.0],
        [0.28, 0.65, 0.0],
        [0.20, 0.62, 0.0],
    ]

    gestures = {}

    # 1. 'A' - Fist with thumb vertical along index
    gestures["A"] = make_hand(
        wrist,
        [tmcp, [0.39, 0.64, 0.02], [0.38, 0.56, 0.03], [0.38, 0.48, 0.03]],
        [imcp] + folded_finger(imcp),
        [mmcp] + folded_finger(mmcp),
        [rmcp] + folded_finger(rmcp),
        [pmcp] + folded_finger(pmcp),
    )

    # 2. 'B' - Flat 4 straight fingers, thumb crossed over palm
    gestures["B"] = make_hand(
        wrist,
        thumb_folded,
        [imcp] + straight_finger(imcp, 0.30),
        [mmcp] + straight_finger(mmcp, 0.32),
        [rmcp] + straight_finger(rmcp, 0.30),
        [pmcp] + straight_finger(pmcp, 0.25),
    )

    # 3. 'C' - Hand curled like letter C
    gestures["C"] = make_hand(
        wrist,
        [tmcp, [0.37, 0.66, 0.02], [0.38, 0.58, 0.02], [0.42, 0.52, 0.01]],
        [imcp] + curved_finger(imcp),
        [mmcp] + curved_finger(mmcp),
        [rmcp] + curved_finger(rmcp),
        [pmcp] + curved_finger(pmcp),
    )

    # 4. 'D' - Index pointing up, other fingers forming circle with thumb
    gestures["D"] = make_hand(
        wrist,
        [tmcp, [0.45, 0.66, 0.0], [0.48, 0.60, 0.0], [0.50, 0.54, 0.0]],
        [imcp] + straight_finger(imcp, 0.32),
        [mmcp] + folded_finger(mmcp),
        [rmcp] + folded_finger(rmcp),
        [pmcp] + folded_finger(pmcp),
    )

    # 5. 'L' - Index up, Thumb out, rest folded
    gestures["L"] = make_hand(
        wrist,
        thumb_out,
        [imcp] + straight_finger(imcp, 0.32),
        [mmcp] + folded_finger(mmcp),
        [rmcp] + folded_finger(rmcp),
        [pmcp] + folded_finger(pmcp),
    )

    # 6. 'O' - All fingers curved touching thumb tip
    o_tip = [0.46, 0.50, -0.04]
    gestures["O"] = make_hand(
        wrist,
        [tmcp, [0.41, 0.64, -0.02], [0.43, 0.56, -0.03], o_tip],
        [imcp, [0.41, 0.50, -0.04], [0.43, 0.47, -0.05], o_tip],
        [mmcp, [0.48, 0.48, -0.04], [0.47, 0.47, -0.05], o_tip],
        [rmcp, [0.54, 0.50, -0.03], [0.51, 0.48, -0.04], o_tip],
        [pmcp, [0.60, 0.54, -0.02], [0.55, 0.51, -0.03], o_tip],
    )

    # 7. 'V' - Index & Middle open in V, ring & pinky folded
    gestures["V"] = make_hand(
        wrist,
        thumb_folded,
        [imcp] + straight_finger(imcp, 0.32, angle_x=-0.05),
        [mmcp] + straight_finger(mmcp, 0.32, angle_x=0.05),
        [rmcp] + folded_finger(rmcp),
        [pmcp] + folded_finger(pmcp),
    )

    # 8. 'Y' - Thumb out, Pinky out, middle 3 folded
    gestures["Y"] = make_hand(
        wrist,
        thumb_out,
        [imcp] + folded_finger(imcp),
        [mmcp] + folded_finger(mmcp),
        [rmcp] + folded_finger(rmcp),
        [pmcp] + straight_finger(pmcp, 0.28, angle_x=0.08),
    )

    # 9. 'Hello' - Open palm spread wide
    gestures["Hello"] = make_hand(
        wrist,
        thumb_straight_up,
        [imcp] + straight_finger(imcp, 0.33, angle_x=-0.04),
        [mmcp] + straight_finger(mmcp, 0.35, angle_x=0.0),
        [rmcp] + straight_finger(rmcp, 0.33, angle_x=0.03),
        [pmcp] + straight_finger(pmcp, 0.29, angle_x=0.07),
    )

    # 10. 'Thank You' - Flat palm tilted forward
    gestures["Thank You"] = make_hand(
        wrist,
        [tmcp, [0.43, 0.65, 0.05], [0.46, 0.58, 0.06], [0.48, 0.52, 0.06]],
        [imcp, [0.42, 0.48, 0.06], [0.42, 0.38, 0.08], [0.42, 0.28, 0.10]],
        [mmcp, [0.49, 0.46, 0.06], [0.49, 0.36, 0.08], [0.49, 0.26, 0.10]],
        [rmcp, [0.56, 0.48, 0.06], [0.56, 0.38, 0.08], [0.56, 0.28, 0.10]],
        [pmcp, [0.62, 0.52, 0.05], [0.62, 0.42, 0.07], [0.62, 0.32, 0.09]],
    )

    # 11. 'Yes' - Closed fist nodding pose
    gestures["Yes"] = make_hand(
        wrist,
        thumb_folded,
        [imcp] + folded_finger(imcp, depth=-0.08),
        [mmcp] + folded_finger(mmcp, depth=-0.08),
        [rmcp] + folded_finger(rmcp, depth=-0.08),
        [pmcp] + folded_finger(pmcp, depth=-0.08),
    )

    # 12. 'No' - Index + Middle snapping against Thumb
    gestures["No"] = make_hand(
        wrist,
        [tmcp, [0.43, 0.62, -0.01], [0.46, 0.56, -0.02], [0.48, 0.50, -0.02]],
        [imcp, [0.44, 0.52, -0.02], [0.46, 0.48, -0.02], [0.48, 0.50, -0.02]],
        [mmcp, [0.49, 0.51, -0.02], [0.49, 0.48, -0.02], [0.48, 0.50, -0.02]],
        [rmcp] + folded_finger(rmcp),
        [pmcp] + folded_finger(pmcp),
    )

    # 13. 'Help' - Thumbs up pose
    gestures["Help"] = make_hand(
        wrist,
        [tmcp, [0.40, 0.60, 0.0], [0.38, 0.50, 0.0], [0.38, 0.40, 0.0]],
        [imcp] + folded_finger(imcp),
        [mmcp] + folded_finger(mmcp),
        [rmcp] + folded_finger(rmcp),
        [pmcp] + folded_finger(pmcp),
    )

    # 14. 'I Love You' - Thumb, Index, Pinky extended
    gestures["I Love You"] = make_hand(
        wrist,
        thumb_out,
        [imcp] + straight_finger(imcp, 0.32),
        [mmcp] + folded_finger(mmcp),
        [rmcp] + folded_finger(rmcp),
        [pmcp] + straight_finger(pmcp, 0.28, angle_x=0.06),
    )

    return gestures


def augment_landmarks(base_hand, num_samples=90, noise_std=0.015, rot_max_deg=12):
    """
    Applies 3D rotation, scaling, and joint jitter to simulate natural human signing variations.
    """
    samples = []
    for _ in range(num_samples):
        # 1. Scale variance (0.85x to 1.15x)
        scale = np.random.uniform(0.85, 1.15)
        hand = base_hand.copy() * scale

        # 2. Random 2D/3D rotation around wrist
        angle = np.radians(np.random.uniform(-rot_max_deg, rot_max_deg))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=np.float32)
        hand = (hand - hand[0]) @ rot_matrix + hand[0]

        # 3. Gaussian jitter on individual joints (hand tremor / camera noise)
        noise = np.random.normal(0, noise_std, hand.shape).astype(np.float32)
        # Keep wrist jitter very small
        noise[0] *= 0.2
        hand += noise

        # 4. Normalize to position- and scale-invariant 63-vector
        norm_features = normalize_landmarks(hand)
        samples.append(norm_features)

    return samples


def generate_and_save_dataset():
    """Generates synthetic dataset and saves to dataset/raw_landmarks.npz."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    canonical = generate_canonical_gestures()
    
    all_features = []
    all_labels = []

    print(f"Generating starter dataset for {len(canonical)} core gestures...")
    for label, base_hand in canonical.items():
        samples = augment_landmarks(base_hand, num_samples=100)
        all_features.extend(samples)
        all_labels.extend([label] * len(samples))
        print(f"  + Generated {len(samples)} calibrated samples for '{label}'")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)

    # Save to compressed npz
    np.savez_compressed(DATASET_FILE, X=X, y=y)

    # Save labels list
    classes = sorted(list(set(y)))
    with open(LABELS_FILE, "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    print(f"\n✓ Dataset created successfully: {len(X)} samples, {len(classes)} classes.")
    print(f"  Saved to: {DATASET_FILE}")
    return X, y


if __name__ == "__main__":
    generate_and_save_dataset()
