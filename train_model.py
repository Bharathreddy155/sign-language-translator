"""
Sign Language Gesture Recognition - Model Training Engine
Loads normalized landmark datasets, trains Random Forest / MLP classifiers,
evaluates performance metrics, and saves model artifacts.
"""

import os
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

DATASET_DIR = "dataset"
DATASET_FILE = os.path.join(DATASET_DIR, "raw_landmarks.npz")
LABELS_FILE = os.path.join(DATASET_DIR, "gesture_labels.json")

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


def load_dataset():
    """Loads landmark features X and labels y from dataset file."""
    if not os.path.exists(DATASET_FILE):
        # Check for fallback CSV if npz does not exist
        csv_file = os.path.join(DATASET_DIR, "raw_landmarks.csv")
        if os.path.exists(csv_file):
            import csv
            X, y = [], []
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row:
                        continue
                    X.append([float(v) for v in row[:-1]])
                    y.append(row[-1])
            return np.array(X, dtype=np.float32), np.array(y)
        raise FileNotFoundError(f"Dataset not found at {DATASET_FILE} or {csv_file}")

    data = np.load(DATASET_FILE, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    return X, y


def train_and_save_model(model_type="rf", test_size=0.2, random_state=42):
    """
    Trains gesture classification model and saves it to disk.
    
    Args:
        model_type: 'rf' for RandomForest (recommended) or 'mlp' for Multi-Layer Perceptron.
        test_size: fraction of data to reserve for testing.
        random_state: random seed for reproducibility.
        
    Returns:
        dict: Training summary with accuracy, labels, and metrics.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("[1/4] Loading landmark dataset...")
    X, y = load_dataset()
    unique_labels = sorted(list(set(y)))
    print(f"      Total samples: {len(X)} across {len(unique_labels)} gestures: {unique_labels}")

    if len(unique_labels) < 2:
        raise ValueError(f"Need at least 2 distinct gesture classes to train. Found: {unique_labels}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Initialize model
    print(f"[2/4] Training {model_type.upper()} classifier...")
    if model_type.lower() == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=400,
            random_state=random_state,
            early_stopping=True
        )
    else:
        # Random Forest default - fast, robust to noise, excellent multiclass probabilities
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1
        )

    clf.fit(X_train, y_train)

    # Evaluate
    print("[3/4] Evaluating model performance...")
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=clf.classes_).tolist()

    print(f"      Validation Accuracy: {acc * 100:.2f}%")

    # Save model and metadata
    print(f"[4/4] Saving artifacts to {MODELS_DIR}/...")
    joblib.dump(clf, MODEL_PATH)

    metadata = {
        "model_type": model_type,
        "accuracy": acc,
        "total_samples": len(X),
        "classes": [str(c) for c in clf.classes_],
        "report": report,
        "confusion_matrix": conf_matrix,
        "features_dim": int(X.shape[1])
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # Also update gesture_labels.json in dataset/
    with open(LABELS_FILE, "w") as f:
        json.dump({"classes": [str(c) for c in clf.classes_]}, f, indent=2)

    print(f"✓ Model successfully saved to {MODEL_PATH}!")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Sign Language Gesture Model")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "mlp"], help="Model type: rf or mlp")
    args = parser.parse_args()

    try:
        results = train_and_save_model(model_type=args.model)
        print("\n=== Training Summary ===")
        print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
        print(f"Classes:  {', '.join(results['classes'])}")
    except Exception as e:
        print(f"Training failed: {e}")
