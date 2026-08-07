# -*- coding: utf-8 -*-
"""
Created on Sun May 18 21:58:20 2025

@author: omaro
"""

import os
import cv2
import mediapipe as mp
import pickle

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ── Landmark normalization ──────────────────────────────────────────────────
# Translate so wrist (landmark 0) is the origin, then scale by the bounding
# box diagonal so features are translation‑ and scale‑invariant.
# This function MUST be identical in mediapipeee.py and Predictionpy.py.
def normalize_landmarks(landmarks):
    """Return a flat list of 63 normalized (x, y, z) landmark values."""
    lm_list = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

    # Translate: subtract wrist position (index 0)
    wx, wy, wz = lm_list[0]
    lm_list = [(x - wx, y - wy, z - wz) for x, y, z in lm_list]

    # Scale: divide by bounding-box size (max range across x and y)
    xs = [p[0] for p in lm_list]
    ys = [p[1] for p in lm_list]
    scale = max(max(xs) - min(xs), max(ys) - min(ys))
    if scale == 0:
        scale = 1  # guard against degenerate hand detections

    features = []
    for x, y, z in lm_list:
        features.extend([x / scale, y / scale, z / scale])
    return features


# Feature extractor function
def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        # Apply the same normalization used during inference
        return normalize_landmarks(landmarks)
    else:
        return None  # no hand found

# Paths and output containers
dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
X = []
y = []

# LIMIT images per letter for testing (optional)
for label in sorted(os.listdir(dataset_dir)):
    label_dir = os.path.join(dataset_dir, label)
    if not os.path.isdir(label_dir):
        continue

    print(f"\nProcessing label: '{label}'")
    image_count = 0
    total = len(os.listdir(label_dir))

    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        features = extract_hand_landmarks(img_path)

        if features:
            X.append(features)
            y.append(label)
            image_count += 1

            if image_count % 100 == 0:
                print(f"  Processed {image_count}/{total} images for '{label}'")

print(f"\n✅ Feature extraction complete: {len(X)} total images processed.")

# Save data
output_path = "hand_features.pkl"
with open(output_path, "wb") as f:
    pickle.dump((X, y), f)

print(f"✅ Features saved to {output_path}")
