# -*- coding: utf-8 -*-
"""
Created on Tue May 20 02:26:04 2025
@author: omaro
"""

import cv2
import mediapipe as mp
import pickle
import mysql.connector

# === Connect to MySQL (XAMPP) ===
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="ai_proj"
)
cursor = conn.cursor()

# === Load model, scaler, and label encoder ===
with open("asl_classifier.pkl", "rb") as f:
    clf, le = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# === Save to database ===
def save_to_db(image, label):
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()
    sql = "INSERT INTO predictions (label, image) VALUES (%s, %s)"
    cursor.execute(sql, (label, image_bytes))
    conn.commit()
    print(f"💾 Saved to database: Label = '{label}'")

# ── Landmark normalization ──────────────────────────────────────────────────
# Translate so wrist (landmark 0) is the origin, then scale by the bounding
# box size so features are translation‑ and scale‑invariant.
# MUST be identical to the function in mediapipeee.py (training).
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


# ── Feature extractor ─────────────────────────────────────────────────────
# Pipeline: MediaPipe → normalize_landmarks → scaler.transform → clf.predict
def extract_hand_landmarks_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        # Apply the same normalization used during training
        features = normalize_landmarks(landmarks)
        return features, landmarks
    return None, None

# === Start camera ===
cap = cv2.VideoCapture(0)
print("🔴 Controls: [Space] predict | [Backspace] undo | [Enter] clear word | [Esc] exit")

current_word = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_flipped = cv2.flip(frame, 1)
    display_frame = frame_flipped.copy()

    features, landmark_data = extract_hand_landmarks_from_frame(display_frame)

    # Draw hand landmarks if found
    if landmark_data:
        mp_drawing.draw_landmarks(display_frame, landmark_data, mp_hands.HAND_CONNECTIONS)

    # Show current accumulated word on frame
    cv2.putText(display_frame, f'Word: {current_word}', (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("ASL Predictor - Press Space to Capture | [Backspace] undo | [Enter] clear | [Esc] exit", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 8:  # BACKSPACE
        current_word = current_word[:-1]
    elif key == 13:  # ENTER
        current_word = ""
    elif key == 32 and features:  # SPACE
        features_scaled = scaler.transform([features])
        prediction = clf.predict(features_scaled)
        label = le.inverse_transform(prediction)[0]
        print(f"✅ Prediction: {label}")
        current_word += label

        # Show prediction on frame
        cv2.putText(display_frame, f'Prediction: {label}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Prediction", display_frame)
        cv2.waitKey(1500)  # Show result for 1.5 seconds

        # Save to database
        save_to_db(display_frame, label)

print(f"\n📝 Final Word Assembled: {current_word}")

cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
