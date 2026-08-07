# -*- coding: utf-8 -*-
"""
Created on Mon May 19 21:13:06 2025

@author: omaro
"""

import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load features and labels
with open("hand_features.pkl", "rb") as f:
    X, y = pickle.load(f)

print(f"Loaded {len(X)} samples.")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ── Training pipeline ──────────────────────────────────────────────────────
# MediaPipe → normalize_landmarks (mediapipeee.py) → StandardScaler → SVM
# The scaler fitted here is saved to scaler.pkl and loaded by Predictionpy.py.
# Predictionpy.py applies normalize_landmarks first, then scaler.transform.

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM classifier
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)

# Evaluate
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n✅ Training Accuracy: {train_accuracy:.2%}")
print(f"✅ Testing Accuracy: {test_accuracy:.2%}")

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation='vertical', cmap='Blues')
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.show()

# Save the model and label encoder
with open("asl_classifier.pkl", "wb") as f:
    pickle.dump((clf, le), f)

# Save the scaler too
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model and scaler saved.")
