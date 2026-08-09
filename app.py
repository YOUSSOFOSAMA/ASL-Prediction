import streamlit as st
import numpy as np
import pickle
import urllib.request
import os
from PIL import Image

st.set_page_config(page_title="ASL AI Predictor", page_icon="🤟", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .main-title {
        text-align: center; font-family: 'Inter', sans-serif;
        font-size: 3.5rem; font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px; padding-bottom: 10px;
    }
    .subtitle { text-align: center; color: #A0AEC0; font-size: 1.2rem; margin-bottom: 2rem; }
    .word-box {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1);
        border-left: 5px solid #00C9FF; padding: 30px; border-radius: 12px;
        font-size: 2.5rem; font-weight: bold; text-align: center;
        letter-spacing: 8px; margin-top: 10px; margin-bottom: 30px;
        min-height: 100px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); color: #92FE9D;
    }
    button { color: #FAFAFA !important; }
    .stButton>button {
        border-radius: 8px !important; font-weight: bold !important;
        transition: all 0.3s ease !important;
        background-color: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }
    .stButton>button:hover {
        background-color: rgba(0,201,255,0.2) !important;
        border-color: #00C9FF !important; color: #00C9FF !important;
        transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,201,255,0.3);
    }

    /* Make the camera "Take Photo" button always bright and obvious */
    [data-testid="stCameraInputButton"] {
        background: linear-gradient(135deg, #00C9FF, #92FE9D) !important;
        color: #0E1117 !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 20px rgba(0, 201, 255, 0.5) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="stCameraInputButton"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(0, 201, 255, 0.7) !important;
        filter: brightness(1.1) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🤟 ASL AI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time American Sign Language Translation</p>", unsafe_allow_html=True)


TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
TASK_MODEL_PATH = "hand_landmarker.task"

@st.cache_resource
def load_models():
    if not os.path.exists("asl_classifier.pkl"):
        raise FileNotFoundError("asl_classifier.pkl not found in the app directory.")
    if not os.path.exists("scaler.pkl"):
        raise FileNotFoundError("scaler.pkl not found in the app directory.")

    with open("asl_classifier.pkl", "rb") as f:
        clf, le = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    if not os.path.exists(TASK_MODEL_PATH):
        st.info("⬇️ Downloading MediaPipe hand landmarker model (~25 MB)…")
        urllib.request.urlretrieve(TASK_MODEL_URL, TASK_MODEL_PATH)

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(options)
    except ImportError as e:
        raise ImportError(
            f"MediaPipe Tasks API is not available in this environment. "
            f"Ensure mediapipe>=1.0.0 is installed. Details: {e}"
        )

    return clf, le, scaler, landmarker


try:
    clf, le, scaler, landmarker = load_models()
except FileNotFoundError as e:
    st.error(f"❌ Model file missing: {e}")
    st.stop()
except ImportError as e:
    st.error(f"❌ MediaPipe import error: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error while loading models: {type(e).__name__}: {e}")
    st.stop()


def normalize_landmarks(landmark_list):
    """
    Wrist-relative + bounding-box scale normalisation.

    landmark_list: list of (x, y, z) tuples — 21 points from MediaPipe.

    Pipeline
    --------
    1. Translate so wrist (index 0) is the origin.
    2. Divide by max(x_range, y_range) of the translated points.
    3. Flatten to a 63-element feature vector.

    This matches the normalization used in mediapipeee.py during training.
    """
    wx, wy, wz = landmark_list[0]
    rel = [(x - wx, y - wy, z - wz) for x, y, z in landmark_list]

    xs = [p[0] for p in rel]
    ys = [p[1] for p in rel]
    scale = max(max(xs) - min(xs), max(ys) - min(ys))
    if scale == 0:
        scale = 1.0

    features = []
    for x, y, z in rel:
        features.extend([x / scale, y / scale, z / scale])
    return features  # 63 values


def predict_from_image(pil_image: Image.Image):
    """Run the full inference pipeline on a PIL image."""
    import mediapipe as mp

    # Convert to RGB numpy array then to MediaPipe Image
    img_np = np.array(pil_image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

    detection = landmarker.detect(mp_image)

    if not detection.hand_landmarks:
        return None, "no_hand"

    raw_lms = detection.hand_landmarks[0]
    lm_list = [(lm.x, lm.y, lm.z) for lm in raw_lms]

    features = normalize_landmarks(lm_list)
    features_scaled = scaler.transform([features])
    prediction = clf.predict(features_scaled)
    label = le.inverse_transform(prediction)[0]
    return label, "ok"


if "current_word" not in st.session_state:
    st.session_state.current_word = ""
if "processed_image_id" not in st.session_state:
    st.session_state.processed_image_id = None


col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("### 📷 Camera Input")
    st.info("Pose your hand and click **Take Photo** to predict a letter.")
    camera_image = st.camera_input("Capture ASL Pose", label_visibility="collapsed")

with col2:
    st.markdown("### 🔠 Word Builder")
    word_display = st.session_state.current_word if st.session_state.current_word else "..."
    st.markdown(f"<div class='word-box'>{word_display}</div>", unsafe_allow_html=True)

    c2a, c2b = st.columns(2)
    with c2a:
        if st.button("⏪ Undo Last Letter", use_container_width=True):
            st.session_state.current_word = st.session_state.current_word[:-1]
            st.rerun()
    with c2b:
        if st.button("🗑️ Clear Word", use_container_width=True):
            st.session_state.current_word = ""
            st.rerun()


if camera_image is not None:
    if camera_image.file_id != st.session_state.processed_image_id:
        st.session_state.processed_image_id = camera_image.file_id

        pil_img = Image.open(camera_image)

        with st.spinner("Analyzing hand pose…"):
            label, status = predict_from_image(pil_img)

        if status == "no_hand":
            st.error("⚠️ No hand detected in the frame. Please try again.")
        else:
            st.session_state.current_word += label
            st.toast(f"✅ Predicted: **{label}**")
            st.rerun()

st.markdown(
    "<hr><p style='text-align:center;color:gray;font-size:0.9rem;'>"
    "Powered by MediaPipe Tasks API & Scikit-Learn | Built for Streamlit Cloud"
    "</p>",
    unsafe_allow_html=True,
)
