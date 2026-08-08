import streamlit as st
import numpy as np
import pickle
import mysql.connector
from PIL import Image

st.set_page_config(page_title="ASL AI Predictor", page_icon="🤟", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    .main-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #A0AEC0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .word-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-left: 5px solid #00C9FF;
        padding: 30px;
        border-radius: 12px;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        letter-spacing: 8px;
        margin-top: 10px;
        margin-bottom: 30px;
        min-height: 100px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: #92FE9D;
    }
    
    button {
        color: #FAFAFA !important;
    }
    .stButton>button, [data-testid="stCameraInputButton"] {
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .stButton>button:hover, [data-testid="stCameraInputButton"]:hover {
        background-color: rgba(0, 201, 255, 0.2) !important;
        border-color: #00C9FF !important;
        color: #00C9FF !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 201, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🤟 ASL AI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time American Sign Language Translation</p>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    with open("asl_classifier.pkl", "rb") as f:
        clf, le = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # mediapipe 0.10.35 on Python 3.14 does not auto-expose .solutions
    # Use the internal path directly as a guaranteed workaround
    import mediapipe.python.solutions.hands as _mp_hands
    hands = _mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    
    return clf, le, scaler, hands

try:
    clf, le, scaler, hands = load_models()
except Exception as e:
    st.error(f"Failed to load models. Ensure `asl_classifier.pkl` and `scaler.pkl` exist. Error: {e}")
    st.stop()

@st.cache_resource
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="ai_proj"
        )
        return conn
    except Exception:
        # Fails gracefully on Streamlit Cloud where localhost MySQL doesn't exist
        return None

conn = get_db_connection()

def save_to_db(image_array, label):
    if conn is not None:
        try:
            import io
            if conn.is_connected():
                cursor = conn.cursor()
                img = Image.fromarray(image_array)
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                image_bytes = buf.getvalue()
                sql = "INSERT INTO predictions (label, image) VALUES (%s, %s)"
                cursor.execute(sql, (label, image_bytes))
                conn.commit()
                cursor.close()
        except Exception as e:
            st.toast(f"Failed to save to database: {e}")

def normalize_landmarks(landmarks):
    lm_list = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    wx, wy, wz = lm_list[0]
    lm_list = [(x - wx, y - wy, z - wz) for x, y, z in lm_list]
    xs = [p[0] for p in lm_list]
    ys = [p[1] for p in lm_list]
    scale = max(max(xs) - min(xs), max(ys) - min(ys))
    if scale == 0: scale = 1
    features = []
    for x, y, z in lm_list:
        features.extend([x / scale, y / scale, z / scale])
    return features

if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'processed_image_id' not in st.session_state:
    st.session_state.processed_image_id = None

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("### 📷 Camera Input")
    st.info("Pose your hand and click **Take Photo** to predict a letter.")
    
    camera_image = st.camera_input("Capture ASL Pose", label_visibility="collapsed")

with col2:
    st.markdown("### 🔠 Word Builder")
    
    st.markdown(f"<div class='word-box'>{st.session_state.current_word if st.session_state.current_word else '...'}</div>", unsafe_allow_html=True)
    
    col2a, col2b = st.columns(2)
    with col2a:
        if st.button("⏪ Undo Last Letter", use_container_width=True):
            st.session_state.current_word = st.session_state.current_word[:-1]
            st.rerun()
    with col2b:
        if st.button("🗑️ Clear Word", use_container_width=True):
            st.session_state.current_word = ""
            st.rerun()

if camera_image is not None:
    if camera_image.file_id != st.session_state.processed_image_id:
        st.session_state.processed_image_id = camera_image.file_id
        
        image = Image.open(camera_image)
        image_np = np.array(image)
        
        with st.spinner("Analyzing hand pose..."):
            result = hands.process(image_np)
            
            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                features = normalize_landmarks(landmarks)
                
                features_scaled = scaler.transform([features])
                prediction = clf.predict(features_scaled)
                label = le.inverse_transform(prediction)[0]
                
                st.session_state.current_word += label
                
                save_to_db(image_np, label)
                
                st.toast(f"✅ Predicted: {label}")
                st.rerun()
            else:
                st.error("⚠️ No hand detected in the frame. Please try again.")

st.markdown("<hr><p style='text-align: center; color: gray; font-size: 0.9rem;'>Powered by MediaPipe & Scikit-Learn | Built for Streamlit Cloud</p>", unsafe_allow_html=True)
