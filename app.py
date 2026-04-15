import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

import gdown

# download model
gdown.download("https://drive.google.com/file/d/14G1IrDNtYPVXn4cxeqaf5Kus-3458W1m/view?usp=sharing", "unet_best.keras", quiet=False)
gdown.download("https://drive.google.com/file/d/1B0a_KJXVFp-zr2-yv5W8hybk9IYgE8qA/view?usp=sharing", "cnn_best.keras", quiet=False)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# =========================
# CUSTOM UI (DARK + STYLE)
# =========================
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    .main {
        background-color: #0e1117;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1f77b4, #6a5acd);
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    unet = tf.keras.models.load_model("unet_best.keras")
    cnn = tf.keras.models.load_model("cnn_best.keras")
    return unet, cnn

unet_model, cnn_model = load_models()

# =========================
# CONSTANTS
# =========================
SEG_IMG_SIZE = 256
RED_IMG_SIZE = 128
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

# =========================
# TITLE
# =========================
st.title("🧠 Brain Tumor Detection System")
st.caption("U-Net Segmentation + CNN Classification")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # =========================
    # LOAD IMAGE
    # =========================
    image = Image.open(uploaded_file).convert('L')
    img_np = np.array(image)

    # Smaller display
    st.subheader("📥 Input Image")
    st.image(img_np, width=250)

    # =========================
    # PREPROCESS (U-NET)
    # =========================
    img_unet = cv2.resize(img_np, (SEG_IMG_SIZE, SEG_IMG_SIZE))
    img_unet = img_unet / 255.0
    img_unet = np.expand_dims(img_unet, axis=-1)

    # =========================
    # U-NET SEGMENTATION
    # =========================
    mask = unet_model.predict(np.expand_dims(img_unet, axis=0), verbose=0)[0]
    mask = (mask > 0.5).astype(np.uint8)

    # =========================
    # OVERLAY (RED HIGHLIGHT)
    # =========================
    overlay = img_unet.squeeze().copy()
    overlay = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    overlay[mask.squeeze() == 1] = [255, 0, 0]

    # =========================
    # EXTRACTED TUMOR
    # =========================
    extracted = img_unet.squeeze() * mask.squeeze()

    # =========================
    # CNN CLASSIFICATION (ORIGINAL IMAGE)
    # =========================
    img_cnn = cv2.resize(img_unet, (RED_IMG_SIZE, RED_IMG_SIZE))
    img_cnn = np.expand_dims(img_cnn, axis=0)

    pred = cnn_model.predict(img_cnn, verbose=0)[0]
    class_idx = np.argmax(pred)
    confidence = pred[class_idx]
    pred_class = classes[class_idx]

    # =========================
    # DISPLAY RESULTS (2 COLUMN)
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Tumor Overlay")
        st.image(overlay, width=300)

    with col2:
        st.subheader("🧠 Extracted Tumor Region")
        st.image(extracted, width=300)

    # =========================
    # EMPTY MASK WARNING
    # =========================
    if np.sum(mask) == 0:
        st.warning("⚠️ No tumor region detected by segmentation model")

    # =========================
    # PREDICTION BOX
    # =========================
    st.markdown(f"""
        <div class="prediction-box">
            Prediction: {pred_class.upper()} <br>
            Confidence: {confidence:.2f}
        </div>
    """, unsafe_allow_html=True)
