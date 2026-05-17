import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress TF warnings & speed up init
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"   # hide TF logs

import streamlit as st

st.set_page_config(
    page_title="Potato Leaf Disease Detector",
    page_icon="🥔",
    layout="centered",
)

st.title("🥔 Potato Leaf Disease Detector")
st.markdown("Upload a potato leaf image to detect **Early Blight**, **Late Blight**, or **Healthy**.")
st.divider()

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]

DISEASE_INFO = {
    "Early Blight": {
        "description": "Caused by the fungus *Alternaria solani*. Dark brown spots with concentric rings appear on lower leaves.",
        "treatment":   "Apply fungicides (chlorothalonil or mancozeb), remove infected leaves, avoid overhead watering.",
        "color": "#FF6B35", "icon": "🟠",
    },
    "Late Blight": {
        "description": "Caused by *Phytophthora infestans*. Water-soaked lesions that rapidly turn dark brown.",
        "treatment":   "Use copper-based fungicides, destroy infected plants, plant resistant varieties.",
        "color": "#C0392B", "icon": "🔴",
    },
    "Healthy": {
        "description": "The leaf appears healthy with no signs of disease.",
        "treatment":   "No treatment needed. Maintain regular watering and fertilisation.",
        "color": "#27AE60", "icon": "🟢",
    },
}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model... please wait ⏳")
def load_model():
    import pickle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    st.success("✅ Model loaded!")
except FileNotFoundError:
    st.error("❌ model.pkl not found. Make sure it is in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ── Preprocess ────────────────────────────────────────────────────────────────
import numpy as np
from PIL import Image

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload a Leaf Image")
uploaded = st.file_uploader("Choose a JPG/PNG image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analysing leaf..."):
            try:
                arr   = preprocess(image)
                preds = model.predict(arr, verbose=0)[0]
                idx   = int(np.argmax(preds))
                label = CLASS_NAMES[idx]
                confs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.stop()

        info = DISEASE_INFO[label]
        st.markdown(
            f"<h2 style='color:{info['color']};'>{info['icon']} {label}</h2>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{confs[label]*100:.1f}%")
        st.progress(float(confs[label]))

    st.divider()
    st.subheader("📋 About this condition")
    st.info(info["description"])

    st.subheader("💊 Recommended Action")
    if label == "Healthy":
        st.success(info["treatment"])
    else:
        st.warning(info["treatment"])

    with st.expander("📊 All class probabilities"):
        for cls, prob in sorted(confs.items(), key=lambda x: -x[1]):
            st.markdown(f"**{DISEASE_INFO[cls]['icon']} {cls}** — {prob*100:.1f}%")
            st.progress(float(prob))

else:
    st.info("⬆️ Upload a potato leaf image above to get started.")

st.divider()
st.caption("CNN · PlantVillage Potato dataset · 224×224 input · 3-class softmax")