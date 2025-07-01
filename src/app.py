# ────────────────────────────── app.py ───────────────────────────────────────
import sys
from pathlib import Path
from typing import Tuple
import hashlib, json

import numpy as np
from PIL import Image
import torch
import joblib
import streamlit as st

# ─── Patch: PyTorch × Streamlit file-watcher clash ───────────────────────────
torch.classes.__path__ = []

# ─── Local project imports ───────────────────────────────────────────────────
sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from ml.schemas import TinyVGG_V1_3
from ml.preprocessing import eval_transform

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DL_MODEL_PATH  = ROOT / "ml" / "models" / "tinyVGG_v1_3.pth"
SVM_MODEL_PATH = ROOT / "ml" / "models" / "svm_best_model.pkl"
RF_MODEL_PATH  = ROOT / "ml" / "models" / "best_model_rf.pkl"
ENCODER_PATH   = ROOT / "ml" / "models" / "label_encoder.pkl"

LABELS = ["angry", "happy", "neutral", "sad"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SVM_DIMS, RF_DIMS = (64, 64), (64, 64)

# ─── Debug helpers ───────────────────────────────────────────────────────────
def _sha1(x: np.ndarray) -> str:
    return hashlib.sha1(x.view(np.uint8)).hexdigest()[:8]

def debug_ndarray(name: str, arr: np.ndarray):
    st.write(f"**{name}**")
    st.json({
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "sha1": _sha1(arr),
    })

def debug_tinyvgg(tensor: torch.Tensor, logits: torch.Tensor):
    st.subheader("🔍 TinyVGG Debug")
    np_tensor = tensor.cpu().numpy()
    debug_ndarray("Input tensor (NCHW)", np_tensor)
    st.write("**Logits**")
    st.json({LABELS[i]: float(v) for i, v in enumerate(logits.squeeze().tolist())})
    probs = torch.softmax(logits, 1).squeeze().tolist()
    st.write("**Soft-max probs**")
    st.json({LABELS[i]: round(p, 4) for i, p in enumerate(probs)})

def debug_svm(feats: np.ndarray, svm, encoder):
    st.subheader("🔍 SVM Debug")
    debug_ndarray("Raw feature vector", feats)
    st.json({
        "svm.n_features_in_": int(svm.n_features_in_),
        "Label classes": encoder.classes_.tolist(),
    })
    pred = encoder.inverse_transform(svm.predict(feats))[0]
    st.write("Prediction on raw features → **`%s`**" % pred)

def debug_rf(feats: np.ndarray, rf, encoder):
    st.subheader("🔍 Random-Forest Debug")
    debug_ndarray("Raw feature vector", feats)
    st.json({
        "rf.n_features_in_": int(rf.n_features_in_),
        "Label classes": encoder.classes_.tolist(),
    })
    pred = encoder.inverse_transform(rf.predict(feats))[0]
    st.write("Prediction on raw features → **`%s`**" % pred)

# ─── Cached loaders (no scaler / PCA) ────────────────────────────────────────
@st.cache_resource
def load_tinyvgg():
    ckpt = torch.load(DL_MODEL_PATH, map_location=DEVICE)
    model = TinyVGG_V1_3(1, [64, 128], len(LABELS))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()

@st.cache_resource
def load_svm():
    return joblib.load(SVM_MODEL_PATH)

@st.cache_resource
def load_rf():
    return joblib.load(RF_MODEL_PATH)

@st.cache_resource
def load_encoder():
    return joblib.load(ENCODER_PATH)

# ─── Pre-processing ----------------------------------------------------------
def preprocess(img: Image.Image, dims: Tuple[int, int], keep_rgb: bool = False) -> np.ndarray:
    w, h = dims
    img = img.resize((w, h))
    if keep_rgb:
        arr = np.array(img, dtype=np.float32) / 255.0  # RGB image → (H, W, 3)
    else:
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0  # Grayscale
    return arr.flatten()[None, :]  # shape: (1, D)


# ─── Streamlit UI ------------------------------------------------------------
st.title("Mood Folder 😎")
st.caption("Upload your selfie and let us read your mood!")

model_choice = st.selectbox(
    "Choose emotion-recognition model",
    ("TinyVGG (deep learning)", "SVM (classical)", "Random-Forest (classical)")
)
DEBUG = st.sidebar.checkbox("🔧 Debug mode")

uploaded = st.file_uploader("Upload a selfie (JPEG/PNG)", ["jpg", "jpeg", "png"])

if uploaded:
    st.image(Image.open(uploaded), caption=uploaded.name, use_container_width=True)

    if st.button("Submit"):
        st.write("Running prediction…")

        encoder = load_encoder()

        # ── TinyVGG branch ───────────────────────────────────────────────────
        if model_choice.startswith("TinyVGG"):
            model = load_tinyvgg()
            tensor = (
                eval_transform(Image.open(uploaded).convert("L"))
                .unsqueeze(0).to(DEVICE)
            )
            with torch.no_grad():
                logits = model(tensor)
            idx  = int(logits.argmax(1))
            conf = float(torch.softmax(logits, 1)[0, idx])
            label = LABELS[idx]

            if DEBUG:
                debug_tinyvgg(tensor.cpu(), logits.cpu())

        # ── SVM branch ───────────────────────────────────────────────────────
        elif model_choice.startswith("SVM"):
            svm = load_svm()
            feats = preprocess(Image.open(uploaded), SVM_DIMS)

            if feats.shape[1] != svm.n_features_in_:
                st.error(f"SVM expects {svm.n_features_in_} feats but got {feats.shape[1]}")
                st.stop()

            if DEBUG:
                debug_svm(feats, svm, encoder)

            idx  = int(svm.predict(feats)[0])
            conf = float(svm.predict_proba(feats)[0, idx]) if hasattr(svm,"predict_proba") else 1.0
            label = encoder.inverse_transform([idx])[0]

        # ── Random-Forest branch ─────────────────────────────────────────────
        else:
            rf = load_rf()
            feats = preprocess(Image.open(uploaded).convert("RGB"), RF_DIMS, keep_rgb=True)

            if feats.shape[1] != rf.n_features_in_:
                st.error(f"RF expects {rf.n_features_in_} feats but got {feats.shape[1]}")
                st.stop()

            if DEBUG:
                debug_rf(feats, rf, encoder)

            idx  = int(rf.predict(feats)[0])
            conf = float(rf.predict_proba(feats)[0, idx]) if hasattr(rf,"predict_proba") else 1.0
            label = encoder.inverse_transform([idx])[0]

        # ── Display result ───────────────────────────────────────────────────
        if label not in LABELS:
            LABELS.append(label)
        st.success(f"**{label}**  ({conf:.2%} confidence)")
