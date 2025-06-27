"""
Mood Folder 😎
Streamlit app that can classify facial emotion using:
• TinyVGG (.pth)
• SVM (.pkl + optional PCA)
• Random-Forest (.pkl + optional PCA)

Each classical model now carries its *own* image-resize and feature pipeline,
so shape-mismatch errors disappear.
"""
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import joblib
import streamlit as st

# ---- PATCH before Streamlit import ----
torch.classes.__path__ = []          # fix PyTorch × Streamlit file-watcher

# ---- Local imports ----
sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from ml.schemas import TinyVGG_V1_3
from ml.preprocessing import eval_transform

# ---- Paths & constants ----
ROOT = Path(__file__).resolve().parents[1]

DL_MODEL_PATH  = ROOT / "ml" / "models" / "tinyVGG_v1_3.pth"

SVM_MODEL_PATH = ROOT / "ml" / "models" / "svm_best_model.pkl"
SVM_SCALER_PATH = ROOT / "ml" / "models" / "scaler.pkl"         # 64×64
SVM_PCA_PATH    = ROOT / "ml" / "models" / "pca.pkl"

RF_MODEL_PATH  = ROOT / "ml" / "models" / "best_model_randomforest.pkl"
RF_SCALER_PATH = ROOT / "ml" / "models" / "rf_scaler.pkl"       # 128×96 — might not exist
RF_PCA_PATH    = ROOT / "ml" / "models" / "rf_pca.pkl"          # idem — optional

ENCODER_PATH   = ROOT / "ml" / "models" / "label_encoder.pkl"

LABELS = ['angry', 'happy', 'neutral', 'sad']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image shapes used during training
SVM_DIMS = (64, 64)    # (width, height)
RF_DIMS  = (128, 96)   # (width, height)

# ---- Cached model loaders ----
@st.cache_resource(show_spinner="Loading TinyVGG model…")
def load_dl_model():
    ckpt = torch.load(DL_MODEL_PATH, map_location=DEVICE)
    model = TinyVGG_V1_3(1, [64, 128], len(LABELS))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()

@st.cache_resource(show_spinner="Loading SVM pipeline…")
def load_svm_pipeline():
    svm    = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SVM_SCALER_PATH)
    try:           pca = joblib.load(SVM_PCA_PATH)
    except FileNotFoundError: pca = None
    encoder = joblib.load(ENCODER_PATH)
    return svm, scaler, pca, encoder

@st.cache_resource(show_spinner="Loading Random-Forest pipeline…")
def load_rf_pipeline():
    rf     = joblib.load(RF_MODEL_PATH)
    try:           scaler = joblib.load(RF_SCALER_PATH)
    except FileNotFoundError: scaler = None
    try:           pca    = joblib.load(RF_PCA_PATH)
    except FileNotFoundError: pca = None
    encoder = joblib.load(ENCODER_PATH)
    return rf, scaler, pca, encoder

# ---- Pre-processing helper ----
def preprocess(img: Image.Image, dims: Tuple[int, int]) -> np.ndarray:
    """Greyscale-resize-flatten → shape = (1, w*h)."""
    w, h = dims
    arr = img.convert("L").resize((w, h))
    return (np.array(arr, dtype=np.float32) / 255.0).flatten()[None, :]

# ---- UI ----
st.title("Mood Folder 😎")
st.caption("Upload your selfie and let us read your mood!")

model_choice = st.selectbox(
    "Choose emotion-recognition model",
    ("TinyVGG (deep learning)", "SVM (classical)", "Random-Forest (classical)"),
)

uploaded = st.file_uploader("Upload a selfie (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(Image.open(uploaded), caption=uploaded.name, use_container_width=True)

    if st.button("Submit"):
        st.write("Running prediction…")

        if model_choice.startswith("TinyVGG"):
            # ── Deep-learning branch ───────────────────────────────────────────
            model = load_dl_model()
            img_tensor = (
                eval_transform(Image.open(uploaded).convert("L"))
                .unsqueeze(0)
                .to(DEVICE)
            )
            with torch.no_grad():
                logits = model(img_tensor)
                idx  = int(logits.argmax(dim=1))
                conf = float(torch.softmax(logits, dim=1)[0, idx])
                label = LABELS[idx]

        elif model_choice.startswith("SVM"):
            # ── SVM branch (64×64px → 4096 features) ─────────────────────────
            svm, scaler, pca, encoder = load_svm_pipeline()

            feats = preprocess(Image.open(uploaded), SVM_DIMS)
            feats = scaler.transform(feats)
            if pca is not None:                           # SVM *always* used PCA here
                feats = pca.transform(feats)

            idx  = int(svm.predict(feats)[0])
            conf = float(svm.predict_proba(feats)[0, idx]) if hasattr(svm, "predict_proba") else 1.0
            label = encoder.inverse_transform([idx])[0]

        else:
            # ── Random-Forest branch (128×96px → 12288 features) ─────────────
            rf, scaler, pca, encoder = load_rf_pipeline()

            feats = preprocess(Image.open(uploaded), RF_DIMS)

            # Only apply scaler/PCA if they match this feature length
            if scaler is not None and scaler.n_features_in_ == feats.shape[1]:
                feats = scaler.transform(feats)
            if pca is not None and rf.n_features_in_ == pca.n_components_:
                feats = pca.transform(feats)

            if rf.n_features_in_ != feats.shape[1]:
                st.error(
                    f"Random-Forest expects {rf.n_features_in_} features, but you fed {feats.shape[1]}.\n"
                    "Make sure you exported the correct RF scaler/PCA—or simply retrain & export an RF *pipeline*."
                )
                st.stop()

            idx  = int(rf.predict(feats)[0])
            conf = float(rf.predict_proba(feats)[0, idx]) if hasattr(rf, "predict_proba") else 1.0
            label = encoder.inverse_transform([idx])[0]

        # ---- Harmonise label with global list ----
        if label in LABELS:
            idx = LABELS.index(label)
        else:
            LABELS.append(label)
            idx = LABELS.index(label)

        st.success(f"**{LABELS[idx]}**  ({conf:.2%} confidence)")
