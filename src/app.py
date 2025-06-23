import sys
import streamlit as st
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.schemas import TinyVGG_V1
from PIL import Image
import torch
from pathlib import Path
from ml.preprocessing import eval_transform

model_path = Path.cwd() / ".." / "ml" / "models" / "tinyVGG_v1.pth"
labels = ['angry', 'happy', 'neutral', 'sad']
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    checkpoint = torch.load(model_path, map_location=device)
    model = TinyVGG_V1(1, 64, 4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

model = load_model()

st.title("Mood Folder 😎")
st.write("Upload your selfie and let us read your mood!")

uploaded_files = st.file_uploader("Upload your selfies", accept_multiple_files=True)

if len(uploaded_files) > 0:
    if len(uploaded_files) == 1:
        st.image(Image.open(uploaded_files[0]), caption="Uploaded image", use_column_width=False, width=150)
        
    if st.button("Submit"):
        st.write("Running prediction on the uploaded image...")

        for img_path in uploaded_files:
            image = Image.open(img_path).convert("L") # grayscale
            input_tensor = eval_transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                label = labels[pred_idx]
                confidence = torch.softmax(output, dim=1)[0][pred_idx].item()
                st.write(f"Prediction for {img_path.name}: {label} ({confidence:.2%} confidence)")

print("Uploaded files : ", uploaded_files)