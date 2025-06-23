# 😄 Mood-Detection

**Mood-Detection** is a simple image classification web app that detects human emotions from images using a lightweight convolutional neural network.

It uses:
- 🧠 **PyTorch** and **Torchvision** for building and loading the model (currently a TinyVGG architecture)
- 🎛️ **Streamlit** for a clean and interactive web UI

---

## 🧠 Supported Moods

The model currently classifies input images into one of the following moods:
- 😠 Angry  
- 😢 Sad  
- 😐 Neutral  
- 😄 Happy  

---

## 🚀 How to Run the App

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Mood-Detection.git
cd Mood-Detection

# 2. Create and activate virtual environment
python -m virtualenv venv               # For Windows (or)
python3 -m venv venv                    # For macOS/Linux

# Activate virtual environment
venv\Scripts\activate                   # For Windows
source venv/bin/activate                # For macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
cd src
streamlit run app.py
