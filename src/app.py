import streamlit as st

st.title("Mood Folder 😎")
st.write("Upload your selfie and let us read your mood!")

uploaded_files = st.file_uploader("Upload your selfies", accept_multiple_files=True)

