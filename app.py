import streamlit as st
import cv2
import numpy as np
import joblib

st.title("Face Comparison PCA (Kecil vs Dewasa)")

pca = joblib.load("pca_model.pkl")
size = (100, 100)

def extract(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return pca.transform(img.flatten().reshape(1, -1))[0]

col1, col2 = st.columns(2)

with col1:
    img1 = st.file_uploader("Upload Foto 1 (Kecil)", type=["jpg", "png"])

with col2:
    img2 = st.file_uploader("Upload Foto 2 (Dewasa)", type=["jpg", "png"])

if img1 and img2:
    # tampilkan gambar asli
    st.subheader("Gambar yang dibandingkan")

    colA, colB = st.columns(2)

    with colA:
        st.image(img1, caption="Foto 1 (Anak)", use_container_width=True)

    with colB:
        st.image(img2, caption="Foto 2 (Dewasa)", use_container_width=True)

    # proses PCA
    f1 = extract(img1)
    f2 = extract(img2)

    distance = np.linalg.norm(f1 - f2)

    st.write("### Distance:", distance)

    threshold = 9000

    if distance < threshold:
        st.success("KESIMPULAN: ORANG YANG SAMA")
    else:
        st.error("KESIMPULAN: ORANG BERBEDA")