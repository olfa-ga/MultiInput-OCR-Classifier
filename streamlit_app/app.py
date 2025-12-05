import streamlit as st
import requests
import base64

st.title("OCR Insurance ID Classifier")

insurance_type = st.selectbox("Insurance Type", ["home", "life", "auto", "health", "other"])
uploaded_file = st.file_uploader("Upload OCR Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convertir image en base64
    img_bytes = uploaded_file.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Envoyer la requête à l'API
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"insurance_type": insurance_type, "image_base64": img_base64}
    )

    st.write(response.json())
