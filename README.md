# 🧾 OCR Insurance Code Classification  
### Multi-Input Deep Learning Model (PyTorch)

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.15-orange)
![CNN](https://img.shields.io/badge/CNN-implemented-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-app-lightblue)

This project implements a **multi-input OCR classifier** designed to recognize and classify characters from insurance-related documents.  
It combines **image features** and an additional **type vector** to improve prediction accuracy.

---

## 🎯 Overview

The model uses **two inputs**:

- **Grayscale Image** → processed through a CNN  
- **Type Vector** → auxiliary metadata describing the sample  

Both inputs are fused before classification, resulting in a **2-class prediction**.

---

## 📓 Demo Notebook

Use **OCR_Insurance_Demo.ipynb** to:

- Load the dataset  
- Visualize images and type vectors  
- Load the trained OCR model  
- Run predictions  
- Display real vs predicted labels  

---

## 🎯 Training

To train the model:

```bash
python src/train.py

---
## 🌐 Streamlit App

A simple interactive interface is provided. To run the app, use the following commands:

```bash
cd streamlit_app
streamlit run app.py

