# 🧾 OCR Insurance Code Classification  
### Multi-Input Deep Learning Model (PyTorch)

![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![GitHub last commit](https://img.shields.io/github/last-commit/olfa-ga/MultiInput-OCR-Classifier)
![GitHub repo size](https://img.shields.io/github/repo-size/olfa-ga/MultiInput-OCR-Classifier)

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

