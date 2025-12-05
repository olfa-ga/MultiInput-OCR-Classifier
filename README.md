# OCR Insurance ID Classifier

A multi-input OCR classifier that predicts the type of insurance (life, home, auto, health, other) based on scanned insurance ID images.  
Built with **PyTorch** for model training and **Streamlit** for an interactive web interface.

---

## 🚀 Features

- OCR image classification with a multi-input neural network
- Data augmentation to improve performance
- Trained model saved in `saved_models/ocr_model.pth`
- Streamlit interface for easy image upload and prediction
- Notebook demo for visualizing model predictions and dataset

---

Create a virtual environment :

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux / Mac


Upgrade pip and install dependencies:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

🏃 Running the Project
1. Train the Model 
python src/train.py

2. Launch the Streamlit App
python -m streamlit run streamlit_app/app.py


📈 Notebook Demo

Check out notebooks/Demo.ipynb for:

Loading and visualizing the dataset

Testing model predictions on sample images

Exploring performance metrics (Accuracy, F1-score, etc.)



