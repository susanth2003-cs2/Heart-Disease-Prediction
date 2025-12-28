# Heart-Disease-Prediction
# ❤️ Heart Disease Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Flask](https://img.shields.io/badge/Flask-2.3.0-orange) ![License](https://img.shields.io/badge/License-MIT-green)

Predict the likelihood of heart disease in a patient using a machine learning model. This project includes an interactive web interface built with Flask, HTML, CSS, and JavaScript.

---

## 🔹 Project Overview
Heart Disease Prediction is a **classification ML project** where a pre-trained model predicts whether a patient is at risk of heart disease based on health parameters. This project demonstrates an end-to-end workflow:

1. Data preprocessing
2. Feature engineering
3. Model training and evaluation
4. Serialization using `pickle`
5. Deployment with Flask and an interactive web app

---

## 🔹 Dataset
- Based on the **UCI Heart Disease Dataset**
- Features used:
  - Age
  - Sex
  - Chest Pain Type
  - Resting Blood Pressure
  - Cholesterol
  - Fasting Blood Sugar
  - Resting ECG Results
  - Maximum Heart Rate Achieved
  - Exercise Induced Angina
  - ST Depression
  - Slope of ST Segment
  - Number of Major Vessels
  - Thalassemia
- Target: `0 = No Disease`, `1 = Disease`

---

## 🔹 Model Details
- **Algorithm:** Logistic Regression  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score  
- **Serialization:** `pickle` (`Heart_Disease.pkl`)  
- **Activation:** N/A (classification problem, not deep learning)

> You can replace the model with others like Random Forest, Decision Tree, or XGBoost for experimentation.

---

## 🔹 Features
- Real-time prediction through a web interface
- Easy-to-use input form for patient data
- Immediate result: risk of heart disease
- Fully documented Python and Flask backend

---

## 🔹 Tech Stack
- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Python, Flask  
- **ML Libraries:** Scikit-learn, Pandas, NumPy  
- **Model Storage:** Pickle (`Heart_Disease.pkl`)

---
## 🔹 Algorithms Tried
Multiple classification algorithms were tested and compared based on **accuracy** and **ROC-AUC score**:

| Algorithm               | Accuracy | ROC-AUC Score |
|-------------------------|---------|---------------|
| Logistic Regression     | 0.85    | 0.89          |
| Random Forest Classifier| 0.87    | 0.91          |
| Decision Tree Classifier| 0.82    | 0.84          |
| XGBoost Classifier      | 0.88    | 0.92          |
| Support Vector Machine  | 0.83    | 0.86          |

✅ **Best Model Selected:** **XGBoost Classifier** (Highest ROC-AUC: 0.92)  
- This model is saved as `Heart_Disease.pkl` using `pickle` for deployment.

---

## 🔹 Features
- Real-time prediction through a web interface
- Easy-to-use input form for patient data
- Immediate result: risk of heart disease
- Fully documented Python and Flask backend

---

## 🔹 Project Structurecd heart-disease-prediction
heart-disease-prediction/
├── templates/
│   └── index.html          # Frontend HTML page
├── static/
│   ├── style.css           # CSS file
│   └── script.js           # JS file
├── Heart_Disease.pkl       # Pre-trained ML model
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
